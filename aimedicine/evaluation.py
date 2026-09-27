from __future__ import annotations
from copy import deepcopy
import re

from .audit import audit_trace
from .data import ROOT, read_json, load_chunks, load_catalog, save_json
from .pipeline import run_case, conservative_repair, requires_review
from .retrieval import Retriever
from .reviewer import llm_review
from .schemas import Patient, PlanAction
from .finding_verifier import verify_findings, VERIFICATION_LIMITATION


def detected_wrong_dose(findings, action_id, *, version=1, status="open"):

    return any(
        f.code == "catalog_parameter_mismatch"
        and f.plan_version == version and f.action_id == action_id and f.status == status
        and f.category in ("clinical", "citation")
        and re.search(r"(?<![\d.,])999\s*г(?![а-яёa-z])", f.patient_evidence + " " + f.message, re.IGNORECASE)
        and "cap-dose-ampicillin" in f.evidence_ids
        for f in findings
    )


def wrong_dose_metrics(raw, accepted, unverified, rules, action_id, *, version=1):

    raw_llm = [finding for finding in raw if finding.origin == "llm"]

    def detected(findings, status):
        return detected_wrong_dose(findings, action_id, version=version, status=status)

    def critical(findings):
        return sum(f.status == "open" and f.severity == "critical" for f in findings)

    return {
        "raw_llm": {
            "target_open_detected": detected(raw_llm, "open"),
            "target_resolved_detected": detected(raw_llm, "resolved"),
            "open_critical": critical(raw_llm),
            "findings": [finding.model_dump() for finding in raw_llm],
        },
        "rule_verification": {
            "target_open_confirmed": detected(accepted, "open"),
            "target_resolved_confirmed": detected(accepted, "resolved"),
            "accepted_findings": len(accepted),
            "unverified_findings": len(unverified),
            "verified_open_critical": critical(accepted),
        },
        "rule_audit": {
            "target_open_detected": detected(rules, "open"),
            "target_resolved_detected": detected(rules, "resolved"),
            "open_critical": critical(rules),
        },
    }


PROCEDURAL_TARGETS = {
    "contrast_renal_precondition_missing": "demo-policy-renal-contrast",
    "procedure_consent_missing": "demo-policy-consent",
}


def procedural_metrics(raw, accepted, unverified, rules, action_id, *, version=1):
    raw_llm = [finding for finding in raw if finding.origin == "llm"]

    def matches(finding, code):
        return (finding.code == code and finding.plan_version == version
                and finding.action_id == action_id and finding.status == "open"
                and finding.category == "procedural"
                and PROCEDURAL_TARGETS[code] in finding.evidence_ids)

    targets = []
    for code in PROCEDURAL_TARGETS:
        raw_detected = any(matches(f, code) for f in raw_llm)
        confirmed = any(matches(f, code) for f in accepted)
        targets.append({"code": code, "plan_version": version, "action_id": action_id,
                        "raw_llm_detected": raw_detected, "rule_verified": confirmed,
                        "raw_llm_detected_and_verified": raw_detected and confirmed,
                        "rule_detected": any(matches(f, code) for f in rules)})
    return {
        "targets": targets,
        "raw_llm": {"detected": sum(target["raw_llm_detected"] for target in targets),
                    "total": len(targets), "findings": [f.model_dump() for f in raw_llm]},
        "rule_verification": {"confirmed": sum(target["rule_verified"] for target in targets),
                              "total": len(targets), "accepted_findings": len(accepted),
                              "unverified_findings": [f.model_dump() for f in unverified]},
        "rule_audit": {"detected": sum(target["rule_detected"] for target in targets),
                       "total": len(targets)},
        "other_rule_findings": [f.model_dump() for f in rules
                                if not any(matches(f, code) for code in PROCEDURAL_TARGETS)],
    }


def evaluate(output, mode="offline", client=None):
    chunks, catalog = load_chunks(), load_catalog()
    retriever = Retriever(chunks)
    patients = [Patient.model_validate(p) for p in read_json(ROOT / "data/patients.json")]
    scenarios = read_json(ROOT / "data/scenarios.json")
    base = run_case(patients[0], scenarios[patients[0].id], mode="offline")
    allergic = run_case(patients[1], scenarios[patients[1].id], mode="offline")
    reactive = run_case(patients[2], scenarios[patients[2].id], mode="offline")

    def snapshot(trace, last=False):
        t = trace.model_copy(deep=True)
        t.plans = [t.plans[-1] if last else t.plans[0]]
        t.audits = []
        if not last:
            t.checkpoints = []
        t.metadata["fault_injection"] = True
        t.metadata["base_plan_origin"] = "offline_reference_policy"
        return t

    def drug(trace):
        return next(a for a in trace.plans[-1].actions if a.category == "medication")

    tests = []
    wrong_dose = snapshot(base)
    drug(wrong_dose).dose = "999 г"
    tests.append(("wrong_dose", wrong_dose, "catalog_parameter_mismatch"))
    irrelevant = snapshot(base)
    drug(irrelevant).evidence_ids = ["cap-hydration"]
    tests.append(("irrelevant_citation", irrelevant, "citation_support_incomplete"))
    missing = snapshot(base)
    drug(missing).evidence_ids = ["does-not-exist"]
    tests.append(("invented_citation", missing, "citation_missing"))
    allergy = snapshot(base)
    allergy.patient.allergies = ["beta_lactam_immediate"]
    tests.append(("ignored_allergy", allergy, "beta_lactam_allergy"))
    interaction = snapshot(allergic)
    interaction.patient.current_medications = ["amiodarone"]
    tests.append(("drug_interaction", interaction, "levofloxacin_amiodarone"))
    unknown = snapshot(allergic)
    unknown.patient.observations = [o for o in unknown.patient.observations if o.name != "qtc_ms"]
    tests.append(("ordered_ecg_without_result", unknown, "qtc_unknown"))
    resistance = snapshot(reactive, last=True)
    original_drug = deepcopy(next(a for a in reactive.plans[0].actions if a.category == "medication"))
    original_drug.id = f"v{resistance.plans[0].version}:{original_drug.catalog_id}"
    resistance.plans[0].actions = [a for a in resistance.plans[0].actions if a.category != "medication"] + [original_drug]
    tests.append(("ignored_resistance", resistance, "known_culture_resistance"))
    procedure = snapshot(base)
    procedure.patient.observations = [o for o in procedure.patient.observations if o.name not in ("creatinine", "egfr")]
    procedure.plans[0].actions.append(PlanAction(id="v1:contrast_ct", catalog_id="contrast_ct", title="КТ с контрастом",
        category="procedure", status="active", indication="Намеренно внесённое действие без подтверждённых предварительных условий",
        evidence_ids=["demo-policy-renal-contrast", "demo-policy-consent"]))
    tests.append(("procedure_without_renal_result", procedure, "contrast_renal_precondition_missing"))
    tests.append(("procedure_without_consent", procedure.model_copy(deep=True), "procedure_consent_missing"))

    results = []
    for name, t, expected in tests:
        report = audit_trace(t, chunks, catalog)
        t.audits.append(report)
        found = [f.code for f in report.findings if f.status == "open"]
        results.append({"id": name, "expected": expected, "detected": expected in found, "found_codes": found})
        save_json(ROOT / output / f"fault_{name}.json", t.model_dump())
    controls = [{"id": t.patient.id, "open_findings": len([f for f in audit_trace(t, chunks, catalog).findings if f.status == "open"])}
                for t in (base, allergic, reactive)]

    live_example = None
    procedural_example = None
    if mode == "live":
        if client is None:
            raise ValueError("Live evaluation requires client")
        t = wrong_dose.model_copy(deep=True)
        t.mode = "live"
        t.audits = []
        rules = audit_trace(t, chunks, catalog)
        before, retrieval = llm_review(t, retriever, catalog, client)
        report = rules.model_copy(deep=True)
        accepted_before, rejected_before, checks_before = verify_findings(t, before.findings, chunks, catalog)
        report.findings += accepted_before
        report.unverified_findings = rejected_before
        report.llm_validation = checks_before
        report.llm_reviewed = True
        report.limitations.extend([*before.limitations, VERIFICATION_LIMITATION])
        report.llm_raw_summary = before.summary
        report.llm_summary = f"Подтверждено {len(accepted_before)} находок LLM; неподтверждено {len(rejected_before)}."
        t.audits.append(report)
        t.retrievals.append(retrieval)
        fixed = conservative_repair(t, rules, catalog)
        if fixed:
            t.plans.append(fixed)
        after, retrieval = llm_review(t, retriever, catalog, client)
        verification = audit_trace(t, chunks, catalog)
        after_rules = verification.findings.copy()
        accepted_after, rejected_after, checks_after = verify_findings(t, after.findings, chunks, catalog)
        verification.findings += accepted_after
        verification.unverified_findings = rejected_after
        verification.llm_validation = checks_after
        verification.llm_reviewed = True
        verification.limitations.extend([*after.limitations, VERIFICATION_LIMITATION])
        verification.llm_raw_summary = after.summary
        verification.llm_summary = f"После исправления: подтверждено {len(accepted_after)} находок LLM; неподтверждено {len(rejected_after)}."
        t.audits.append(verification)
        t.metadata["requires_human_review"] = requires_review(t)
        t.retrievals.append(retrieval)
        expected_action = drug(wrong_dose).id
        before_metrics = wrong_dose_metrics(before.findings, accepted_before, rejected_before,
                                           rules.findings, expected_action)
        after_metrics = wrong_dose_metrics(after.findings, accepted_after, rejected_after,
                                          after_rules, expected_action)
        llm_detected = before_metrics["raw_llm"]["target_open_detected"]
        live_example = {"llm_detected_wrong_dose": llm_detected,
                        "verified_wrong_dose_detected": llm_detected and before_metrics["rule_verification"]["target_open_confirmed"],
                        "evaluated_faults": 1,
                        "fault_id": "wrong_dose",
                        "note": "Raw LLM detection оценивает исходный ответ модели. Rule verification подтверждает цель и нормализует текст; её результат не засчитывается как самостоятельное обнаружение LLM. Процедурные цели проверяются в отдельном примере; остальные дефекты — только правилами.",
                        "before": before_metrics,
                        "after": after_metrics,
                        "repair_applied": fixed is not None,
                        "before_findings": len(before.findings),
                        "before_verified_findings": len(accepted_before),
                        "before_unverified_findings": len(rejected_before),
                        "after_open_critical": len([f for f in after.findings if f.status == "open" and f.severity == "critical"]),
                        "after_verified_open_critical": len([f for f in accepted_after if f.status == "open" and f.severity == "critical"]),
                        "after_unverified_findings": len(rejected_after),
                        "report": "fault_llm_before_after.json"}
        save_json(ROOT / output / "fault_llm_before_after.json", t.model_dump())

        procedural_trace = procedure.model_copy(deep=True)
        procedural_trace.mode = "live"
        procedural_trace.audits = []
        procedural_rules = audit_trace(procedural_trace, chunks, catalog)
        reviewed, retrieval = llm_review(procedural_trace, retriever, catalog, client)
        accepted, unverified, checks = verify_findings(procedural_trace, reviewed.findings, chunks, catalog)
        report = procedural_rules.model_copy(deep=True)
        report.findings.extend(accepted)
        report.unverified_findings = unverified
        report.llm_validation = checks
        report.llm_reviewed = True
        report.limitations.extend([*reviewed.limitations, VERIFICATION_LIMITATION])
        report.llm_raw_summary = reviewed.summary
        report.llm_summary = f"Процедурный пример: подтверждено {len(accepted)} находок LLM; неподтверждено {len(unverified)}."
        procedural_trace.audits.append(report)
        procedural_trace.retrievals.append(retrieval)
        procedural_trace.metadata["requires_human_review"] = requires_review(procedural_trace)
        procedural_example = procedural_metrics(reviewed.findings, accepted, unverified,
                                                 procedural_rules.findings, "v1:contrast_ct")
        procedural_example.update({
            "audited_traces": 1, "evaluated_faults": 2,
            "note": "Две процедурные цели в одном синтетическом плане проверены одним вызовом LLM. Это требования демонстрационных политик, а не универсальные противопоказания. Правила и верификация не засчитываются как raw-обнаружение модели; прочие rule-находки перечислены отдельно.",
            "report": "fault_llm_procedure.json",
        })
        save_json(ROOT / output / "fault_llm_procedure.json", procedural_trace.model_dump())

    gold = read_json(ROOT / "data/retrieval_gold.json")
    retrieval_results = []
    for row in gold:
        top5 = retriever.search(row["query"], 5)
        ids = [x["id"] for x in top5]
        covered = [reference["id"] for reference in retriever.covered_references(top5)]
        expected = set(row["expected_ids"])
        retrieval_results.append({**row, "actual_ids": ids,
                                  "covered_reference_ids": covered,
                                  "reference_recall_at_5": len(expected & set(covered)) / len(expected),
                                  "reference_hit_at_5": bool(expected & set(covered))})
    mean_reference_recall = sum(x["reference_recall_at_5"] for x in retrieval_results) / len(gold)
    reference_hit_rate = sum(x["reference_hit_at_5"] for x in retrieval_results) / len(gold)
    result = {
        "mode": mode, "fault_injection": True,
        "note": "Базовые планы для мутаций — явно обозначенные офлайн-эталоны. LLM-аудит выполняется независимо в live-примерах дозы и процедурных условий. Это малый демонстрационный набор, не оценка клинической надежности.",
        "rule_detection": {"detected": sum(x["detected"] for x in results), "total": len(results), "cases": results,
                           "evaluator": "deterministic_rules", "llm_evaluated": False},
        "clean_controls": controls, "llm_example": live_example,
        "llm_procedural_example": procedural_example,
        "retrieval": {"queries": len(gold), "reference_recall_at_5": mean_reference_recall,
                      "reference_hit_at_5": reference_hit_rate,
                      "mean_recall_at_5": mean_reference_recall, "hit_at_5": reference_hit_rate,
                      "note": "Покрытие эталонных фрагментов текстом только top-5 найденных чанков, без соседей. mean_recall_at_5 и hit_at_5 — совместимые псевдонимы reference-метрик; это не recall идентификаторов автоматического индекса.",
                      "results": retrieval_results},
        "usage": client.summary() if client else {"api_attempts": 0},
    }
    save_json(ROOT / output / "evaluation.json", result)
    return result
