import json
from pathlib import Path

from aimedicine.audit import audit_trace
from aimedicine.data import load_catalog, load_chunks
from aimedicine.finding_verifier import VERIFICATION_LIMITATION, verify_findings
from aimedicine.schemas import Finding, Patient, Plan, PlanAction, Trace
from aimedicine.simulation import simulate


ROOT = Path(__file__).resolve().parents[1]


def patient(index=0):
    return Patient.model_validate(json.loads((ROOT / "data/patients.json").read_text())[index])


def plan(drug="ampicillin_iv", day=0, version=1):
    actions = []
    for aid in (drug, "sputum_culture"):
        item = next(a for a in load_catalog() if a["id"] == aid)
        fields = {k: item.get(k) for k in ("title", "category", "drug_id", "dose", "route", "frequency", "evidence_ids")}
        actions.append(PlanAction(id=f"v{version}:{aid}", catalog_id=aid, status="active", indication="Unit test", **fields))
    return Plan(version=version, day=day, actions=actions, rationale="Unit test", origin="fixture")


def trace(p=None, plans=None, checkpoints=None):
    return Trace(patient=p or patient(), plans=plans or [plan()], checkpoints=checkpoints or [], mode="offline", run_id="verification-test")


def proposal(code="catalog_parameter_mismatch", version=1, action_id="v1:ampicillin_iv", evidence=None, **overrides):
    values = dict(code=code, severity="critical", category="citation", plan_version=version,
                  action_id=action_id, message="LLM allegation", evidence_ids=evidence or ["cap-dose-ampicillin"],
                  patient_evidence="Proposed factual support", recommendation="Review", origin="llm", status="open")
    values.update(overrides)
    return Finding(**values)


def verify(t, proposals):
    return verify_findings(t, proposals, load_chunks(), load_catalog())


def test_actual_wrong_dose_target_is_accepted_with_canonical_facts():
    t = trace()
    t.plans[0].actions[0].dose = "999 г"
    candidate = proposal(patient_evidence="Назначено 999 г", message="Доза противоречит таблице")
    accepted, unverified, records = verify(t, [candidate])
    assert len(accepted) == 1 and not unverified
    actual = next(f for f in audit_trace(t, load_chunks(), load_catalog()).findings if f.code == candidate.code)
    assert accepted[0].message == actual.message
    assert accepted[0].patient_evidence == actual.patient_evidence
    assert accepted[0].origin == "llm"
    assert records[0]["proposed"] == candidate.model_dump()
    assert records[0]["accepted"] is True


def test_correct_dose_does_not_become_wrong_from_llm_claim():
    candidate = proposal(message="Правильная доза ошибочна")
    accepted, unverified, records = verify(trace(), [candidate])
    assert accepted == [] and unverified == [candidate]
    assert records[0]["decision"] == "unverified"
    assert "Нет фактической находки" in records[0]["reason"]


def test_qtc_419_is_not_greater_than_450_even_if_model_says_so():
    t = trace(patient(1), [plan("levofloxacin_iv")])
    candidate = proposal(code="qt_risk", action_id="v1:levofloxacin_iv", evidence=["demo-policy-qtc-scope"],
                         message="QTc 419 превышает 450", patient_evidence="qtc_ms=419", category="uncertainty")
    accepted, unverified, records = verify(t, [candidate])
    assert accepted == [] and unverified == [candidate]
    assert records[0]["accepted"] is False


def test_performed_collection_before_drug_is_not_confirmed_as_missing_or_late():
    p = patient()
    initial = plan()
    checkpoint = simulate(p, initial, {"day": 3})
    assert [e["event"] for e in checkpoint.execution_log] == ["sputum_collected", "medication_started"]
    candidate = proposal(code="required_sputum_culture_missing", action_id=None,
                         evidence=["cap-sputum-before-antibiotics"], category="procedural",
                         message="Забор мокроты пропущен или выполнен после антибиотика")
    accepted, unverified, _ = verify(trace(p, [initial], [checkpoint]), [candidate])
    assert accepted == [] and unverified == [candidate]


def test_actual_qtc_outside_scope_keeps_canonical_severity_and_status():
    p = patient(1)
    next(o for o in p.observations if o.name == "qtc_ms").value = 470
    candidate = proposal(code="qt_risk", action_id="v1:levofloxacin_iv", evidence=["demo-policy-qtc-scope"],
                         severity="info", status="resolved")
    accepted, unverified, records = verify(trace(p, [plan("levofloxacin_iv")]), [candidate])
    assert not unverified
    assert accepted[0].severity == "critical" and accepted[0].status == "open"
    assert {"severity", "status"} <= set(records[0]["normalizations"])


def resistant_trace(fixed=False):
    p = patient(2)
    initial = plan()
    scenarios = json.loads((ROOT / "data/scenarios.json").read_text())
    checkpoint = simulate(p, initial, scenarios[p.id])
    plans = [initial, plan(day=3, version=2)]
    if fixed:
        plans.append(plan("amoxiclav_iv", day=3, version=3))
    return trace(p, plans, [checkpoint])


def test_future_resistance_cannot_be_used_against_day_zero_but_day_three_is_valid():
    t = resistant_trace()
    early = proposal(code="known_culture_resistance", evidence=["cap-culture-review"], category="clinical")
    current = early.model_copy(update={"plan_version": 2, "action_id": "v2:ampicillin_iv"})
    accepted, unverified, records = verify(t, [early, current])
    assert [f.plan_version for f in accepted] == [2]
    assert [f.plan_version for f in unverified] == [1]
    assert [r["accepted"] for r in records] == [False, True]


def test_historical_fixed_issue_cannot_be_reopened_by_model():
    candidate = proposal(code="known_culture_resistance", version=2, action_id="v2:ampicillin_iv",
                         evidence=["cap-culture-review"], category="clinical", status="open")
    accepted, unverified, records = verify(resistant_trace(fixed=True), [candidate])
    assert not unverified
    assert accepted[0].status == "resolved"
    assert records[0]["reference_status"] == "resolved"


def test_unknown_allergy_stays_unknown_and_critical_despite_model_self_resolution():
    p = patient()
    p.allergies = None
    candidate = proposal(code="allergy_unknown", evidence=["cap-beta-allergy"], severity="info",
                         status="resolved", category="uncertainty", message="Аллергии можно считать отсутствующими")
    t = trace(p)
    accepted, unverified, records = verify(t, [candidate])
    assert not unverified
    assert accepted[0].status == "open" and accepted[0].severity == "critical"
    assert accepted[0].patient_evidence == "allergies=null"
    assert "можно считать отсутствующими" not in accepted[0].message
    assert t.patient.allergies is None
    assert records[0]["proposed"]["message"] == candidate.message


def test_no_reference_match_cannot_be_autoapproved_as_safe():
    candidate = proposal(code="unverified_other", message="Возможен риск вне базы правил", evidence=["cap-empiric"])
    accepted, unverified, records = verify(trace(), [candidate])
    assert accepted == [] and unverified == [candidate]
    assert records[0]["coverage_limitation"] == VERIFICATION_LIMITATION
    assert "эксперт" in records[0]["reason"]


def test_same_code_on_wrong_action_is_not_a_match():
    t = trace()
    t.plans[0].actions[0].dose = "999 г"
    candidate = proposal(action_id="v1:sputum_culture")
    accepted, unverified, _ = verify(t, [candidate])
    assert accepted == [] and unverified == [candidate]


def test_same_code_on_wrong_version_is_not_a_match():
    t = trace()
    t.plans[0].actions[0].dose = "999 г"
    candidate = proposal(version=2)
    accepted, unverified, _ = verify(t, [candidate])
    assert accepted == [] and unverified == [candidate]


def test_real_but_unrelated_source_cannot_validate_a_matching_target():
    t = trace()
    t.plans[0].actions[0].dose = "999 г"
    candidate = proposal(evidence=["cap-hydration"])
    accepted, unverified, records = verify(t, [candidate])
    assert accepted == [] and unverified == [candidate]
    assert "не пересекаются" in records[0]["reason"]


def test_empty_source_list_cannot_validate_matching_target():
    t = trace()
    t.plans[0].actions[0].dose = "999 г"
    candidate = proposal()
    candidate.evidence_ids = []
    accepted, unverified, records = verify(t, [candidate])
    assert accepted == [] and unverified == [candidate]
    assert "не содержит источников" in records[0]["reason"]


def test_hallucinated_extra_source_is_preserved_as_unverified():
    t = trace()
    t.plans[0].actions[0].dose = "999 г"
    candidate = proposal(evidence=["cap-dose-ampicillin", "invented-reference"])
    accepted, unverified, records = verify(t, [candidate])
    assert accepted == [] and unverified == [candidate]
    assert "invented-reference" in records[0]["reason"]


def test_verification_never_modifies_trace_or_model_proposal():
    t = trace()
    t.plans[0].actions[0].dose = "999 г"
    candidate = proposal(severity="info", status="resolved")
    before_trace, before_candidate = t.model_dump_json(), candidate.model_dump_json()
    accepted, _, _ = verify(t, [candidate])
    accepted[0].message = "Changed after verification"
    assert t.model_dump_json() == before_trace
    assert candidate.model_dump_json() == before_candidate


def test_matching_rule_cannot_launder_invented_replacement_dose():
    t = trace()
    t.plans[0].actions[0].dose = "999 г"
    candidate = proposal(recommendation="Назначить вместо этого 777 г каждые 5 минут")
    accepted, _, records = verify(t, [candidate])
    assert "777" not in accepted[0].recommendation
    assert records[0]["proposed"]["recommendation"] == candidate.recommendation
    assert "recommendation" in records[0]["normalizations"]


def test_rejected_proposal_is_a_copy_and_every_proposal_has_decision_record():
    first, second = proposal(), proposal(code="unverified_other")
    accepted, unverified, records = verify(trace(), [first, second])
    assert not accepted and len(unverified) == len(records) == 2
    unverified[0].message = "Changed after verification"
    assert first.message == "LLM allegation"
    assert records[0]["proposed"]["message"] == "LLM allegation"


def test_empty_proposal_list_has_empty_outputs():
    assert verify(trace(), []) == ([], [], [])
