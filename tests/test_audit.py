import json
from pathlib import Path

import pytest

from aimedicine.audit import audit_trace, patient_at
from aimedicine.schemas import Patient, Plan, PlanAction, Trace
from aimedicine.simulation import simulate


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "data" / name).read_text())


def corpus():
    chunks = load("chunks.json")
    for filename in ("policies.json", "supplemental_chunks.json"):
        if (ROOT / "data" / filename).exists():
            chunks.extend(load(filename))
    return chunks


def catalog():
    return load("catalog.json")


def patient(index=0):
    return Patient.model_validate(load("patients.json")[index])


def action(catalog_id, status="active", suffix=""):
    item = next(a for a in catalog() if a["id"] == catalog_id)
    fields = {key: item.get(key) for key in ("title", "category", "drug_id", "dose", "route", "frequency", "evidence_ids")}
    return PlanAction(id=catalog_id + suffix, catalog_id=catalog_id, status=status, indication="Test", **fields)


def plan(drug="ampicillin_iv", day=0, version=1):
    return Plan(version=version, day=day, actions=[action(drug), action("sputum_culture"), action("reassess_72h")],
                rationale="Test plan", origin="fixture")


def trace(p=None, plans=None, checkpoints=None):
    return Trace(patient=p or patient(), plans=plans or [plan()], checkpoints=checkpoints or [], mode="offline", run_id="test")


def report(t):
    return audit_trace(t, corpus(), catalog())


def codes(t):
    return {f.code for f in report(t).findings if f.status == "open"}


def remove_observations(p, *names):
    p.observations = [o for o in p.observations if o.name not in names]


def test_clean_baseline_has_no_false_positives():
    assert report(trace()).findings == []


def test_clean_allergy_alternative_has_no_false_positives():
    assert report(trace(patient(1), [plan("levofloxacin_iv")])).findings == []
    assert patient(1).comorbidities == []
    assert patient(1).allergies == ["beta_lactam_immediate"]


@pytest.mark.parametrize("drug", ["ampicillin_iv", "amoxiclav_iv", "levofloxacin_iv"])
def test_diabetes_outside_demo_scope_even_without_mdr_risk_flag(drug):
    p = patient()
    p.comorbidities = ["Сахарный диабет 2 типа"]
    assert p.risk_resistant_pathogens is False
    finding = next(f for f in report(trace(p, [plan(drug)])).findings if f.code == "comorbidity_outside_demo_scope")
    assert finding.severity == "critical" and finding.category == "uncertainty"
    assert "demo-policy-comorbidity-scope" in finding.evidence_ids
    assert "не универсальное противопоказание" in finding.recommendation


def test_comorbidity_scope_still_reported_when_antibiotics_blocked():
    p = patient()
    p.comorbidities = ["Сахарный диабет 2 типа"]
    treatment = plan()
    treatment.actions[0].status = "blocked"
    assert "comorbidity_outside_demo_scope" in codes(trace(p, [treatment]))


@pytest.mark.parametrize("allergy", ["levofloxacin", "fluoroquinolone"])
def test_fluoroquinolone_allergy_has_its_own_supporting_source(allergy):
    p = patient()
    p.allergies = [allergy]
    finding = next(f for f in report(trace(p, [plan("levofloxacin_iv")])).findings if f.code == "fluoroquinolone_allergy")
    assert finding.evidence_ids == ["drug-levo-allergy"]


def test_beta_lactam_allergy_detected_without_dependence_on_id():
    p = patient(1)
    p.id = "unseen"
    assert "beta_lactam_allergy" in codes(trace(p))


def test_unknown_allergy_is_not_empty_list():
    p = patient()
    p.allergies = None
    assert "allergy_unknown" in codes(trace(p))
    p.allergies = []
    assert "allergy_unknown" not in codes(trace(p))


def test_missing_ecg_result_not_satisfied_by_ecg_order():
    p = patient(1)
    remove_observations(p, "qtc_ms")
    treatment = plan("levofloxacin_iv")
    treatment.actions.append(action("ecg"))
    assert "qtc_unknown" in codes(trace(p, [treatment]))


def test_future_qtc_result_cannot_satisfy_current_prerequisite():
    p = patient(1)
    next(o for o in p.observations if o.name == "qtc_ms").day = 3
    assert "qtc_unknown" in codes(trace(p, [plan("levofloxacin_iv")]))


@pytest.mark.parametrize("name", ["creatinine", "egfr"])
def test_missing_renal_result_not_satisfied_by_biochemistry_order(name):
    p = patient()
    remove_observations(p, name)
    treatment = plan()
    treatment.actions.append(action("biochemistry"))
    assert "renal_data_missing" in codes(trace(p, [treatment]))


def test_renal_boundary_is_marked_as_demonstration_scope():
    p = patient()
    next(o for o in p.observations if o.name == "egfr").value = 65
    finding = next(f for f in report(trace(p)).findings if f.code == "renal_outside_demo_scope")
    assert finding.category == "uncertainty"
    assert "demo-policy-dose-scope" in finding.evidence_ids
    assert "не универсальное" in finding.recommendation


def test_missing_hepatic_function_not_assumed_normal():
    p = patient()
    remove_observations(p, "alt")
    assert "hepatic_data_missing" in codes(trace(p))


def test_known_qt_interaction_detected():
    p = patient(1)
    p.current_medications = ["amiodarone"]
    findings = report(trace(p, [plan("levofloxacin_iv")])).findings
    assert any(f.code == "levofloxacin_amiodarone" and f.evidence_ids == ["drug-levo-qt"] for f in findings)


def test_missing_potassium_is_unknown_even_with_biochemistry_order():
    p = patient(1)
    remove_observations(p, "potassium")
    treatment = plan("levofloxacin_iv")
    treatment.actions.append(action("biochemistry"))
    assert "potassium_unknown" in codes(trace(p, [treatment]))


def test_low_potassium_requires_review_for_levofloxacin():
    p = patient(1)
    next(o for o in p.observations if o.name == "potassium").value = 3.1
    assert "hypokalemia_qt_risk" in codes(trace(p, [plan("levofloxacin_iv")]))


def test_qtc_outside_scope_has_explicit_policy_basis():
    p = patient(1)
    next(o for o in p.observations if o.name == "qtc_ms").value = 470
    finding = next(f for f in report(trace(p, [plan("levofloxacin_iv")])).findings if f.code == "qt_risk")
    assert finding.category == "uncertainty"
    assert "demo-policy-qtc-scope" in finding.evidence_ids


def test_stopped_amiodarone_not_counted_as_concurrent():
    p = patient(1)
    p.current_medications = ["amiodarone"]
    treatment = plan("levofloxacin_iv")
    treatment.actions.append(PlanAction(id="stop", catalog_id="external_amiodarone", title="Остановка амиодарона",
                                        category="medication", drug_id="amiodarone", status="stopped",
                                        indication="Synthetic cancellation", evidence_ids=[]))
    assert "levofloxacin_amiodarone" not in codes(trace(p, [treatment]))
    assert "amiodarone_washout_unknown" in codes(trace(p, [treatment]))


def test_stopped_antibiotic_not_duplicate_or_allergy_exposure():
    p = patient(1)
    treatment = plan("levofloxacin_iv")
    treatment.actions.append(action("ampicillin_iv", status="stopped"))
    actual = codes(trace(p, [treatment]))
    assert "duplicate_antibiotics" not in actual
    assert "beta_lactam_allergy" not in actual


def test_active_duplicate_antibiotics_detected():
    treatment = plan()
    treatment.actions.append(action("amoxiclav_iv"))
    assert "duplicate_antibiotics" in codes(trace(plans=[treatment]))


def test_omitting_all_antibiotics_is_not_clean_plan():
    treatment = plan()
    treatment.actions = [a for a in treatment.actions if a.category != "medication"]
    assert "antibiotic_missing" in codes(trace(plans=[treatment]))


def test_nonexistent_citation_detected():
    treatment = plan()
    treatment.actions[0].evidence_ids = ["fictional-guideline-3.2.7"]
    actual = codes(trace(plans=[treatment]))
    assert "citation_missing" in actual
    assert "citation_support_incomplete" in actual


def test_real_irrelevant_citation_does_not_support_medication():
    treatment = plan()
    treatment.actions[0].evidence_ids = ["cap-hydration"]
    actual = codes(trace(plans=[treatment]))
    assert "citation_missing" not in actual
    assert "citation_support_incomplete" in actual
    assert "citation_unrelated" in actual


def test_real_citation_cannot_justify_altered_dose():
    treatment = plan()
    treatment.actions[0].dose = "100 г"
    assert "catalog_parameter_mismatch" in codes(trace(plans=[treatment]))


def test_culture_is_time_aware_without_future_hindsight():
    p = patient(2)
    initial = plan()
    checkpoint = simulate(p, initial, load("scenarios.json")[p.id])
    unchanged = plan(day=3, version=2)
    audit = report(trace(p, [initial, unchanged], [checkpoint]))
    failures = [f for f in audit.findings if f.code == "known_culture_resistance"]
    assert [f.plan_version for f in failures] == [2]
    assert patient_at(trace(p, [initial], [checkpoint]), 0).culture is None


def test_improperly_attached_future_culture_still_ignored_before_available_day():
    p = patient(2)
    checkpoint = simulate(p, plan(), load("scenarios.json")[p.id])
    p.culture = checkpoint.patient.culture
    assert "known_culture_resistance" not in codes(trace(p))


def test_fixed_history_is_retained_as_resolved_and_clean_current_plan():
    p = patient(2)
    initial = plan()
    checkpoint = simulate(p, initial, load("scenarios.json")[p.id])
    flawed = plan(day=3, version=2)
    fixed = plan("amoxiclav_iv", day=3, version=3)
    t = trace(p, [initial, flawed, fixed], [checkpoint])
    original = t.model_dump_json()
    audit = report(t)
    assert t.model_dump_json() == original
    findings = [f for f in audit.findings if f.code == "known_culture_resistance"]
    assert len(findings) == 1
    assert findings[0].status == "resolved" and findings[0].plan_version == 2
    assert not any(f.status == "open" for f in audit.findings)


def test_repeated_flaw_remains_open_on_latest_plan():
    p = patient(1)
    audit = report(trace(p, [plan(version=1), plan(version=2)]))
    findings = [f for f in audit.findings if f.code == "beta_lactam_allergy"]
    assert [f.status for f in findings] == ["historical", "open"]


def test_missing_required_investigation_detected():
    p = patient()
    remove_observations(p, "wbc")
    assert "required_cbc_missing" in codes(trace(p))


def test_culture_order_not_mistaken_for_result():
    p = patient()
    assert p.culture is None
    report(trace(p))
    assert p.culture is None


def test_contrast_requires_existing_consent_and_renal_results():
    p = patient()
    remove_observations(p, "creatinine", "egfr")
    treatment = plan()
    treatment.actions.append(PlanAction(id="contrast", catalog_id="contrast_ct", title="КТ с контрастом",
                                        category="procedure", status="active", indication="Injected procedural test",
                                        evidence_ids=["demo-policy-consent", "demo-policy-renal-contrast"]))
    treatment.actions.append(action("biochemistry"))
    actual = codes(trace(p, [treatment]))
    assert "procedure_consent_missing" in actual
    assert "contrast_renal_precondition_missing" in actual
    assert p.consents == {}


@pytest.mark.parametrize("consent", [None, False])
def test_unknown_and_refused_consent_both_block_procedure(consent):
    p = patient()
    p.consents["thoracentesis"] = consent
    treatment = plan()
    treatment.actions.append(PlanAction(id="procedure", catalog_id="thoracentesis", title="Пункция",
                                        category="procedure", status="active", indication="Injected procedural test",
                                        evidence_ids=["demo-policy-consent"]))
    assert "procedure_consent_missing" in codes(trace(p, [treatment]))


def test_blocked_procedure_does_not_masquerade_as_execution():
    p = patient()
    treatment = plan()
    treatment.actions.append(PlanAction(id="procedure", catalog_id="thoracentesis", title="Пункция",
                                        category="procedure", status="blocked", indication="Needs consent",
                                        evidence_ids=["demo-policy-consent"]))
    assert "procedure_consent_missing" not in codes(trace(p, [treatment]))
    assert p.consents == {}
