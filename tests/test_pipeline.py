import pytest
from pydantic import ValidationError

from aimedicine.data import ROOT, load_catalog, load_chunks, read_json
from aimedicine.schemas import Patient, Observation, Culture, Finding
from aimedicine.pipeline import run_case, conservative_repair, requires_review
from aimedicine.audit import audit_trace


@pytest.fixture
def cases():
    return [Patient.model_validate(x) for x in read_json(ROOT / "data/patients.json")]


def run(patient):
    return run_case(patient, read_json(ROOT / "data/scenarios.json")[patient.id])


def test_all_three_cases_follow_required_branches(cases):
    traces = [run(p) for p in cases]
    drugs = lambda plan: {a.drug_id for a in plan.actions if a.category == "medication" and a.status == "active"}
    assert drugs(traces[0].plans[0]) == drugs(traces[0].plans[-1]) == {"ampicillin"}
    assert drugs(traces[1].plans[0]) == {"levofloxacin"}
    assert drugs(traces[2].plans[0]) == {"ampicillin"}
    assert drugs(traces[2].plans[-1]) == {"amoxiclav"}
    assert traces[2].checkpoints[0].day == 3
    assert any(t["stage"] == "revision" for t in traces[2].transitions)
    for trace in traces:
        assert not [f for f in trace.audits[-1].findings if f.status == "open"]
        assert trace.retrievals
        assert not trace.metadata["requires_human_review"]


def test_original_state_and_plan_preserved_after_revision(cases):
    p = cases[2]
    before = p.model_dump()
    t = run(p)
    assert p.model_dump() == before
    assert t.patient.culture is None
    assert t.checkpoints[0].patient.culture is not None
    assert any(a.drug_id == "ampicillin" for a in t.plans[0].actions)
    assert not any(a.drug_id == "ampicillin" for a in t.plans[-1].actions)


def test_unknown_allergies_cannot_reach_active_exposure(cases):
    p = cases[0]
    p.allergies = None
    t = run(p)
    assert any(step["stage"] == "preflight_block_or_repair" for step in t.transitions)
    assert not any(a.category == "medication" and a.status == "active" for a in t.plans[-1].actions)
    assert t.metadata["requires_human_review"]
    assert t.patient.allergies is None


def test_repair_restores_exact_dose_without_rewriting_history(cases):
    t = run(cases[0])
    t.plans = [t.plans[0]]
    t.checkpoints = []
    medication = next(a for a in t.plans[0].actions if a.category == "medication")
    correct = medication.dose
    medication.dose = "999 г"
    report = audit_trace(t, load_chunks(), load_catalog())
    mismatch = next(f for f in report.findings if f.code == "catalog_parameter_mismatch")
    assert "999 г" in mismatch.patient_evidence and correct in mismatch.patient_evidence
    fixed = conservative_repair(t, report, load_catalog())
    assert next(a for a in fixed.actions if a.category == "medication").dose == correct
    assert medication.dose == "999 г"
    t.plans.append(fixed)
    verified = audit_trace(t, load_chunks(), load_catalog())
    assert any(f.code == "catalog_parameter_mismatch" and f.status == "resolved" for f in verified.findings)
    assert not [f for f in verified.findings if f.status == "open"]


@pytest.mark.parametrize("name,value,unit", [("spo2",101,"%"),("crp",-1,"mg/L"),("creatinine",80,"mg/dL"),("qtc_ms",420,"s"),("crp",float("nan"),"mg/L")])
def test_bad_observation_units_or_ranges_rejected(name,value,unit):
    with pytest.raises(ValidationError):
        Observation(name=name,value=value,unit=unit)


def test_culture_cannot_precede_sampling():
    with pytest.raises(ValidationError):
        Culture(organism="synthetic",collected_day=3,available_day=0,susceptibility={"ampicillin":"R"})


def test_blocked_action_never_becomes_false_clean(cases):
    t = run(cases[0])
    t.plans[-1].actions[-1].status = "blocked"
    t.plans[-1].missing_data = ["Требуется клиническая проверка"]
    assert requires_review(t)


def test_unverified_hypothesis_requires_expert_without_altering_plan(cases):
    t = run(cases[0])
    previous = t.plans[-1].model_dump()
    t.audits[-1].unverified_findings.append(Finding(
        code="unverified_other", severity="warning", category="clinical",
        plan_version=t.plans[-1].version, action_id=None, message="Вне покрытия правил",
        evidence_ids=["cap-empiric"], patient_evidence="Синтетический пример",
        recommendation="Проверить экспертом", origin="llm", status="open",
    ))
    assert requires_review(t)
    assert conservative_repair(t, t.audits[-1], load_catalog()) is None
    assert t.plans[-1].model_dump() == previous
