import json
from pathlib import Path

import pytest

from aimedicine.schemas import Checkpoint, Patient, Plan, PlanAction, latest
from aimedicine.simulation import ANTIBIOTICS, active_antibiotics, needs_revision, simulate


ROOT = Path(__file__).resolve().parents[1]


def patients():
    return [Patient.model_validate(p) for p in json.loads((ROOT / "data/patients.json").read_text())]


def scenarios():
    return json.loads((ROOT / "data/scenarios.json").read_text())


def plan(drug="ampicillin", drug_status="active", culture=True, culture_status="active", day=0):
    actions = [PlanAction(id="abx", catalog_id=f"{drug}_iv", title=drug, category="medication", drug_id=drug,
                          status=drug_status, indication="Synthetic test", evidence_ids=[])] if drug else []
    if culture:
        actions.append(PlanAction(id="culture", catalog_id="sputum_culture", title="Посев", category="investigation",
                                  status=culture_status, indication="Synthetic test", evidence_ids=[]))
    return Plan(version=1, day=day, actions=actions, rationale="Synthetic test", origin="fixture")


def test_three_patients_validate_and_have_no_future_payload():
    data = patients()
    assert len(data) == 3
    assert all(p.synthetic and p.culture is None for p in data)
    assert all(p.id.startswith("patient_") and all(o.day == 0 for o in p.observations) for p in data)
    for p in data:
        assert "susceptibility" not in p.model_dump_json()
        assert "scenario" not in p.model_dump()


def test_seed_reproducible_and_originals_immutable():
    patient = patients()[0]
    original = patient.model_dump_json()
    original_plan = plan()
    serialized_plan = original_plan.model_dump_json()
    first = simulate(patient, original_plan, scenarios()[patient.id], seed=19)
    second = simulate(patient, original_plan, scenarios()[patient.id], seed=19)
    assert first == second
    assert patient.model_dump_json() == original
    assert original_plan.model_dump_json() == serialized_plan
    first.patient.observations[0].value = 10
    assert patient.model_dump_json() == original


def test_effective_treatment_improves_observations():
    patient = patients()[0]
    checkpoint = simulate(patient, plan(), scenarios()[patient.id])
    assert latest(checkpoint.patient, "temperature").value < 37.2
    assert latest(checkpoint.patient, "crp").value < latest(patient, "crp").value
    assert checkpoint.events == []


@pytest.mark.parametrize("status", ["stopped", "blocked", "proposed"])
def test_nonactive_antibiotic_never_creates_recovery(status):
    patient = patients()[0]
    checkpoint = simulate(patient, plan(drug_status=status), scenarios()[patient.id])
    assert latest(checkpoint.patient, "temperature").value >= 38
    assert "no_active_antibiotic" in {e.kind for e in checkpoint.events}


def test_no_antibiotic_never_creates_recovery():
    patient = patients()[0]
    checkpoint = simulate(patient, plan(drug=None), scenarios()[patient.id])
    assert latest(checkpoint.patient, "crp").value >= latest(patient, "crp").value


@pytest.mark.parametrize("drug", sorted(ANTIBIOTICS))
def test_resistant_scenario_requires_review_for_actual_active_drug(drug):
    patient = patients()[2]
    checkpoint = simulate(patient, plan(drug=drug), scenarios()[patient.id])
    assert checkpoint.patient.culture.organism == "Haemophilus influenzae"
    assert checkpoint.patient.culture.available_day == 3
    assert checkpoint.patient.culture.susceptibility == {
        candidate: "R" if candidate == drug else "S" for candidate in ANTIBIOTICS
    }
    assert patient.culture is None
    assert latest(checkpoint.patient, "temperature").value >= 38
    assert {e.kind for e in checkpoint.events} == {"lack_of_response", "resistant_culture"}
    assert any("не прогноз" in assumption for assumption in checkpoint.assumptions)


def test_susceptible_actual_drug_changes_response_not_case_label():
    patient = patients()[2]
    fixed_scenario = {"day": 3, "culture": {"organism": "Haemophilus influenzae", "mode": "fixed",
                                            "susceptibility": {"ampicillin": "R", "amoxiclav": "S"}}}
    checkpoint = simulate(patient, plan(drug="amoxiclav"), fixed_scenario)
    assert latest(checkpoint.patient, "temperature").value < 37.2
    assert "resistant_culture" not in {e.kind for e in checkpoint.events}


@pytest.mark.parametrize("status", ["stopped", "blocked", "proposed"])
def test_reactive_culture_never_marks_inactive_drug_as_resistant(status):
    patient = patients()[2]
    checkpoint = simulate(patient, plan(drug_status=status), scenarios()[patient.id])
    assert set(checkpoint.patient.culture.susceptibility.values()) == {"S"}
    assert latest(checkpoint.patient, "temperature").value >= 38
    assert "no_active_antibiotic" in {event.kind for event in checkpoint.events}
    assert "resistant_culture" not in {event.kind for event in checkpoint.events}


def test_reactive_culture_without_antibiotic_does_not_create_recovery():
    patient = patients()[2]
    checkpoint = simulate(patient, plan(drug=None), scenarios()[patient.id])
    assert set(checkpoint.patient.culture.susceptibility.values()) == {"S"}
    assert latest(checkpoint.patient, "crp").value >= latest(patient, "crp").value
    assert "no_active_antibiotic" in {event.kind for event in checkpoint.events}


def test_reactive_culture_uses_all_and_only_active_antibiotics_without_mutating_scenario():
    patient = patients()[2]
    scenario = scenarios()[patient.id]
    before = json.dumps(scenario, sort_keys=True)
    treatment = plan(drug="amoxiclav")
    treatment.actions.extend([plan(drug="levofloxacin").actions[0], plan(drug_status="stopped").actions[0]])
    checkpoint = simulate(patient, treatment, scenario)
    assert checkpoint.patient.culture.susceptibility == {
        "ampicillin": "S", "amoxiclav": "R", "levofloxacin": "R"
    }
    assert json.dumps(scenario, sort_keys=True) == before


def test_unknown_culture_mode_fails_explicitly():
    with pytest.raises(ValueError, match="culture mode"):
        simulate(patients()[2], plan(), {"day": 3, "culture": {"mode": "typo"}})


@pytest.mark.parametrize("status", ["stopped", "blocked", "proposed"])
def test_culture_result_requires_active_prior_order(status):
    patient = patients()[2]
    checkpoint = simulate(patient, plan(culture_status=status), scenarios()[patient.id])
    assert checkpoint.patient.culture is None
    assert "resistant_culture" not in {e.kind for e in checkpoint.events}
    assert "lack_of_response" in {e.kind for e in checkpoint.events}


def test_culture_never_materializes_without_order():
    patient = patients()[2]
    checkpoint = simulate(patient, plan(culture=False), scenarios()[patient.id])
    assert checkpoint.patient.culture is None


def test_simulation_not_selected_by_patient_identifier():
    patient = patients()[0]
    arbitrary = patient.model_copy(update={"id": "unseen-patient"}, deep=True)
    scenario = scenarios()["patient_003"]
    one = simulate(patient, plan(), scenario)
    two = simulate(arbitrary, plan(), scenario)
    assert one.patient.observations == two.patient.observations
    assert one.events == two.events
    assert one.patient.culture == two.patient.culture


def test_review_is_reconstructed_from_observations_not_checkpoint_events():
    patient = patients()[2]
    checkpoint = simulate(patient, plan(), scenarios()[patient.id])
    expected = checkpoint.events.copy()
    checkpoint.events = []
    assert needs_revision(patient, checkpoint, plan()) == expected


def test_allergy_case_never_improves_on_contraindicated_beta_lactam():
    patient = patients()[1]
    checkpoint = simulate(patient, plan(), scenarios()[patient.id])
    assert latest(checkpoint.patient, "temperature").value >= 38
    assert "contraindicated_exposure" in {e.kind for e in checkpoint.events}
    alternative = simulate(patient, plan(drug="levofloxacin"), scenarios()[patient.id])
    assert latest(alternative.patient, "temperature").value < 37.2


@pytest.mark.parametrize("drug", ["ampicillin", "levofloxacin"])
def test_unknown_allergy_never_creates_safe_effective_response(drug):
    patient = patients()[0]
    patient.allergies = None
    checkpoint = simulate(patient, plan(drug=drug), scenarios()[patient.id])
    assert latest(checkpoint.patient, "temperature").value >= 38
    assert "allergy_information_missing" in {e.kind for e in checkpoint.events}


@pytest.mark.parametrize("allergy", ["levofloxacin", "fluoroquinolone"])
def test_explicit_drug_and_class_allergy_both_prevent_safe_response(allergy):
    patient = patients()[0]
    patient.allergies = [allergy]
    checkpoint = simulate(patient, plan(drug="levofloxacin"), scenarios()[patient.id])
    assert latest(checkpoint.patient, "temperature").value >= 38
    assert "contraindicated_exposure" in {e.kind for e in checkpoint.events}


def test_checkpoint_must_follow_plan():
    with pytest.raises(ValueError, match="after"):
        simulate(patients()[0], plan(day=3), {"day": 3})


def test_stopped_old_drug_excluded_from_actual_exposure():
    p = plan(drug="amoxiclav")
    p.actions.append(PlanAction(id="old", catalog_id="ampicillin_iv", title="Ампициллин", category="medication",
                                drug_id="ampicillin", status="stopped", indication="Replaced", evidence_ids=[]))
    assert active_antibiotics(p) == {"amoxiclav"}


def test_execution_log_collects_before_antibiotic_independent_of_action_list_order():
    patient = patients()[0]
    treatment = plan()
    assert treatment.actions[0].category == "medication"
    checkpoint = simulate(patient, treatment, scenarios()[patient.id])
    assert checkpoint.execution_log == [
        {"day": 0, "sequence": 1, "event": "sputum_collected", "action_id": "culture", "synthetic": True},
        {"day": 0, "sequence": 2, "event": "medication_started", "drug_id": "ampicillin", "action_id": "abx", "synthetic": True},
    ]
    assert checkpoint.patient.culture is None
    assert patient.consents == checkpoint.patient.consents == {}


@pytest.mark.parametrize("status", ["blocked", "stopped", "proposed"])
def test_execution_log_never_executes_nonactive_actions(status):
    patient = patients()[0]
    checkpoint = simulate(patient, plan(drug_status=status, culture_status=status), scenarios()[patient.id])
    assert checkpoint.execution_log == []


def test_execution_log_has_no_sample_without_culture_order():
    patient = patients()[2]
    checkpoint = simulate(patient, plan(culture=False), scenarios()[patient.id])
    assert [entry["event"] for entry in checkpoint.execution_log] == ["medication_started"]
    assert checkpoint.patient.culture is None


def test_execution_log_is_reproducible_and_records_only_current_active_drug():
    patient = patients()[2]
    treatment = plan(drug="amoxiclav")
    treatment.actions.append(PlanAction(id="old", catalog_id="ampicillin_iv", title="Ампициллин", category="medication",
                                        drug_id="ampicillin", status="stopped", indication="Replaced", evidence_ids=[]))
    first = simulate(patient, treatment, scenarios()[patient.id], seed=5)
    second = simulate(patient, treatment, scenarios()[patient.id], seed=5)
    assert first.execution_log == second.execution_log
    assert [entry["drug_id"] for entry in first.execution_log if entry["event"] == "medication_started"] == ["amoxiclav"]
    assert first.patient.culture.collected_day == first.execution_log[0]["day"]
    first.execution_log[0]["sequence"] = 99
    assert second.execution_log[0]["sequence"] == 1


def test_old_saved_checkpoint_without_execution_log_still_loads_as_unknown():
    patient = patients()[0]
    checkpoint = simulate(patient, plan(), scenarios()[patient.id])
    saved = checkpoint.model_dump()
    del saved["execution_log"]
    old = Checkpoint.model_validate(saved)
    assert old.execution_log == []
