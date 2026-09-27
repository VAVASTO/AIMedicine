from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aimedicine.data import load_catalog, load_chunks
from aimedicine.document_index import covered_reference_ids
from aimedicine.planner import fixture_selection, make_plan, materialize, public_patient
from aimedicine.retrieval import Retriever, patient_queries
from aimedicine.schemas import Culture, Observation, Patient, PlanSelection, ReplanSelection


ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def baseline():
    return Patient.model_validate(read("patients.json")[0])


def selection_for(action_id="ampicillin_iv"):
    item = next(x for x in load_catalog() if x["id"] == action_id)
    return PlanSelection(
        selected_action_ids=[action_id], rationale="Проверка структурного контракта",
        rationale_by_action={action_id: "Клиническое условие и источник указаны явно."},
        evidence_by_action={action_id: list(item["evidence_ids"])},
    )


def materialize_selection(selection, *, evidence=None):
    return materialize(selection, load_catalog(), load_chunks() if evidence is None else evidence,
                       day=0, version=1, origin="llm")


def patient_with_future_data():
    patient = baseline()
    patient.id = "secret-test-case-label"
    patient.observations.append(Observation(name="future_marker", value=987654,
                                            unit="synthetic", day=3))
    patient.culture = Culture(organism="future-test-organism", collected_day=0, available_day=3,
                              susceptibility={"ampicillin": "R", "amoxiclav": "S"})
    return patient


def test_omitted_allergies_remain_unknown_and_explicit_empty_remains_known():
    data = read("patients.json")[0]
    del data["allergies"]
    assert Patient.model_validate(data).allergies is None
    data["allergies"] = []
    assert Patient.model_validate(data).allergies == []


def test_public_patient_hides_future_results_and_identity_without_mutating_input():
    patient = patient_with_future_data()
    before = patient.model_dump()
    exposed = public_patient(patient, day=0)
    assert "id" not in exposed
    assert exposed["culture"] is None
    assert all(x["day"] <= 0 for x in exposed["observations"])
    serialized = json.dumps(exposed, ensure_ascii=False)
    for secret in ("secret-test-case-label", "future_marker", "987654", "future-test-organism"):
        assert secret not in serialized
    assert not {"scenario", "case_id", "expected", "expected_action"}.intersection(exposed)
    assert patient.model_dump() == before


def test_results_become_visible_exactly_at_their_available_day():
    patient = patient_with_future_data()
    assert public_patient(patient, 2)["culture"] is None
    exposed = public_patient(patient, 3)
    assert exposed["culture"]["organism"] == "future-test-organism"
    assert any(x["name"] == "future_marker" for x in exposed["observations"])


def test_scenario_cannot_be_smuggled_into_patient_schema():
    data = baseline().model_dump()
    data["scenario"] = {"expected_action": "amoxiclav_iv", "day": 3}
    with pytest.raises(ValidationError, match="scenario"):
        Patient.model_validate(data)


def test_retrieval_queries_do_not_reveal_unavailable_culture():
    patient = patient_with_future_data()
    assert "future-test-organism" not in " ".join(patient_queries(patient, 0))
    assert "future-test-organism" in " ".join(patient_queries(patient, 3))


def test_live_planner_payload_excludes_hidden_future_data():
    class CaptureClient:
        def __init__(self):
            self.payload = None

        def complete(self, role, system, payload, schema):
            assert role == "planner" and schema is PlanSelection
            self.payload = payload
            return selection_for()

    client = CaptureClient()
    plan, _ = make_plan(patient_with_future_data(), day=0, version=1, mode="live",
                        catalog=load_catalog(), retriever=Retriever(load_chunks()), client=client)
    assert plan.origin == "llm"
    serialized = json.dumps(client.payload, ensure_ascii=False)
    for secret in ("secret-test-case-label", "future_marker", "987654", "future-test-organism"):
        assert secret not in serialized
    assert client.payload["new_events"] == []
    assert client.payload["previous_plan"] is None


def test_live_model_failure_is_not_silently_replaced_with_fixture():
    class ProviderFailure(RuntimeError):
        pass

    class FailingClient:
        def complete(self, *args, **kwargs):
            raise ProviderFailure("Simulated unavailable provider")

    with pytest.raises(ProviderFailure):
        make_plan(baseline(), 0, 1, "live", load_catalog(), Retriever(load_chunks()), client=FailingClient())


def test_replanner_does_not_confuse_history_action_ids_with_catalog_ids():
    previous = materialize_selection(selection_for())

    class CaptureClient:
        def complete(self, role, system, payload, schema):
            assert role == "replanner" and schema is ReplanSelection
            assert all("id" not in action and "catalog_id" in action
                       for action in payload["previous_plan"]["actions"])
            return ReplanSelection.model_validate(selection_for("amoxiclav_iv").model_dump())

    plan, _ = make_plan(patient_with_future_data(), 3, 2, "live", load_catalog(),
                        Retriever(load_chunks()), CaptureClient(), previous)
    assert plan.actions[0].catalog_id == "amoxiclav_iv"
    assert plan.actions[0].id == "v2:amoxiclav_iv"


def test_replanner_schema_rejects_versioned_action_identifiers():
    data = selection_for().model_dump()
    data["selected_action_ids"] = ["v2:ampicillin_iv"]
    with pytest.raises(ValidationError):
        ReplanSelection.model_validate(data)


def test_citation_retry_contains_complete_requirements_for_the_selected_action():
    class CaptureClient:
        calls = 0

        def complete(self, role, system, payload, schema):
            self.calls += 1
            selection = selection_for()
            if self.calls == 1:
                selection.evidence_by_action["ampicillin_iv"].remove("cap-dose-ampicillin")
            else:
                assert role == "planner_validation_retry"
                assert payload["citation_requirements"]["required_evidence_by_action"] == selection.evidence_by_action
            return selection

    client = CaptureClient()
    plan, retrieval = make_plan(baseline(), 0, 1, "live", load_catalog(), Retriever(load_chunks()), client)
    assert client.calls == 2
    assert plan.actions[0].catalog_id == "ampicillin_iv"
    assert "validation_retry" in retrieval


def test_unknown_catalog_action_is_rejected():
    selection = selection_for()
    selection.selected_action_ids = ["unregistered_drug"]
    with pytest.raises(ValueError, match="Unknown catalog action"):
        materialize_selection(selection)


@pytest.mark.parametrize("invalid_evidence", [["invented-source"], ["cap-hydration"], []])
def test_missing_unknown_and_existing_but_unrelated_evidence_are_rejected(invalid_evidence):
    selection = selection_for()
    selection.evidence_by_action["ampicillin_iv"] = invalid_evidence
    with pytest.raises(ValueError, match="Invalid or unsupported evidence"):
        materialize_selection(selection)


def test_source_context_must_include_dosing_ground_even_when_choice_citation_is_valid():
    selection = selection_for()
    selection.evidence_by_action["ampicillin_iv"] = ["cap-empiric"]
    truncated_context = [c for c in load_chunks() if c["id"] != "cap-dose-ampicillin"]
    with pytest.raises(ValueError, match="Incomplete source context"):
        materialize_selection(selection, evidence=truncated_context)


@pytest.mark.parametrize("rationale", [None, "", "   "])
def test_missing_patient_specific_rationale_is_rejected(rationale):
    selection = selection_for()
    if rationale is None:
        del selection.rationale_by_action["ampicillin_iv"]
    else:
        selection.rationale_by_action["ampicillin_iv"] = rationale
    with pytest.raises(ValueError, match="Missing rationale"):
        materialize_selection(selection)


def test_duplicate_actions_are_rejected():
    selection = selection_for()
    selection.selected_action_ids.append("ampicillin_iv")
    with pytest.raises(ValueError, match="Duplicate selected actions"):
        materialize_selection(selection)


def test_related_additional_table_citation_is_retained():
    selection = selection_for()
    selection.evidence_by_action["ampicillin_iv"].append("cap-inpatient-table")
    plan = materialize_selection(selection)
    assert "cap-inpatient-table" in plan.actions[0].evidence_ids
    assert plan.actions[0].catalog_id == "ampicillin_iv"


def test_related_citation_must_still_be_in_the_retrieved_context():
    selection = selection_for()
    selection.evidence_by_action["ampicillin_iv"].append("cap-inpatient-table")
    context = [chunk for chunk in load_chunks() if chunk["id"] != "cap-inpatient-table"]
    with pytest.raises(ValueError, match="Invalid or unsupported evidence"):
        materialize_selection(selection, evidence=context)


def test_missing_required_citation_is_not_silently_added_to_model_output():
    selection = selection_for()
    selection.evidence_by_action["ampicillin_iv"].remove("cap-dose-ampicillin")
    with pytest.raises(ValueError, match="Incomplete selected citations"):
        materialize_selection(selection)


def test_selection_cannot_supply_an_unchecked_dose():
    payload = selection_for().model_dump()
    payload["dose"] = "999 г"
    with pytest.raises(ValidationError, match="dose"):
        PlanSelection.model_validate(payload)


@pytest.mark.parametrize("action_id", ["ampicillin_iv", "amoxiclav_iv", "levofloxacin_iv"])
def test_dose_route_frequency_and_evidence_are_bound_to_catalog_and_copied(action_id):
    catalog = load_catalog()
    source = next(x for x in catalog if x["id"] == action_id)
    source_before = deepcopy(source)
    result = materialize(selection_for(action_id), catalog, load_chunks(), 0, 1, "llm")
    action = result.actions[0]
    for field in ("drug_id", "dose", "route", "frequency", "evidence_ids"):
        assert getattr(action, field) == source[field]
    
    action.dose = "999 г"
    action.evidence_ids.append("invented-source")
    assert source == source_before


@pytest.mark.parametrize("patient_index", [0, 1, 2])
def test_offline_policy_choices_do_not_depend_on_case_identifier(patient_index):
    patient = Patient.model_validate(read("patients.json")[patient_index])
    renamed = patient.model_copy(deep=True)
    renamed.id = "completely-new-unseen-patient"
    first = fixture_selection(patient, 0, load_catalog())
    second = fixture_selection(renamed, 0, load_catalog())
    assert first.model_dump() == second.model_dump()


def test_offline_policy_uses_current_culture_only_when_result_is_available():
    patient = patient_with_future_data()
    before = fixture_selection(patient, 0, load_catalog()).selected_action_ids
    after = fixture_selection(patient, 3, load_catalog()).selected_action_ids
    assert "ampicillin_iv" in before and "amoxiclav_iv" not in before
    assert "amoxiclav_iv" in after and "ampicillin_iv" not in after


def test_lookup_rejects_unknown_ids_and_deduplicates_known_ids():
    retriever = Retriever(load_chunks())
    assert [x["id"] for x in retriever.lookup(["cap-ecg", "cap-ecg", "cap-crp"])] == ["cap-ecg", "cap-crp"]
    with pytest.raises(ValueError, match="Unknown evidence"):
        retriever.lookup(["invented-source"])


def test_gold_benchmark_is_deterministic_and_records_recall(tmp_path, record_property):

    chunks = load_chunks()
    retriever = Retriever(chunks)
    independent = Retriever(list(reversed(chunks)))
    rows = []
    for row in read("retrieval_gold.json"):
        actual = retriever.search(row["query"], k=5)
        repeated = retriever.search(row["query"], k=5)
        reindexed = independent.search(row["query"], k=5)
        comparable = lambda hits: [(x["id"], x["score"]) for x in hits]
        assert comparable(actual) == comparable(repeated) == comparable(reindexed)
        ids = [x["id"] for x in actual]
        assert len(ids) <= 5 and len(ids) == len(set(ids))
        assert set(ids) <= set(retriever.by_id)
        assert all(x["score"] > 0 for x in actual)
        expected = set(row["expected_ids"])
        covered = set(covered_reference_ids(actual))
        recall = len(expected.intersection(covered)) / len(expected)
        rows.append({**row, "actual_ids": ids, "covered_reference_ids": sorted(covered),
                     "recall_at_5": recall})
    macro_recall = sum(x["recall_at_5"] for x in rows) / len(rows)
    assert 0 <= macro_recall <= 1
    report = {"queries": rows, "macro_recall_at_5": macro_recall,
              "scope": "Reference-span coverage within the five automatic BM25 hits, without neighbor expansion."}
    (tmp_path / "retrieval_gold_measurement.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    record_property("retrieval_macro_recall_at_5", macro_recall)
