from copy import deepcopy
import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from aimedicine.data import load_catalog, load_chunks
from aimedicine.document_index import covered_reference_ids
from aimedicine.planner import SYSTEM, build_context, make_plan, materialize
from aimedicine.retrieval import Retriever, patient_queries
from aimedicine.schemas import Patient, PlanSelection


ROOT = Path(__file__).resolve().parents[1]


def patient():
    return Patient.model_validate(json.loads((ROOT / "data/patients.json").read_text())[0])


def results(retriever, query):
    return [(hit["id"], hit["score"]) for hit in retriever.search(query, k=5)]


def test_empty_search_cannot_be_rescued_by_catalog_reference_lookup(monkeypatch):
    retriever = Retriever(load_chunks())
    assert all(reference in retriever.by_id for action in load_catalog() for reference in action["evidence_ids"])
    monkeypatch.setattr(retriever, "search", lambda *args, **kwargs: [])
    candidates, evidence, log = build_context(patient(), 0, load_catalog(), retriever)
    assert candidates == []
    assert evidence == []
    assert all(not search["hits"] for search in log["searches"])
    retrieved, references, retrieval_log = retriever.retrieve(["ампициллин дозировка"])
    assert retrieved == references == []
    assert retrieval_log["retrieved_ids"] == retrieval_log["covered_reference_ids"] == []


def test_manual_titles_and_tags_do_not_affect_search():
    original = load_chunks()
    poisoned = deepcopy(original)
    for chunk in poisoned:
        chunk["title"] = "qzxpoisonmarker"
        chunk["tags"] = ["qzxpoisonmarker"] * 50
    before, after = Retriever(original), Retriever(poisoned)
    for query in patient_queries(patient(), 0):
        assert results(before, query) == results(after, query)
    assert before.search("qzxpoisonmarker") == after.search("qzxpoisonmarker") == []
    assert all(chunk.get("origin") != "reference" for chunk in after.indexed_chunks)


@pytest.mark.parametrize("retain_reference_aliases", [False, True])
def test_missing_dose_page_disables_medications_even_with_neighbor_expansion(retain_reference_aliases):
    chunks = load_chunks()
    intact = Retriever(chunks)
    baseline, _, _ = build_context(patient(), 0, load_catalog(), intact)
    assert any(action["category"] == "medication" for action in baseline)
    filtered = [chunk for chunk in chunks if chunk.get("page") != 64
                or (retain_reference_aliases and chunk.get("origin") == "reference")]
    retriever = Retriever(filtered)
    assert all(chunk.get("page") != 64 or chunk.get("origin") == "reference"
               for chunk in retriever.by_id.values())
    candidates, evidence, _ = build_context(patient(), 0, load_catalog(), retriever)
    assert not any(action["category"] == "medication" for action in candidates)
    assert "cap-dose-scope" not in {chunk["id"] for chunk in evidence}
    retrieved, references, _ = retriever.retrieve(patient_queries(patient(), 0))
    assert not any(chunk.get("page") == 64 for chunk in retrieved)
    assert not any(reference["id"].startswith("cap-dose-") for reference in references)


def test_reference_aliases_need_delivered_source_intervals():
    retriever = Retriever(load_chunks())
    retrieved, references, log = retriever.retrieve(patient_queries(patient(), 0))
    assert references
    assert set(log["covered_reference_ids"]) == set(covered_reference_ids(retrieved))
    assert {reference["id"] for reference in references} == set(log["covered_reference_ids"])
    delivered = {chunk["id"]: chunk for chunk in retrieved}
    for mapping in log["reference_map"]:
        assert mapping["source_chunk_ids"]
        assert set(mapping["source_chunk_ids"]) <= set(delivered)
        source = [delivered[chunk_id] for chunk_id in mapping["source_chunk_ids"]]
        assert mapping["id"] in covered_reference_ids(source)


def test_live_payload_and_plan_citations_are_grounded_in_delivered_chunks():
    class CaptureClient:
        payload = None

        def complete(self, role, system, payload, schema):
            self.payload = deepcopy(payload)
            assert schema is PlanSelection
            choices = [action for action in payload["catalog"] if action["category"] != "medication"]
            medications = [action for action in payload["catalog"] if action["category"] == "medication"]
            choices += medications[:1]
            return PlanSelection(
                selected_action_ids=[action["id"] for action in choices],
                rationale="Проверка происхождения источников.",
                rationale_by_action={action["id"]: "Учтены данные пациента и доставленный источник."
                                     for action in choices},
                evidence_by_action={action["id"]: list(action["evidence_ids"]) for action in choices},
            )

    client = CaptureClient()
    source = Retriever(load_chunks())
    plan, log = make_plan(patient(), 0, 1, "live", load_catalog(), source, client=client)
    payload = client.payload
    assert plan.actions
    assert payload is not None
    delivered_ids = {item["id"] for item in payload["evidence"]}
    delivered = source.lookup(delivered_ids)
    assert all(chunk.get("origin") == "automatic" or chunk["source_type"] != "guideline"
               for chunk in delivered)
    assert not delivered_ids.intersection({chunk["id"] for chunk in source.chunks
                                          if chunk.get("origin") == "reference"})
    supported = set(covered_reference_ids(delivered)) | {
        chunk["id"] for chunk in delivered if chunk["source_type"] != "guideline"}
    for action in plan.actions:
        assert set(action.evidence_ids) <= supported
    assert "reference_map" in payload
    for mapping in payload["reference_map"]:
        assert set(mapping["source_chunk_ids"]) <= delivered_ids
        sources = [item for item in delivered if item["id"] in mapping["source_chunk_ids"]]
        assert mapping["id"] in covered_reference_ids(sources)
    assert set(log["retrieved_ids"]) == delivered_ids


def test_system_does_not_preselect_ampicillin():
    assert "предпочитай ампициллин" not in SYSTEM.casefold()
    assert "ампициллин" not in SYSTEM.casefold()
    assert "ampicillin" not in SYSTEM.casefold()


@pytest.mark.parametrize("field", ["command", "tool_calls", "execute", "system"])
def test_selection_schema_cannot_request_external_commands(field):
    value = {"selected_action_ids": [], "rationale": "Проверка схемы.",
             "rationale_by_action": {}, "evidence_by_action": {},
             field: "run-an-external-command"}
    with pytest.raises(ValidationError, match=field):
        PlanSelection.model_validate(value)


def test_unknown_action_from_model_cannot_bypass_the_catalog():
    selection = PlanSelection(
        selected_action_ids=["execute-external-command"], rationale="Проверка схемы.",
        rationale_by_action={"execute-external-command": "external instruction"},
        evidence_by_action={"execute-external-command": ["cap-dose-ampicillin"]},
    )
    with pytest.raises(ValueError, match="Unknown catalog action"):
        materialize(selection, load_catalog(), load_chunks(), day=0, version=1, origin="llm")
