import json

from aimedicine.data import ROOT, load_catalog, load_chunks, read_json
from aimedicine.reviewer import SYSTEM, build_review_payload, llm_review
from aimedicine.schemas import AuditReport, Event, LLMAudit, Observation, Patient, Plan, PlanAction, Trace
from aimedicine.simulation import simulate


def make_plan(drug="ampicillin_iv", day=0, version=1):
    actions = []
    for aid in (drug, "sputum_culture"):
        item = next(a for a in load_catalog() if a["id"] == aid)
        fields = {k: item.get(k) for k in ("title", "category", "drug_id", "dose", "route", "frequency", "evidence_ids")}
        actions.append(PlanAction(id=f"v{version}:{aid}", catalog_id=aid, status="active",
                                  indication="UNTRUSTED_PLANNER_EXPLANATION", **fields))
    return Plan(version=version, day=day, actions=actions, rationale="UNTRUSTED_PLANNER_CONCLUSION", origin="fixture")


def track():
    p = Patient.model_validate(read_json(ROOT / "data/patients.json")[2])
    first = make_plan()
    cp = simulate(p, first, read_json(ROOT / "data/scenarios.json")[p.id])
    return Trace(patient=p, plans=[first, make_plan("amoxiclav_iv", 3, 2)], checkpoints=[cp],
                 mode="offline", run_id="test-review")


def payload(t):
    return build_review_payload(t, load_catalog(), load_chunks())


def observations(state):
    return sorted((name, value, unit, day) for name, readings in state["observations"].items()
                  for value, unit, day in readings)


def test_every_clinical_field_and_observation_is_preserved_once_per_snapshot_day():
    t = track()
    result = payload(t)
    assert set(result["patient_states"]) == {"0", "3"}
    for day, patient in ((0, t.patient), (3, t.checkpoints[0].patient)):
        state = result["patient_states"][str(day)]
        assert observations(state) == sorted((o.name, o.value, o.unit, o.day) for o in patient.observations if o.day <= day)
        for key, value in patient.model_dump(exclude={"id", "observations"}).items():
            assert state[key] == value, key
    assert "facts_by_day" not in result and "checkpoints" not in result and "patient_day0" not in result


def test_same_day_multiple_measurements_are_not_collapsed():
    t = track()
    t.patient.observations.append(Observation(name="temperature", value=38.2, unit="°C", day=0))
    result = payload(t)
    assert len(result["patient_states"]["0"]["observations"]["temperature"]) == 2


def test_future_culture_does_not_appear_in_day_zero_but_is_available_on_day_three():
    t = track()
    
    t.patient.culture = t.checkpoints[0].patient.culture.model_copy(deep=True)
    result = payload(t)
    assert result["patient_states"]["0"]["culture"] is None
    culture = result["patient_states"]["3"]["culture"]
    assert culture["available_day"] == 3 and culture["susceptibility"]["ampicillin"] == "R"
    assert [(p["version"], p["state_day"]) for p in result["plans"]] == [(1, "0"), (2, "3")]


def test_future_measurement_is_not_available_to_earlier_version():
    t = track()
    t.patient.observations.append(Observation(name="qtc_ms", value=700, unit="ms", day=3))
    result = payload(t)
    assert all(row[2] == 0 and row[0] != 700 for row in result["patient_states"]["0"]["observations"]["qtc_ms"])


def test_unknown_allergies_empty_allergies_and_comorbidities_remain_distinct():
    t = track()
    t.patient.allergies = None
    t.patient.comorbidities = ["Сахарный диабет"]
    t.checkpoints[0].patient.allergies = []
    result = payload(t)
    assert result["patient_states"]["0"]["allergies"] is None
    assert result["patient_states"]["0"]["comorbidities"] == ["Сахарный диабет"]
    assert result["patient_states"]["3"]["allergies"] == []


def test_mutation_history_retains_actual_wrong_and_correct_dose_separately_from_reference():
    t = track()
    t.checkpoints = []
    t.plans = [make_plan(version=1), make_plan(version=2)]
    t.plans[0].actions[0].dose = "999 г"
    result = payload(t)
    assert len(result["patient_states"]) == 1
    assert [p["actions"][0]["dose"] for p in result["plans"]] == ["999 г", "2,0 г"]
    canonical = next(a for a in result["source_catalog"] if a["id"] == "ampicillin_iv")
    assert canonical["dose"] == "2,0 г"
    assert [p["actions"][0]["id"] for p in result["plans"]] == ["v1:ampicillin_iv", "v2:ampicillin_iv"]


def test_execution_order_and_blocked_actions_are_preserved_without_inference():
    t = track()
    t.plans[-1].actions[0].status = "blocked"
    result = payload(t)
    assert result["plans"][-1]["actions"][0]["status"] == "blocked"
    assert result["execution_log"][0]["entries"] == t.checkpoints[0].execution_log
    assert [entry["sequence"] for entry in result["execution_log"][0]["entries"]] == [1, 2]
    old_trace = track()
    old_trace.checkpoints[0].execution_log = []
    assert payload(old_trace)["execution_log"][0]["entries"] == []


def test_dated_clinical_events_are_retained_even_without_corresponding_observation():
    t = track()
    event = Event(day=3, kind="rash", description="Синтетическое сообщение о сыпи; числового показателя нет")
    t.checkpoints[0].events.append(event)
    result = payload(t)
    assert result["clinical_events"][0]["checkpoint_day"] == 3
    assert event.model_dump() in result["clinical_events"][0]["events"]
    assert result["clinical_events"][0]["events"] == [e.model_dump() for e in t.checkpoints[0].events]


def test_reference_findings_expected_answers_and_planner_self_assessment_are_withheld():
    t = track()
    marker = "SHOULD_NOT_REACH_REVIEWER"
    t.audits = [AuditReport(findings=[], checked_rules=[marker], limitations=[marker], llm_summary=marker)]
    t.metadata["expected_answer"] = marker
    t.transitions = [{"expected_finding": marker}]
    t.retrievals = [{"previous_auditor_verdict": marker}]
    serialized = json.dumps(payload(t))
    assert marker not in serialized
    assert "UNTRUSTED_PLANNER_CONCLUSION" not in serialized
    assert "UNTRUSTED_PLANNER_EXPLANATION" not in serialized


def test_source_texts_are_complete_deduplicated_and_separate_from_actual_citations():
    t = track()
    t.plans[0].actions[0].evidence_ids = ["fictional-source"]
    chunks = load_chunks()
    result = build_review_payload(t, load_catalog(), chunks + chunks[:1])
    actual = {c["id"]: c for c in result["evidence"]}
    assert len(actual) == len(result["evidence"]) == len(chunks)
    for chunk in chunks:
        assert actual[chunk["id"]]["text"] == chunk["text"]
    assert result["plans"][0]["actions"][0]["evidence_ids"] == ["fictional-source"]
    canonical = next(a for a in result["source_catalog"] if a["id"] == "ampicillin_iv")
    assert "cap-dose-ampicillin" in canonical["evidence_ids"]


def test_payload_does_not_mutate_or_share_nested_state_with_trace():
    t = track()
    before = t.model_dump_json()
    result = payload(t)
    result["plans"][0]["actions"][0]["evidence_ids"].clear()
    result["execution_log"][0]["entries"][0]["sequence"] = 99
    assert t.model_dump_json() == before


class FakeRetriever:
    def __init__(self):
        self.by_id = {c["id"]: c for c in load_chunks()}

    def search(self, query, count):
        return [self.by_id["cap-empiric"]]

    def lookup(self, ids):
        return [self.by_id[i] for i in dict.fromkeys(ids)]


class FakeClient:
    def complete(self, role, system, data, schema):
        self.role, self.system, self.payload, self.schema = role, system, data, schema
        return LLMAudit(findings=[], summary="No supported defects", limitations=[])


def test_llm_review_passes_compact_independent_payload_without_network():
    t = track()
    t.plans[0].actions[0].evidence_ids = ["cap-hydration"]
    client = FakeClient()
    result, retrieval = llm_review(t, FakeRetriever(), load_catalog(), client)
    assert result.findings == []
    assert client.role == "auditor" and client.schema is LLMAudit
    assert "cap-dose-ampicillin" in retrieval["lookup_ids"]
    assert retrieval["payload_version"] == "compact-timeline-v2"
    assert set(client.payload["patient_states"]) == {"0", "3"}
    assert client.payload["decision_facts"][0]["culture_available_at_decision"] is None


def test_decision_facts_exclude_future_culture_from_the_initial_version():
    t = track()
    t.patient.culture = t.checkpoints[0].patient.culture.model_copy(deep=True)
    facts = payload(t)["decision_facts"]
    assert [(fact["version"], fact["day"]) for fact in facts] == [(1, 0), (2, 3)]
    assert facts[0]["culture_available_at_decision"] is None
    assert "susceptibility" not in json.dumps(facts[0])
    assert facts[1]["culture_available_at_decision"] == t.checkpoints[0].patient.culture.model_dump()


def test_latest_observation_fact_keeps_its_original_measurement_day():
    t = track()
    result = payload(t)
    facts = {fact["version"]: fact for fact in result["decision_facts"]}
    original = next(observation for observation in t.patient.observations if observation.name == "qtc_ms")
    assert facts[2]["latest_observations"]["qtc_ms"] == {
        "value": original.value, "unit": original.unit, "measured_day": original.day}
    assert original.day < facts[2]["day"]
    assert facts[1]["latest_observations"]["temperature"]["measured_day"] == 0
    assert facts[2]["latest_observations"]["temperature"]["measured_day"] == 3
    assert len(result["patient_states"]["3"]["observations"]["temperature"]) == 2


def test_decision_execution_facts_sort_existing_numbers_without_changing_raw_log():
    t = track()
    t.checkpoints[0].execution_log.reverse()
    result = payload(t)
    assert [entry["sequence"] for entry in result["execution_log"][0]["entries"]] == [2, 1]
    first, second = result["decision_facts"]
    assert [entry["sequence"] for entry in first["execution_sequence"]] == [1, 2]
    assert [entry["event"] for entry in first["execution_sequence"]] == ["sputum_collected", "medication_started"]
    assert second["execution_sequence"] == []
    assert sorted(first["execution_sequence"], key=lambda entry: entry["sequence"]) == sorted(
        t.checkpoints[0].execution_log, key=lambda entry: entry["sequence"])


def test_decision_execution_facts_do_not_create_missing_events_or_sequence_numbers():
    t = track()
    del t.checkpoints[0].execution_log[0]["sequence"]
    result = payload(t)
    first = result["decision_facts"][0]["execution_sequence"]
    assert first == t.checkpoints[0].execution_log
    assert "sequence" not in first[0]
    assert first[1]["sequence"] == 2
    t.checkpoints[0].execution_log = []
    assert all(fact["execution_sequence"] == [] for fact in payload(t)["decision_facts"])


def test_decision_facts_have_only_observations_culture_and_execution_data_at_the_end():
    t = track()
    marker = "PRIVATE_RULE_VERDICT"
    t.audits = [AuditReport(findings=[], checked_rules=[marker], limitations=[marker], llm_summary=marker)]
    t.metadata["expected_finding"] = marker
    result = payload(t)
    for fact in result["decision_facts"]:
        assert set(fact) == {"version", "day", "culture_available_at_decision",
                             "latest_observations", "execution_sequence"}
        for observation in fact["latest_observations"].values():
            assert set(observation) == {"value", "unit", "measured_day"}
    serialized = json.dumps(result)
    assert marker not in serialized
    assert serialized.index('"evidence"') < serialized.index('"patient_states"')
    assert list(result)[-3:] == ["patient_states", "plans", "decision_facts"]
    assert "decision_facts" in SYSTEM


def test_decision_facts_do_not_share_mutable_data_with_the_trace():
    t = track()
    before = t.model_dump_json()
    result = payload(t)
    result["decision_facts"][0]["execution_sequence"][0]["sequence"] = 999
    result["decision_facts"][1]["culture_available_at_decision"]["susceptibility"].clear()
    assert t.model_dump_json() == before
