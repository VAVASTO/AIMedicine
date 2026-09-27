import json

import pytest

from aimedicine import evaluation
from aimedicine.evaluation import detected_wrong_dose, procedural_metrics, wrong_dose_metrics
from aimedicine.schemas import Finding, LLMAudit, LLMCandidate


def finding(**updates):
    values = dict(code="catalog_parameter_mismatch", severity="critical", category="citation", plan_version=1,
                  action_id="v1:ampicillin_iv", message="Доза 999 г не соответствует источнику.",
                  evidence_ids=["cap-dose-ampicillin"], patient_evidence="Назначено 999 г",
                  recommendation="Исправить дозу", origin="llm")
    values.update(updates)
    return Finding(**values)


def procedural_finding(code="contrast_renal_precondition_missing", **updates):
    values = dict(code=code, severity="critical", category="procedural", plan_version=1,
                  action_id="v1:contrast_ct", message="Не подтверждены предварительные условия процедуры.",
                  evidence_ids=[evaluation.PROCEDURAL_TARGETS[code]],
                  patient_evidence="Нет результатов creatinine/egfr и подтверждённого согласия для contrast_ct.",
                  recommendation="Блокировать до подтверждения предварительных условий.", origin="llm")
    values.update(updates)
    return Finding(**values)


def test_wrong_dose_rubric_requires_specific_defect_and_source():
    assert detected_wrong_dose([finding()], "v1:ampicillin_iv")
    assert not detected_wrong_dose([finding(message="Требуется уточнить аллергию",patient_evidence="allergies=[]")], "v1:ampicillin_iv")
    assert not detected_wrong_dose([finding(evidence_ids=["cap-empiric"])], "v1:ampicillin_iv")
    assert not detected_wrong_dose([finding(action_id="v1:ecg")], "v1:ampicillin_iv")
    assert not detected_wrong_dose([finding(status="resolved")], "v1:ampicillin_iv")
    assert not detected_wrong_dose([finding(code="unverified_other")], "v1:ampicillin_iv")
    assert not detected_wrong_dose([finding(code="required_ecg_missing")], "v1:ampicillin_iv")


@pytest.mark.parametrize("text", ["Назначено 1999 г", "Назначено 9990 г", "Назначено 999 мг", "Показатель 999"])
def test_wrong_dose_rubric_requires_exact_injected_value_and_unit(text):
    assert not detected_wrong_dose([finding(message=text, patient_evidence=text)], "v1:ampicillin_iv")


def test_raw_llm_detection_is_not_created_by_canonical_verifier_text():
    raw = finding(message="Неизвестен аллергологический анамнез", patient_evidence="allergies=[]")
    canonical = finding()
    metrics = wrong_dose_metrics([raw], [canonical], [], [finding(origin="rule")], "v1:ampicillin_iv")
    assert not metrics["raw_llm"]["target_open_detected"]
    assert metrics["rule_verification"]["target_open_confirmed"]
    assert metrics["rule_audit"]["target_open_detected"]
    assert metrics["raw_llm"]["findings"][0]["message"] == raw.message


def test_rule_findings_cannot_count_as_raw_llm_detection():
    rule = finding(origin="rule")
    metrics = wrong_dose_metrics([rule], [], [], [rule], "v1:ampicillin_iv")
    assert not metrics["raw_llm"]["target_open_detected"]
    assert metrics["raw_llm"]["findings"] == []
    assert metrics["rule_audit"]["target_open_detected"]


def test_verifier_status_correction_remains_separate_from_raw_model_status():
    raw = finding(status="resolved")
    metrics = wrong_dose_metrics([raw], [finding()], [], [finding(origin="rule")], "v1:ampicillin_iv")
    assert not metrics["raw_llm"]["target_open_detected"]
    assert metrics["raw_llm"]["target_resolved_detected"]
    assert metrics["rule_verification"]["target_open_confirmed"]


def test_before_after_metrics_preserve_resolved_history_and_current_open_findings():
    before = wrong_dose_metrics([finding()], [finding()], [], [finding(origin="rule")], "v1:ampicillin_iv")
    historical = finding(status="resolved")
    other_open = finding(code="unverified_other", plan_version=2, action_id="v2:ampicillin_iv")
    after = wrong_dose_metrics([historical, other_open], [historical], [other_open],
                               [finding(origin="rule", status="resolved")], "v1:ampicillin_iv")
    assert before["raw_llm"]["target_open_detected"]
    assert not after["raw_llm"]["target_open_detected"]
    assert after["raw_llm"]["target_resolved_detected"]
    assert after["raw_llm"]["open_critical"] == 1
    assert after["rule_verification"]["unverified_findings"] == 1
    assert after["rule_verification"]["verified_open_critical"] == 0
    assert after["rule_audit"]["target_resolved_detected"]


def test_empty_model_response_is_a_miss_even_when_rules_find_defect():
    metrics = wrong_dose_metrics([], [], [], [finding(origin="rule")], "v1:ampicillin_iv")
    assert not metrics["raw_llm"]["target_open_detected"]
    assert not metrics["rule_verification"]["target_open_confirmed"]
    assert metrics["rule_audit"]["target_open_detected"]


def use_reference_retriever(monkeypatch):
    gold = evaluation.read_json(evaluation.ROOT / "data/retrieval_gold.json")
    expected = {row["query"]: row["expected_ids"] for row in gold}
    calls = []

    class ReferenceRetriever:
        def __init__(self, chunks):
            pass

        def search(self, query, k):
            assert k == 5
            calls.append(query)
            return [{"id": "automatic-chunk", "query": query}]

        def covered_references(self, hits):
            assert len(hits) == 1 and hits[0]["id"] == "automatic-chunk"
            return [{"id": anchor} for anchor in expected[hits[0]["query"]]]

    monkeypatch.setattr(evaluation, "Retriever", ReferenceRetriever)
    return calls, gold


def test_evaluation_counts_covered_references_without_confusing_them_with_index_ids(monkeypatch, tmp_path):
    calls, gold = use_reference_retriever(monkeypatch)
    result = evaluation.evaluate(tmp_path)
    assert calls == [row["query"] for row in gold]
    assert result["retrieval"]["reference_recall_at_5"] == 1
    assert result["retrieval"]["mean_recall_at_5"] == 1
    for row in result["retrieval"]["results"]:
        assert row["actual_ids"] == ["automatic-chunk"]
        assert row["covered_reference_ids"] == row["expected_ids"]
        assert not set(row["actual_ids"]) & set(row["expected_ids"])
    assert result["rule_detection"]["detected"] == result["rule_detection"]["total"] == 9
    assert result["rule_detection"]["llm_evaluated"] is False
    assert result["llm_example"] is None
    assert result["llm_procedural_example"] is None


def test_live_evaluation_does_not_turn_normalization_into_raw_llm_success(monkeypatch, tmp_path):
    use_reference_retriever(monkeypatch)
    responses = [
        [finding(message="Аллергологический анамнез неизвестен", patient_evidence="allergies=[]")],
        [finding(status="resolved")],
        [procedural_finding(), procedural_finding("procedure_consent_missing")],
    ]
    calls = []

    def review(trace, retriever, catalog, client):
        calls.append(trace.model_copy(deep=True))
        response = LLMAudit(findings=[LLMCandidate.model_validate(f.model_dump()) for f in responses[len(calls) - 1]],
                            summary="Stubbed auditor response", limitations=[])
        return response, {"stage": "independent_audit", "test_stub": True}

    class Client:
        def summary(self):
            return {"api_attempts": 0, "test_stub": True}

    monkeypatch.setattr(evaluation, "llm_review", review)
    result = evaluation.evaluate(tmp_path, mode="live", client=Client())
    example = result["llm_example"]
    assert len(calls) == 3
    assert len(calls[0].plans) == 1 and len(calls[1].plans) == 2
    assert example["repair_applied"]
    assert not example["llm_detected_wrong_dose"]
    assert not example["verified_wrong_dose_detected"]
    assert not example["before"]["raw_llm"]["target_open_detected"]
    assert example["before"]["rule_verification"]["target_open_confirmed"]
    assert example["before"]["rule_audit"]["target_open_detected"]
    assert example["after"]["raw_llm"]["target_resolved_detected"]
    assert example["after"]["rule_audit"]["target_resolved_detected"]
    assert not example["after"]["rule_audit"]["target_open_detected"]
    assert example["evaluated_faults"] == 1
    saved = evaluation.read_json(tmp_path / "fault_llm_before_after.json")
    assert saved["plans"][0]["actions"] != saved["plans"][1]["actions"]
    assert saved["audits"][0]["llm_validation"][0]["proposed"]["message"] == responses[0][0].message
    procedural = result["llm_procedural_example"]
    assert procedural["raw_llm"]["detected"] == procedural["raw_llm"]["total"] == 2
    assert procedural["rule_verification"]["confirmed"] == 2


@pytest.mark.parametrize("updates", [
    {"code": "unverified_other"}, {"action_id": "v1:ampicillin_iv"}, {"plan_version": 2},
    {"status": "resolved"}, {"origin": "rule"}, {"category": "clinical"},
    {"evidence_ids": ["cap-empiric"]},
])
def test_procedural_raw_detection_requires_exact_current_target_and_source(updates):
    raw = Finding.model_validate({**procedural_finding().model_dump(), **updates})
    canonical = procedural_finding()
    metrics = procedural_metrics([raw], [canonical], [], [procedural_finding(origin="rule")], "v1:contrast_ct")
    assert metrics["raw_llm"]["detected"] == 0
    assert metrics["rule_verification"]["confirmed"] == 1
    assert metrics["rule_audit"]["detected"] == 1
    assert not metrics["targets"][0]["raw_llm_detected_and_verified"]


def test_procedural_metrics_separate_other_rule_findings_and_do_not_double_count():
    renal = procedural_finding()
    consent = procedural_finding("procedure_consent_missing")
    other = finding(origin="rule", code="renal_data_missing")
    metrics = procedural_metrics([renal, renal, consent], [renal, consent], [],
                                  [renal, consent, other], "v1:contrast_ct")
    assert metrics["raw_llm"]["detected"] == metrics["raw_llm"]["total"] == 2
    assert len(metrics["raw_llm"]["findings"]) == 3
    assert metrics["other_rule_findings"] == [other.model_dump()]


@pytest.mark.parametrize("detect_procedural", [True, False])
def test_procedural_live_flow_uses_blind_payload_and_preserves_raw_output(tmp_path, detect_procedural):
    procedural = [procedural_finding(), procedural_finding("procedure_consent_missing")] if detect_procedural else []
    responses = [[finding()], [finding(status="resolved")], procedural]

    class RecordingClient:
        def __init__(self):
            self.calls = []

        def complete(self, role, system, payload, schema):
            assert role == "auditor" and schema is LLMAudit
            self.calls.append(payload)
            return schema.model_validate({"findings": [f.model_dump() for f in responses[len(self.calls) - 1]],
                                          "summary": "Stubbed auditor response", "limitations": []})

        def summary(self):
            return {"api_attempts": 0, "test_stub": True}

    client = RecordingClient()
    result = evaluation.evaluate(tmp_path, mode="live", client=client)
    assert len(client.calls) == 3
    payload = client.calls[2]
    encoded = json.dumps(payload, ensure_ascii=False)
    for marker in ("fault_injection", "expected", "base_plan_origin", "внесённый дефект",
                   "Намеренно внесённое", "procedure_without_renal_result"):
        assert marker not in encoded
    assert "audits" not in payload and "metadata" not in payload and "transitions" not in payload
    state = payload["patient_states"]["0"]
    assert "creatinine" not in state["observations"] and "egfr" not in state["observations"]
    assert state["consents"].get("contrast_ct") is not True
    actual = next(action for action in payload["plans"][0]["actions"] if action["id"] == "v1:contrast_ct")
    assert actual["status"] == "active"
    example = result["llm_procedural_example"]
    assert example["raw_llm"]["detected"] == (2 if detect_procedural else 0)
    assert example["rule_verification"]["confirmed"] == (2 if detect_procedural else 0)
    assert example["rule_audit"]["detected"] == 2
    assert example["other_rule_findings"]
    saved = evaluation.read_json(tmp_path / "fault_llm_procedure.json")
    assert len(saved["plans"]) == 1 and len(saved["audits"]) == 1
    assert saved["metadata"]["requires_human_review"]
    raw_saved = [check["proposed"] for check in saved["audits"][0]["llm_validation"]]
    assert raw_saved == [f.model_dump() for f in procedural]
    assert example["raw_llm"]["findings"] == raw_saved
    assert example["report"] == "fault_llm_procedure.json"
