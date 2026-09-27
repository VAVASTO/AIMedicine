from __future__ import annotations

from .audit import audit_trace
from .schemas import Finding, Trace


VERIFICATION_LIMITATION = (
    "Проверка предложений LLM ограничена реализованными фактическими правилами. "
    "Утверждения вне этого набора сохраняются как непроверенные и требуют эксперта. "
    "Совпадение идентификаторов источников не доказывает произвольное смысловое утверждение; "
    "принятые находки используют каноническую формулировку проверенного правила."
)


def verify_findings(
    trace: Trace, proposed: list[Finding], chunks: list[dict], catalog: list[dict],
) -> tuple[list[Finding], list[Finding], list[dict]]:


    reference = audit_trace(trace, chunks, catalog)
    known_sources = {chunk["id"] for chunk in chunks}
    by_target: dict[tuple[str, int, str | None], list[Finding]] = {}
    for finding in reference.findings:
        by_target.setdefault((finding.code, finding.plan_version, finding.action_id), []).append(finding)

    accepted: list[Finding] = []
    unverified: list[Finding] = []
    validation: list[dict] = []
    for index, proposal in enumerate(proposed):
        target = (proposal.code, proposal.plan_version, proposal.action_id)
        candidates = by_target.get(target, [])
        record = {
            "index": index, "code": proposal.code, "plan_version": proposal.plan_version,
            "action_id": proposal.action_id, "proposed": proposal.model_dump(),
        }
        evidence = set(proposal.evidence_ids)
        unknown = evidence - known_sources
        matched = next((ref for ref in candidates if evidence.intersection(ref.evidence_ids)), None)

        if not candidates:
            reason = (
                "Нет фактической находки с теми же code, plan_version и action_id в независимой "
                "проверке истории. Утверждение может противоречить данным либо выходить за покрытие "
                "реализованных правил; оно не подтверждено и требует экспертной проверки."
            )
        elif not evidence:
            reason = "Предложение не содержит источников; подтверждение по фактическому правилу невозможно."
        elif unknown:
            reason = "Предложение ссылается на отсутствующие источники: " + ", ".join(sorted(unknown))
        elif matched is None:
            reason = "Источники предложения не пересекаются с основаниями соответствующего фактического правила."
        else:


            confirmed = matched.model_copy(deep=True)
            confirmed.origin = "llm"
            normalized = [field for field in (
                "severity", "category", "status", "message", "patient_evidence", "recommendation", "evidence_ids",
            ) if getattr(proposal, field) != getattr(confirmed, field)]
            accepted.append(confirmed)
            record.update({
                "decision": "accepted", "accepted": True,
                "reason": "Цель замечания подтверждена независимым фактическим правилом; содержание и временной статус приведены к проверенному правилу.",
                "reference_code": matched.code, "reference_status": matched.status,
                "reference_severity": matched.severity,
                "matched_evidence_ids": sorted(evidence.intersection(matched.evidence_ids)),
                "normalizations": normalized,
                "canonical_finding": confirmed.model_dump(),
            })
            validation.append(record)
            continue

        unverified.append(proposal.model_copy(deep=True))
        record.update({"decision": "unverified", "accepted": False, "reason": reason,
                       "coverage_limitation": VERIFICATION_LIMITATION})
        validation.append(record)
    return accepted, unverified, validation
