from __future__ import annotations

from typing import TypedDict
from uuid import uuid4
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END

from .audit import audit_trace
from .data import load_catalog, load_chunks
from .planner import make_plan
from .retrieval import Retriever
from .reviewer import llm_review
from .schemas import Trace, Patient, Plan
from .simulation import simulate
from .finding_verifier import verify_findings, VERIFICATION_LIMITATION


class WorkflowState(TypedDict):
    trace: Trace


def requires_review(trace):
    plan = trace.plans[-1]
    return bool(plan.missing_data or trace.audits[-1].unverified_findings or
                any(a.status == "blocked" for a in plan.actions) or
                any(f.status == "open" and f.severity in ("critical", "warning")
                    for f in trace.audits[-1].findings))


def plan_diff(before: Plan, after: Plan):
    old = {a.catalog_id: a for a in before.actions if a.status == "active"}
    new = {a.catalog_id: a for a in after.actions if a.status == "active"}
    changed = [k for k in old.keys() & new.keys()
               if (old[k].dose, old[k].route, old[k].frequency) != (new[k].dose, new[k].route, new[k].frequency)]
    return {"removed": sorted(old.keys() - new.keys()), "added": sorted(new.keys() - old.keys()),
            "changed": sorted(changed), "continued": sorted(old.keys() & new.keys() - set(changed))}


def conservative_repair(trace, report, catalog):

    previous = trace.plans[-1]
    relevant = [f for f in report.findings if f.status == "open" and f.plan_version == previous.version
                and f.severity == "critical"]
    if not relevant:
        return None
    plan = previous.model_copy(deep=True)
    plan.version += 1
    plan.origin = "repair"
    plan.rationale = "Ограниченное исправление после аудита: восстановлены подтверждённые параметры либо действие заблокировано до решения врача. Новые результаты и согласия не создавались."
    cat = {a["id"]: a for a in catalog}
    for action in plan.actions:
        defects = [f for f in relevant if f.action_id == action.id or
                   (f.action_id is None and action.category in ("medication", "procedure"))]
        source = cat.get(action.catalog_id)
        if defects:
            repairable = {"catalog_parameter_mismatch", "citation_missing", "citation_support_incomplete", "citation_unrelated"}
            if source and all(f.code in repairable for f in defects):
                
                for field in ("category", "drug_id", "dose", "route", "frequency", "evidence_ids"):
                    setattr(action, field, source.get(field))
            else:
                action.status = "blocked"
                plan.missing_data.append(f"Требуется клиническая проверка: {action.title}")
        action.id = f"v{plan.version}:{action.catalog_id}"
    return plan


def run_case(patient: Patient, scenario: dict, mode="offline", client=None, seed=42):
    chunks, catalog = load_chunks(), load_catalog()
    retriever = Retriever(chunks)
    initial = Trace(patient=patient.model_copy(deep=True), plans=[], checkpoints=[], mode=mode,
                    run_id=str(uuid4()), metadata={"created_at": datetime.now(timezone.utc).isoformat(),
                    "guideline_version": "654_2 / 2024", "synthetic": True,
                    "scenario_label": scenario.get("label", ""), "prompt_version": "2",
                    "simulation_not_clinically_validated": True})

    def start(state):
        t = state["trace"].model_copy(deep=True)
        plan, retrieval = make_plan(t.patient, 0, 1, mode, catalog, retriever, client)
        t.plans.append(plan)
        t.retrievals.append(retrieval)
        t.transitions.append({"stage": "initial_plan", "day": 0, "version": plan.version})
        report = audit_trace(t, chunks, catalog)
        t.audits.append(report)
        repaired = conservative_repair(t, report, catalog)
        if repaired:
            t.plans.append(repaired)
            t.transitions.append({"stage": "preflight_block_or_repair", "day": 0, "version": repaired.version,
                                  "diff": plan_diff(plan, repaired)})
        return {"trace": t}

    def checkpoint(state):
        t = state["trace"].model_copy(deep=True)
        cp = simulate(t.patient, t.plans[-1], scenario, seed)
        t.checkpoints.append(cp)
        t.transitions.append({"stage": "checkpoint", "day": cp.day, "events": [e.kind for e in cp.events]})
        return {"trace": t}

    def choose(state):
        return "revise" if state["trace"].checkpoints[-1].events else "continue"

    def revise(state):
        t = state["trace"].model_copy(deep=True)
        cp, old = t.checkpoints[-1], t.plans[-1]
        plan, retrieval = make_plan(cp.patient, cp.day, old.version + 1, mode, catalog, retriever,
                                    client, old, cp.events)
        t.plans.append(plan)
        t.retrievals.append(retrieval)
        t.transitions.append({"stage": "revision", "day": cp.day, "reasons": [e.kind for e in cp.events],
                              "diff": plan_diff(old, plan)})
        return {"trace": t}

    def keep(state):
        t = state["trace"].model_copy(deep=True)
        old, cp = t.plans[-1], t.checkpoints[-1]
        plan = old.model_copy(deep=True)
        plan.day = cp.day
        plan.version += 1
        plan.origin = "rule"
        for action in plan.actions:
            action.id = f"v{plan.version}:{action.catalog_id}"
        plan.rationale = "Положительная динамика в сценарной контрольной точке; триггеров смены антибиотика не обнаружено. Продолжение с оценкой переносимости. Переход на приём внутрь требует отдельной проверки всех критериев стабильности, включая повторные измерения температуры."
        t.plans.append(plan)
        t.transitions.append({"stage": "continue", "day": cp.day, "diff": plan_diff(old, plan)})
        return {"trace": t}

    def audit(state):
        t = state["trace"].model_copy(deep=True)
        report = audit_trace(t, chunks, catalog)
        if mode == "live":
            reviewed, retrieval = llm_review(t, retriever, catalog, client)
            accepted, unverified, checks = verify_findings(t, reviewed.findings, chunks, catalog)
            report.findings.extend(accepted)
            report.unverified_findings = unverified
            report.llm_validation = checks
            report.llm_reviewed = True
            report.llm_raw_summary = reviewed.summary
            report.llm_summary = f"Независимый LLM-аудит: подтверждено {len(accepted)} замечаний; {len(unverified)} предположений не подтверждено проверяемыми правилами. Они сохранены отдельно и не изменяют лечение автоматически."
            report.limitations.extend(reviewed.limitations)
            report.limitations.append(VERIFICATION_LIMITATION)
            t.retrievals.append(retrieval)
        t.audits.append(report)
        t.transitions.append({"stage": "final_audit", "day": t.plans[-1].day,
                              "open_findings": sum(f.status == "open" for f in report.findings)})
        repaired = conservative_repair(t, report, catalog)
        if repaired:
            old = t.plans[-1]
            t.plans.append(repaired)
            t.transitions.append({"stage": "audit_repair", "day": repaired.day, "diff": plan_diff(old, repaired)})
            verification = audit_trace(t, chunks, catalog)
            if mode == "live":
                reviewed, retrieval = llm_review(t, retriever, catalog, client)
                accepted, unverified, checks = verify_findings(t, reviewed.findings, chunks, catalog)
                verification.findings.extend(accepted)
                verification.unverified_findings = unverified
                verification.llm_validation = checks
                verification.llm_reviewed = True
                verification.llm_raw_summary = reviewed.summary
                verification.llm_summary = f"Повторный аудит: подтверждено {len(accepted)} замечаний; {len(unverified)} предположений не подтверждено проверяемыми правилами."
                verification.limitations.extend(reviewed.limitations)
                verification.limitations.append(VERIFICATION_LIMITATION)
                t.retrievals.append(retrieval)
            t.audits.append(verification)
        t.metadata["requires_human_review"] = requires_review(t)
        return {"trace": t}

    graph = StateGraph(WorkflowState)
    graph.add_node("initial", start)
    graph.add_node("simulate", checkpoint)
    graph.add_node("revise", revise)
    graph.add_node("continue", keep)
    graph.add_node("audit", audit)
    graph.add_edge(START, "initial")
    graph.add_edge("initial", "simulate")
    graph.add_conditional_edges("simulate", choose, {"revise": "revise", "continue": "continue"})
    graph.add_edge("revise", "audit")
    graph.add_edge("continue", "audit")
    graph.add_edge("audit", END)
    return graph.compile().invoke({"trace": initial}, {"recursion_limit": 12})["trace"]
