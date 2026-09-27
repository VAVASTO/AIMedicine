import argparse
import json
import sys
from pathlib import Path

from .data import ROOT, read_json, save_json
from .llm import MistralClient, LLMError
from .pipeline import run_case
from .schemas import Patient, Trace


def demo(mode, output, client=None):
    patients = [Patient.model_validate(x) for x in read_json(ROOT / "data/patients.json")]
    scenarios = read_json(ROOT / "data/scenarios.json")
    rows = []
    for patient in patients:
        print(f"Running {patient.id} [{mode}]", flush=True)
        trace = run_case(patient, scenarios[patient.id], mode, client)
        save_json(output / f"{patient.id}.json", trace.model_dump())
        active = lambda p: [a.drug_id for a in p.actions if a.category == "medication" and a.status == "active"]
        rows.append({"patient_id": patient.id, "path": f"{patient.id}.json",
                     "initial_drugs": active(trace.plans[0]), "final_drugs": active(trace.plans[-1]),
                     "findings": len(trace.audits[-1].findings),
                     "open_findings": sum(f.status == "open" for f in trace.audits[-1].findings),
                     "unverified_llm_findings": len(trace.audits[-1].unverified_findings),
                     "requires_human_review": trace.metadata["requires_human_review"],
                     "revision_triggered": any(t["stage"] == "revision" for t in trace.transitions)})
        print(f"  completed: {rows[-1]['initial_drugs']} -> {rows[-1]['final_drugs']}; open findings={rows[-1]['open_findings']}", flush=True)
    summary = {"mode": mode, "source_version": "654_2 / 2024", "cases": rows,
               "usage": client.summary() if client else {"api_attempts": 0},
               "note": "Синтетический учебный прототип; source edition frozen, not asserted latest."}
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Synthetic inpatient CDS demonstration")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--mode", choices=["offline", "live"], default="offline")
        p.add_argument("--output", type=Path)
        p.add_argument("--max-calls", type=int)
    p = sub.add_parser("replay")
    p.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "replay":
        t = Trace.model_validate(read_json(args.path))
        print(json.dumps({"patient": t.patient.id, "mode": t.mode, "versions": len(t.plans),
                          "audit": t.audits[-1].model_dump()}, ensure_ascii=False, indent=2))
        return
    output = args.output or Path("reports") / (args.mode if args.command == "demo" else "evaluation")
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    client = None


    save_json(output / "run_status.json", {"status": "running", "mode": args.mode})
    try:
        client = MistralClient(max_calls=args.max_calls) if args.mode == "live" else None
        if args.command == "demo":
            result = demo(args.mode, output, client)
        else:
            from .evaluation import evaluate
            result = evaluate(output, args.mode, client)
        print("Saved:", output)
        (output / "run_error.json").unlink(missing_ok=True)
        save_json(output / "run_status.json", {"status": "complete", "mode": args.mode})
        if client:
            print("API usage:", json.dumps(client.summary(), ensure_ascii=False))
    except (LLMError, ValueError) as exc:
        
        message = str(exc) if isinstance(exc, LLMError) else f"Structured validation failed ({type(exc).__name__}); no fallback to offline."
        save_json(output / "run_error.json", {"error": message, "mode": args.mode, "usage": client.summary() if client else {}})
        save_json(output / "run_status.json", {"status": "failed", "mode": args.mode})
        print(message, file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        if client:
            save_json(output / "usage.json", client.summary())
            client.close()


if __name__ == "__main__":
    main()
