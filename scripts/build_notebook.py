from __future__ import annotations

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(source).strip() + "\n"}


cells = [
    markdown("""
    # AIMedicine: разбор трёх случаев

    Синтетическая когорта взрослых пациентов с внебольничной пневмонией. Клинические рекомендации — редакция 2024 года.

    Карточка пациента → поиск по документу → план → сценарная динамика дня 3 → пересмотр → независимый аудит.

    Notebook показывает сохранённые результаты из `reports/`: завершённый live-прогон с Mistral либо, если его нет, явно обозначенный offline-прогон. Показатели симуляции задаются сценарием; они используются для проверки переходов между этапами.
    """),
    code(r"""
    import json
    from pathlib import Path
    import pandas as pd
    from IPython.display import display, Markdown

    candidates = [Path.cwd(), Path.cwd().parent]
    ROOT = next((p for p in candidates if (p / "aimedicine").is_dir() and (p / "reports").is_dir()), None)
    if ROOT is None:
        raise RuntimeError("Откройте notebook из корня проекта или папки notebooks")
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    expected_ids = {p["id"] for p in load(ROOT / "data/patients.json")}

    def completed_run(mode):
        folder = ROOT / "reports" / mode
        manifest = folder / "summary.json"
        if not manifest.exists():
            return None, "нет завершающего summary.json"
        status_file = folder / "run_status.json"
        if status_file.exists() and load(status_file).get("status") != "complete":
            return None, "run_status.json не подтверждает завершение запуска"
        summary = load(manifest)
        if summary.get("mode") != mode or summary.get("status") in {"failed", "error"}:
            return None, "режим/статус манифеста не подтверждает успешный запуск"
        if {c.get("patient_id") for c in summary.get("cases", [])} != expected_ids or len(summary["cases"]) != len(expected_ids):
            return None, "манифест не содержит полный набор трёх пациентов"
        error_file = folder / "run_error.json"
        if error_file.exists() and error_file.stat().st_mtime_ns > manifest.stat().st_mtime_ns:
            return None, "после манифеста зафиксирована ошибка запуска"
        result = []
        for row in summary["cases"]:
            report_path = (folder / row["path"]).resolve()
            if report_path.parent != folder.resolve() or not report_path.is_file():
                return None, "отсутствует файл, указанный в манифесте"
            if report_path.stat().st_mtime_ns > manifest.stat().st_mtime_ns:
                return None, "файлы менялись после манифеста; возможен незавершённый повторный запуск"
            item = load(report_path)
            if item.get("mode") != mode or item.get("patient", {}).get("id") != row["patient_id"] or not item.get("plans"):
                return None, "история не соответствует манифесту"
            result.append((report_path, item))
        return result, "полный набор подтверждён summary.json"

    traces, live_status = completed_run("live")
    if traces is not None:
        selected_mode = "live"
        display(Markdown("Выбран **live**: найден полный завершённый запуск с API."))
    else:
        traces, offline_status = completed_run("offline")
        selected_mode = "offline"
        display(Markdown(f"Выбран **offline**. Live не подтверждён: {live_status}. Offline не заменяет проверку LLM."))
        if traces is None:
            raise RuntimeError("Нет полного offline-запуска: " + offline_status + ". Выполните python -m aimedicine.cli demo --mode offline --output reports/offline")
    display(Markdown(f"Показано **{len(traces)}** сохранённых историй."))
    """),
    markdown("""
    ## Сводка запусков

    Каждый результат связан с конкретной историей и режимом. Находки приведены по записям аудита, включая исторические замечания.
    """),
    code(r"""
    rows = []
    for path, trace in traces:
        initial = trace["plans"][0] if trace["plans"] else {}
        final = trace["plans"][-1] if trace["plans"] else {}
        def drug_names(plan):
            return ", ".join(a["title"] for a in plan.get("actions", []) if a.get("category") == "medication" and a.get("status") != "stopped")
        findings = [f for audit in trace.get("audits", []) for f in audit.get("findings", [])]
        unverified = [f for audit in trace.get("audits", []) for f in audit.get("unverified_findings", [])]
        rows.append({"Пациент": trace["patient"]["id"], "Режим": trace["mode"], "Начальная терапия": drug_names(initial), "Итоговая терапия": drug_names(final), "Версий плана": len(trace["plans"]), "Подтверждённых записей": len(findings), "Неподтверждённых гипотез LLM": len(unverified), "Нужна экспертная проверка": trace.get("metadata", {}).get("requires_human_review", "не указано"), "Файл": str(path.relative_to(ROOT))})
    display(pd.DataFrame(rows))
    """),
    markdown("""
    ## Пациенты, динамика и планы

    Каждое действие показывает исходное обоснование и идентификаторы источников. Доступные показатели сравниваются по времени наблюдения. Отсутствующие данные не заменяются нулями.
    """),
    code(r"""
    for path, trace in traces:
        patient = trace["patient"]
        display(Markdown(f"### {patient['id']} · {trace['mode']}\n\n{patient['age']} лет; {patient['diagnosis']}. Причина госпитализации: {patient['hospitalization_reason']}"))
        display(Markdown("**Аллергии:** " + ("неизвестны" if patient.get("allergies") is None else ", ".join(patient["allergies"]) or "нет в сценарии")))
        display(Markdown("**Сопутствующие заболевания:** " + (", ".join(patient.get("comorbidities", [])) or "нет в сценарии")))
        observations = patient["observations"] + [o for c in trace.get("checkpoints", []) for o in c["patient"]["observations"]]
        frame = pd.DataFrame(observations).drop_duplicates(["name", "unit", "day"], keep="last")
        display(frame.pivot_table(index=["name", "unit"], columns="day", values="value", aggfunc="last"))
        for checkpoint in trace.get("checkpoints", []):
            display(Markdown(f"**День {checkpoint['day']}:** " + "; ".join(e["description"] for e in checkpoint.get("events", []))))
            if checkpoint.get("execution_log"):
                display(Markdown("**Журнал действий симуляции:** относительный порядок задан полем sequence; это синтетические события."))
                display(pd.DataFrame(checkpoint["execution_log"]))
            display(Markdown("Допущения: " + "; ".join(checkpoint.get("assumptions", []))))
        for plan in trace["plans"]:
            display(Markdown(f"**План {plan['version']} · день {plan['day']} · {plan['origin']}**\n\n{plan['rationale']}"))
            display(pd.DataFrame([{"Действие": a["title"], "Доза": a.get("dose"), "Частота": a.get("frequency"), "Статус": a["status"], "Обоснование": a["indication"], "Источники": ", ".join(a["evidence_ids"])} for a in plan["actions"]]))
            if plan.get("missing_data"):
                display(Markdown("Неизвестные данные: " + "; ".join(plan["missing_data"])))
    """),
    markdown("""
    ## Независимый аудит

    Находки правил и LLM имеют отдельную маркировку. Подтверждение связывает замечание с фактом, версией плана, действием и источником. Остальные предположения модели сохраняются отдельно для экспертной оценки. Исправленные находки остаются в истории.

    Проверка покрывает заданный каталог и перечисленные правила; произвольные утверждения за их пределами автоматически не подтверждаются.
    """),
    code(r"""
    for path, trace in traces:
        display(Markdown(f"### {trace['patient']['id']} · {trace['mode']}"))
        if trace.get("metadata", {}).get("requires_human_review"):
            display(Markdown("**Требуется экспертная проверка.** Неподтверждённые гипотезы и недостаточные данные не считаются чистым аудитом."))
        for number, audit in enumerate(trace.get("audits", []), 1):
            display(Markdown(f"**Аудит {number}. LLM-проверка: {'да' if audit.get('llm_reviewed') else 'нет'}.**"))
            if audit.get("llm_summary"):
                display(Markdown(audit["llm_summary"]))
            records = [{"Код": f["code"], "Уровень": f["severity"], "Статус": f["status"], "Происхождение": f["origin"], "Версия": f["plan_version"], "Находка": f["message"], "Доказательство": f["patient_evidence"], "Исправление": f["recommendation"], "Источники": ", ".join(f["evidence_ids"])} for f in audit.get("findings", [])]
            display(pd.DataFrame(records)) if records else display(Markdown("Подтверждённых нарушений в покрытых проверках не найдено."))
            if audit.get("unverified_findings"):
                display(Markdown("**Неподтверждённые предположения LLM — отдельная экспертная оценка**"))
                display(pd.DataFrame([{"Код": f["code"], "Версия": f["plan_version"], "Гипотеза": f["message"], "Заявленное основание": f["patient_evidence"], "Источники": ", ".join(f["evidence_ids"])} for f in audit["unverified_findings"]]))
                if audit.get("llm_validation"):
                    display(pd.DataFrame(audit["llm_validation"]))
            if audit.get("llm_raw_summary"):
                display(Markdown("**Необработанное заключение LLM (может содержать неподтверждённые утверждения):**\n\n" + audit["llm_raw_summary"]))
            display(Markdown("Ограничения: " + "; ".join(audit.get("limitations", []))))
    """),
    markdown("""
    ## Проверяемые источники

    Ниже приведены использованные локальные фрагменты. Номер страницы и раздел относятся к зафиксированной редакции документа; технический ID не заменяет номер пункта КР.
    """),
    code(r"""
    chunks = []
    for name in ["document_chunks.json", "chunks.json", "chunks.jsonl", "guideline_chunks.json", "guideline_chunks.jsonl", "supplemental_chunks.json", "policies.json"]:
        path = ROOT / "data" / name
        if path.exists():
            data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.suffix == ".jsonl" else load(path)
            records = data.get("chunks", data.get("documents", list(data.values()))) if isinstance(data, dict) else data
            role = "Автоматический фрагмент PDF" if name == "document_chunks.json" else "Опорная выдержка для проверки" if name == "chunks.json" else "Дополнительный источник"
            chunks.extend({**item, "display_role": role} for item in records)
    used = {e for _, t in traces for p in t["plans"] for a in p["actions"] for e in a["evidence_ids"]}
    used |= {e for _, t in traces for audit in t.get("audits", []) for f in audit["findings"] for e in f["evidence_ids"]}
    used |= {e for _, t in traces for audit in t.get("audits", []) for f in audit.get("unverified_findings", []) for e in f["evidence_ids"]}
    for chunk in chunks:
        identifier = chunk.get("id", chunk.get("chunk_id"))
        if identifier in used:
            page = chunk.get("page", chunk.get("pdf_page", chunk.get("pages", "—")))
            section = chunk.get("section", chunk.get("section_title", "—"))
            url = chunk.get("source_url", chunk.get("url")) or chunk.get("source", {}).get("url")
            if url and isinstance(page, int) and ".pdf" in url.lower():
                url = url.split("#", 1)[0] + f"#page={page}"
            link = f"\n\n[Открыть источник]({url})" if url else ""
            display(Markdown(f"**{identifier} · {chunk['display_role']} · {section} · страница {page or '—'}**\n\n{chunk.get('text', chunk.get('content', ''))}{link}"))
    missing = used - {c.get("id", c.get("chunk_id")) for c in chunks}
    if missing:
        display(Markdown("**Не найдены локальные фрагменты:** " + ", ".join(sorted(missing))))
    """),
    markdown("""
    ## Проверочные прогоны

    Внесённые дефекты проверяют обнаружение конкретных ошибок на копиях эталонных планов. Они отделены от трёх основных случаев. Результаты характеризуют этот набор тестов.
    """),
    code(r"""
    evaluation_files = sorted(path for path in (ROOT / "reports").rglob("evaluation.json") if "development" not in path.relative_to(ROOT / "reports").parts)
    if not evaluation_files:
        display(Markdown("Отчёт evaluation.json отсутствует; результаты оценивания здесь не заявляются."))
    for path in evaluation_files:
        display(Markdown(f"**{path.relative_to(ROOT)}**"))
        display(load(path))
    """),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"aimedicine-{index:02d}"
output = ROOT / "notebooks" / "demo.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Notebook created: {output}")
