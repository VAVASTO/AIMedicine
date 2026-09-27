from __future__ import annotations

import json
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PUBLIC_DEMO = os.environ.get("AIMEDICINE_PUBLIC_DEMO") == "1"
if not PUBLIC_DEMO:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

st.set_page_config(page_title="AIMedicine · проверяемые решения", page_icon="🩺", layout="wide")

STATUS = {"proposed": "Предложено", "active": "Активно", "blocked": "Заблокировано", "stopped": "Отменено", "open": "Открыто", "resolved": "Исправлено", "historical": "Историческое"}
ORIGIN = {"llm": "LLM", "fixture": "локальный проверочный сценарий", "repair": "исправление", "rule": "правило"}
METRIC = {"temperature": "Температура", "spo2": "SpO₂", "respiratory_rate": "Частота дыхания", "heart_rate": "Частота пульса", "crp": "С-реактивный белок", "wbc": "Лейкоциты", "creatinine": "Креатинин", "egfr": "Расчётная СКФ", "alt": "АЛТ", "ast": "АСТ", "qtc_ms": "QTc", "chest_xray_infiltrate": "Инфильтрат на рентгенограмме"}
CATEGORY = {"medication": "Препарат", "investigation": "Исследование", "monitoring": "Мониторинг", "regimen": "Режим", "procedure": "Процедура", "review": "Повторная оценка"}
SEVERITY = {"critical": "Критическое", "warning": "Предупреждение", "info": "Информация"}
MODE = {"offline": "Offline · проверяемые сценарии", "live": "Live · сохранённый ответ LLM", "replay": "Replay · просмотр результата"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def completed_directory(folder: Path) -> bool:
    manifest = folder / "summary.json"
    if not manifest.exists():
        return False
    try:
        run_status = folder / "run_status.json"
        if run_status.exists() and read_json(run_status).get("status") != "complete":
            return False
        summary = read_json(manifest)
        rows = summary.get("cases", [])
        expected = {p["id"] for p in read_json(ROOT / "data/patients.json")}
        if len(rows) != len(expected) or {x.get("patient_id") for x in rows} != expected:
            return False
        error = folder / "run_error.json"
        if error.exists() and error.stat().st_mtime_ns > manifest.stat().st_mtime_ns:
            return False
        for row in rows:
            path = (folder / row["path"]).resolve()
            if path.parent != folder.resolve() or not path.is_file() or path.stat().st_mtime_ns > manifest.stat().st_mtime_ns:
                return False
            trace = read_json(path)
            if trace.get("mode") != summary.get("mode") or trace.get("patient", {}).get("id") != row["patient_id"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


@st.cache_data
def chunks_from_file(path: str, modified: float) -> dict:
    file = Path(path)
    payload = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()] if file.suffix == ".jsonl" else read_json(file)
    if isinstance(payload, dict):
        payload = payload.get("chunks", payload.get("documents", list(payload.values())))
    return {str(x.get("id", x.get("chunk_id", ""))): x for x in payload if isinstance(x, dict)}


def load_chunks() -> dict:
    result = {}
    for name in ("document_chunks.json", "chunks.json", "chunks.jsonl", "guideline_chunks.json", "guideline_chunks.jsonl", "supplemental_chunks.json", "policies.json"):
        candidate = ROOT / "data" / name
        if candidate.exists():
            role = "Автоматический фрагмент PDF" if name == "document_chunks.json" else "Опорная выдержка для проверки" if name == "chunks.json" else "Дополнительный источник"
            result.update({identifier: {**chunk, "display_role": role} for identifier, chunk in chunks_from_file(str(candidate), candidate.stat().st_mtime).items()})
    return result


def sources(ids: list[str], chunks: dict, expandable: bool = True):
    for identifier in dict.fromkeys(ids):
        chunk = chunks.get(identifier)
        if not chunk:
            st.warning(f"Источник {identifier} отсутствует в локальном индексе.")
            continue
        section = chunk.get("section", chunk.get("section_title", ""))
        page = chunk.get("page", chunk.get("pdf_page", chunk.get("pages", ""))) or "—"
        heading = f"{identifier} · {section} · стр. {page}"
        with st.expander(heading) if expandable else nullcontext():
            if not expandable:
                st.markdown(f"**{heading}**")
            st.write(chunk.get("text", chunk.get("content", "Текст не найден")))
            url = chunk.get("source_url", chunk.get("url")) or chunk.get("source", {}).get("url")
            if url:
                if isinstance(page, int) and ".pdf" in url.lower():
                    url = url.split("#", 1)[0] + f"#page={page}"
                st.markdown(f"[Открыть исходный документ]({url})")
            st.caption(f"{chunk.get('display_role', 'Источник')}. Происхождение: {chunk.get('source_type', chunk.get('origin', 'клинические рекомендации'))}; версия: {chunk.get('source_version', chunk.get('version', '2024'))}")


def plan_view(plan: dict, chunks: dict):
    st.markdown(f"**Версия {plan.get('version')} · день {plan.get('day')}**")
    st.write(plan.get("rationale", ""))
    st.caption(f"Происхождение плана: {ORIGIN.get(plan.get('origin'), plan.get('origin', 'не указано'))}")
    if plan.get("missing_data"):
        st.warning("Недостающие данные: " + "; ".join(plan["missing_data"]))
    actions = plan.get("actions", [])
    rows = [{"Действие": a.get("title"), "Тип": CATEGORY.get(a.get("category"), a.get("category")), "Доза": a.get("dose") or "—", "Введение": a.get("route") or "—", "Частота": a.get("frequency") or "—", "Статус": STATUS.get(a.get("status"), a.get("status")), "Основание": ", ".join(a.get("evidence_ids", []))} for a in actions]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    for action in actions:
        with st.expander(f"Обоснование: {action.get('title')}"):
            st.write(action.get("indication", ""))
            sources(action.get("evidence_ids", []), chunks, expandable=False)


def observation_table(patient: dict, checkpoints: list[dict]) -> pd.DataFrame:
    unique = {}
    observations = patient.get("observations", []) + [o for c in checkpoints for o in c.get("patient", {}).get("observations", [])]
    for obs in observations:
        unique[(obs.get("day", 0), obs.get("name"), obs.get("unit"))] = obs.get("value")
    rows = [{"День": day, "Показатель": METRIC.get(name, name), "Единица": unit, "Значение": value} for (day, name, unit), value in unique.items()]
    return pd.DataFrame(rows)


def actions_diff(first: dict, last: dict) -> list[dict]:
    before = {a.get("catalog_id", a.get("id")): a for a in first.get("actions", [])}
    after = {a.get("catalog_id", a.get("id")): a for a in last.get("actions", [])}
    result = []
    for key in sorted(before.keys() | after.keys()):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        material = ("dose", "route", "frequency", "status")
        if old and new and all(old.get(k) == new.get(k) for k in material):
            continue
        label = "Добавлено" if old is None else "Удалено из активного плана" if new is None else "Изменено"
        result.append({"Действие": (new or old).get("title"), "Изменение": label, "Было": "—" if old is None else " · ".join(str(old.get(k) or "—") for k in material), "Стало": "—" if new is None else " · ".join(str(new.get(k) or "—") for k in material), "Причина": (new or old).get("indication", ""), "Источники": ", ".join((new or old).get("evidence_ids", []))})
    return result


def run_demo(mode: str):
    if PUBLIC_DEMO:
        st.error("Публичная демонстрация предназначена для просмотра сохранённых результатов.")
        return
    target = ROOT / "reports" / mode
    try:
        with st.spinner("Выполняется демонстрация трёх пациентов…"):
            result = subprocess.run([sys.executable, "-m", "aimedicine.cli", "demo", "--mode", mode, "--output", str(target)], cwd=ROOT, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        st.error("Запуск превысил 15 минут и остановлен. Проверьте сохранённые отчёты перед повтором.")
        return
    if result.returncode:
        st.error("Запуск завершился с ошибкой. Подробности проверьте в локальном терминале; ответы провайдера не выводятся в интерфейсе.")
    else:
        st.success("Демонстрация сохранена. Обновите список результатов кнопкой ниже.")


st.title("AIMedicine")
st.caption("Объяснимый план → состояние на третий день → пересмотр → аудит")
st.caption("Три синтетических случая внебольничной пневмонии · сценарная динамика на третий день")

with st.sidebar:
    st.header("Сохранённые результаты")
    if st.button("Обновить список"):
        st.rerun()
    run_dirs = sorted({p.parent for p in (ROOT / "reports").rglob("*.json") if p.name not in {"summary.json", "evaluation.json", "usage.json", "run_error.json", "run_status.json"}}, key=lambda p: (0 if p.name == "live" and completed_directory(p) else 1 if p.name == "offline" and completed_directory(p) else 2, str(p)))
    if PUBLIC_DEMO:
        allowed = {ROOT / "reports" / name for name in ("live", "offline", "evaluation", "evaluation_live")}
        run_dirs = [folder for folder in run_dirs if folder in allowed]
    selected_dir = st.selectbox("Папка запуска", run_dirs, format_func=lambda p: str(p.relative_to(ROOT)), key="run_directory") if run_dirs else None
    st.divider()
    if PUBLIC_DEMO:
        st.caption("Планы, динамика, источники и аудит завершённых запусков.")
    else:
        with st.expander("Создать новый запуск"):
            st.caption("Offline выполняет эталонные сценарии. Live использует Mistral для планирования и аудита.")
            if st.button("Запустить offline"):
                run_demo("offline")
            key_available = bool(os.environ.get("MISTRAL_API_KEY", "").strip())
            st.caption(f"Планировщик: {os.environ.get('MISTRAL_MODEL', 'ministral-8b-latest')}")
            st.caption(f"Аудитор: {os.environ.get('MISTRAL_AUDITOR_MODEL', 'ministral-14b-latest')}")
            confirmed = st.checkbox("Подтверждаю новый live-запуск трёх случаев")
            if not key_available:
                st.caption("Для Live нужен MISTRAL_API_KEY в окружении или локальном .env.")
            if st.button("Запустить live", disabled=not (key_available and confirmed)):
                run_demo("live")

if selected_dir is None:
    st.warning("Сохранённых результатов пока нет. Создайте offline-запуск в боковой панели или выполните команду из README.")
    st.stop()

traces = []
for path in sorted(selected_dir.glob("*.json")):
    try:
        candidate = read_json(path)
        if isinstance(candidate, dict) and "patient" in candidate and "plans" in candidate:
            traces.append((path, candidate))
    except (OSError, json.JSONDecodeError):
        st.warning(f"Не удалось прочитать {path.name}.")
if not traces:
    st.warning("В выбранной папке нет историй пациентов.")
    st.stop()

selection = st.selectbox("Пациент", range(len(traces)), format_func=lambda i: f"{traces[i][1]['patient'].get('id', traces[i][0].stem)} · {traces[i][1]['patient'].get('age', '?')} лет · {traces[i][0].stem}", key="patient_selection")
path, trace = traces[selection]
if trace.get("metadata", {}).get("fault_injection"):
    st.warning("Проверочный вариант с намеренно внесённым дефектом. Это тест аудитора, отдельный от трёх клинических сценариев.")
elif not completed_directory(selected_dir):
    st.warning("Для этой папки не подтверждён полный завершённый прогон. Показан отдельный сохранённый результат.")
if trace.get("metadata", {}).get("requires_human_review"):
    st.warning("В этом результате требуется проверка врача: изучите открытые замечания, неподтверждённые гипотезы LLM, заблокированные действия и недостающие данные.")
patient = trace["patient"]
chunks = load_chunks()
plans = trace.get("plans", [])
checkpoints = trace.get("checkpoints", [])
st.caption(f"{MODE.get(trace.get('mode'), trace.get('mode'))} · запуск {trace.get('run_id')} · {path.name}")
col1, col2, col3 = st.columns(3)
col1.metric("Возраст", patient.get("age", "—"))
col2.metric("Версий плана", len(plans))
col3.metric("Контрольных точек", len(checkpoints))
tabs = st.tabs(["Пациент и динамика", "Планы и изменения", "Аудитор", "История и источники"])
with tabs[0]:
    st.write(f"**Диагноз:** {patient.get('diagnosis')}")
    st.write(f"**Причина госпитализации:** {patient.get('hospitalization_reason')}")
    allergy = patient.get("allergies")
    st.write("**Аллергии:** " + ("неизвестны" if allergy is None else ", ".join(allergy) or "нет по данным сценария"))
    st.write("**Сопутствующие заболевания:** " + (", ".join(patient.get("comorbidities", [])) or "нет в сценарии"))
    st.write("**Исходные препараты:** " + (", ".join(patient.get("current_medications", [])) or "нет в сценарии"))
    frame = observation_table(patient, checkpoints)
    if not frame.empty:
        st.dataframe(frame.pivot_table(index=["Показатель", "Единица"], columns="День", values="Значение", aggfunc="last"), use_container_width=True)
        metric = st.selectbox("Показатель для графика", list(frame["Показатель"].drop_duplicates()))
        selected = frame[frame["Показатель"] == metric].sort_values("День")
        st.caption(f"{metric} · {selected['Единица'].iloc[0]}; точки связаны только для сравнения, промежуточные значения не рассчитаны.")
        st.line_chart(selected.set_index("День")[["Значение"]])
    for checkpoint in checkpoints:
        st.markdown(f"**События дня {checkpoint.get('day')}**")
        for event in checkpoint.get("events", []):
            st.write(f"• {event.get('description')}")
        with st.expander("Допущения симуляции"):
            st.write(checkpoint.get("assumptions", []))
            st.caption(f"Симулятор {checkpoint.get('simulator')}; seed={checkpoint.get('seed')}")
with tabs[1]:
    if plans:
        plan_tabs = st.tabs([f"Версия {p.get('version')} / день {p.get('day')}" for p in plans])
        for tab, plan in zip(plan_tabs, plans):
            with tab:
                plan_view(plan, chunks)
        if len(plans) > 1:
            st.subheader("Изменения от первого к последнему плану")
            changes = actions_diff(plans[0], plans[-1])
            if changes:
                st.dataframe(pd.DataFrame(changes), hide_index=True, use_container_width=True)
            else:
                st.write("Состав и режим действий не изменились.")
with tabs[2]:
    audits = trace.get("audits", [])
    if not audits:
        st.warning("В истории нет сохранённого аудита.")
    for i, audit in enumerate(audits, 1):
        st.markdown(f"**Проверка {i}**")
        st.caption("LLM-проверка выполнена" if audit.get("llm_reviewed") else "Проверка правилами; LLM-аудит не выполнен")
        if audit.get("llm_summary"):
            st.write(audit["llm_summary"])
        findings = audit.get("findings", [])
        unverified = audit.get("unverified_findings", [])
        st.caption(f"Подтверждённые замечания: {len(findings)}; неподтверждённые предположения LLM: {len(unverified)}")
        if not findings and unverified:
            st.warning("Подтверждённых нарушений в покрытых правилах нет, но аудит требует экспертной проверки неподтверждённых предположений LLM.")
        elif not findings:
            st.success("Подтверждённых нарушений в покрытых проверках не обнаружено.")
        for finding in findings:
            title = f"{SEVERITY.get(finding.get('severity'), finding.get('severity'))} · {finding.get('code')} · {STATUS.get(finding.get('status'), finding.get('status'))}"
            with st.expander(title, expanded=finding.get("status") == "open"):
                st.write(finding.get("message"))
                st.write("**Данные пациента:** " + finding.get("patient_evidence", ""))
                st.write("**Исправление:** " + finding.get("recommendation", ""))
                st.caption(f"Происхождение: {finding.get('origin')}; версия плана: {finding.get('plan_version')}; категория: {finding.get('category')}")
                sources(finding.get("evidence_ids", []), chunks, expandable=False)
        if unverified:
            with st.expander(f"Неподтверждённые предположения LLM ({len(unverified)})"):
                st.write("Эти предположения не получили независимого подтверждения по проверяемым правилам и фактам. Они сохранены для экспертной оценки и не являются подтверждёнными нарушениями.")
                for finding in unverified:
                    st.markdown(f"**{finding.get('code')} · версия {finding.get('plan_version')}**")
                    st.write(finding.get("message"))
                    st.write("Заявленное моделью основание: " + finding.get("patient_evidence", ""))
                    sources(finding.get("evidence_ids", []), chunks, expandable=False)
                if audit.get("llm_validation"):
                    st.markdown("**Результаты независимой проверки предположений**")
                    st.dataframe(pd.DataFrame(audit["llm_validation"]).astype(str), hide_index=True, use_container_width=True)
        if audit.get("llm_raw_summary"):
            with st.expander("Необработанное заключение LLM · требует проверки"):
                st.caption("Исходный текст модели сохранён для прозрачности. Он может содержать неподтверждённые утверждения и не заменяет подтверждённые выводы выше.")
                st.write(audit["llm_raw_summary"])
        with st.expander("Покрытие и ограничения проверки"):
            st.write("Правила:", audit.get("checked_rules", []))
            st.write("Ограничения:", audit.get("limitations", []))
with tabs[3]:
    st.subheader("Переходы системы")
    if trace.get("transitions"):
        st.dataframe(pd.DataFrame(trace["transitions"]).astype(str), hide_index=True, use_container_width=True)
    for checkpoint in checkpoints:
        if checkpoint.get("execution_log"):
            st.markdown(f"**Журнал действий симуляции до дня {checkpoint.get('day')}**")
            st.dataframe(pd.DataFrame(checkpoint["execution_log"]), hide_index=True, use_container_width=True)
            st.caption("sequence задаёт порядок событий симуляции внутри дня.")
    retrievals = trace.get("retrievals", [])
    with st.expander("Запросы поиска"):
        st.json(retrievals)
    if any(item.get("retrieved_ids") for item in retrievals):
        with st.expander("Найденные фрагменты документа"):
            retrieval_index = st.selectbox("Этап поиска", range(len(retrievals)), format_func=lambda index: f"{index + 1}. {retrievals[index].get('stage', 'Поиск')} · день {retrievals[index].get('day', '—')}", key="retrieval_selection")
            sources(retrievals[retrieval_index].get("retrieved_ids", []), chunks, expandable=False)
    used_ids = [x for p in plans for a in p.get("actions", []) for x in a.get("evidence_ids", [])]
    used_ids += [x for audit in trace.get("audits", []) for f in audit.get("findings", []) for x in f.get("evidence_ids", [])]
    st.subheader("Использованные источники")
    sources(used_ids, chunks)
    st.download_button("Скачать историю JSON", data=json.dumps(trace, ensure_ascii=False, indent=2), file_name=path.name, mime="application/json")
