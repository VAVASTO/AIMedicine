#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://spulmo.ru/upload/KR-vnebolnichnaya-pnevmoniya-u-vzroslyh-2024.pdf"
DOCUMENT_ID = "ru-cap-654_2-2024"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


SPECS = [
    ("cap-history", 13, "2.1", "Анамнез и факторы риска", "У всех пациентов с подозрением на ВП рекомендуется провести оценку жалоб", "Уровень убедительности", ["history", "risk", "анамнез"]),
    ("cap-vitals", 13, "2.2", "Жизненные показатели", "У всех пациентов с ВП рекомендуется провести общий осмотр", "Уровень убедительности", ["vitals", "temperature", "blood_pressure", "respiratory_rate", "heart_rate"]),
    ("cap-pulse-oximetry", 13, "2.2", "Пульсоксиметрия", "Всем пациентам с ВП рекомендуется пульсоксиметрия", "Уровень убедительности", ["monitoring", "spo2", "пульсоксиметрия"]),
    ("cap-cbc", 14, "2.3", "Клинический анализ крови", "Всем пациентам с ВП рекомендуется выполнение общего", "Уровень убедительности", ["investigation", "cbc", "лейкоциты", "анализ крови"]),
    ("cap-biochemistry", 14, "2.3", "Биохимический анализ крови", "Всем госпитализированным пациентам с ВП рекомендуется выполнить анализ крови биохимический", "Уровень убедительности", ["investigation", "biochemistry", "renal", "hepatic", "creatinine", "potassium", "креатинин", "калий"]),
    ("cap-crp", 14, "2.3", "С-реактивный белок", "Всем госпитализированным пациентам с ВП рекомендуется исследование уровня", "Уровень убедительности", ["investigation", "crp", "СРБ"]),
    ("cap-sputum-culture", 15, "2.3.1", "Микроскопия и посев мокроты", "Всем госпитализированным пациентам с ВП рекомендуется микроскопическое", "\n2 ", ["investigation", "sputum_culture", "susceptibility", "мокрота", "посев"]),
    ("cap-sputum-before-antibiotics", 16, "2.3.1, комментарий", "Сбор мокроты до начала антибиотиков", "Образец свободно отделяемой мокроты должен быть получен", "На сегодняшний день", ["procedure", "sputum_culture", "before_antibiotics"]),
    ("cap-culture-interpretation", 16, "2.3.1, комментарий", "Интерпретация посева с учётом качества образца", "Интерпретация результатов культурального исследования мокроты", "Всем госпитализированным пациентам с ВП при наличии плеврального", ["sputum_culture", "contamination", "culture_review"]),
    ("cap-chest-xray", 18, "2.4", "Рентгенография органов грудной клетки", "Всем пациентам с подозрением на ВП рекомендуется обзорная рентгенография", "Уровень убедительности", ["investigation", "chest_xray", "рентгенография"]),
    ("cap-ecg", 19, "2.4", "ЭКГ и безопасный выбор антибиотика", "Всем госпитализированным пациентам с ВП рекомендуется ЭКГ", "Всем пациентам с ВП и подозрением", ["investigation", "ecg", "qt", "ЭКГ"]),
    ("cap-antibiotic-timing", 24, "3.2.1", "Срок начала антибиотикотерапии", "Всем пациентам с определенным диагнозом ВП рекомендуется назначение АБП", "Уровень убедительности", ["antibiotic", "timing", "inpatient"]),
    ("cap-parenteral", 24, "3.2.1", "Парентеральная стартовая терапия в стационаре", "АБТ ВП у госпитализированных пациентов рекомендуется начинать", "Стартовую АБТ ВП рекомендуется", ["antibiotic", "route", "inpatient", "intravenous"]),
    ("cap-risk-stratification", 24, "3.2.1, комментарий", "Группы риска редких и полирезистентных возбудителей", "К первой группе относят пациентов", "Пациентам с ВП без значимых", ["risk", "mdr", "inpatient"]),
    ("cap-empiric", 24, "3.2.1", "Первая линия при нетяжёлой ВП без факторов риска", "Пациентам с ВП без значимых сопутствующих заболеваний", "Уровень убедительности", ["antibiotic", "empiric", "ampicillin", "amoxiclav", "nonsevere", "ампициллин"]),
    ("cap-beta-allergy", 25, "3.2.1, комментарий", "Альтернатива при немедленной аллергии на бета-лактамы", "Комментарии: Наиболее частыми", "Пациентам с ВП, значимыми", ["allergy", "beta_lactam", "immediate", "fluoroquinolone", "levofloxacin", "аллергия"]),
    ("cap-no-routine-combination", 25, "3.2.1, комментарий", "Отсутствие показаний к рутинной комбинации при нетяжёлой ВП", "Несмотря на наличие когортных", "В случае госпитализации", ["antibiotic", "combination", "nonsevere"]),
    ("cap-inpatient-table", 25, "3.2.1, таблица 8", "Нетяжёлая ВП в стационаре: препараты и пути введения", "Таблица 8.", "Нетяжелая ВП у пациентов с сопутствующими", ["antibiotic", "nonsevere", "ampicillin", "amoxiclav", "levofloxacin", "route"]),
    ("cap-reassess-72h", 28, "3.2.1", "Оценка эффективности и безопасности через 48–72 часа", "Всем пациентам с ВП через 48-72 ч", "Если у пациента сохраняется", ["review", "reassess_72h", "dynamics", "день 3", "48-72"]),
    ("cap-treatment-failure", 28, "3.2.1, комментарий", "Признаки неэффективности и повторная оценка тяжести", "Если у пациента сохраняется", "При неэффективности АБТ на втором этапе", ["review", "failure", "adverse_event", "complication", "лихорадка", "неэффективность"]),
    ("cap-culture-review", 28, "3.2.1, комментарий", "Пересмотр лечения по результатам микробиологии", "При неэффективности АБТ на втором этапе", "В исследованиях пациентов ОРИТ", ["review", "culture", "susceptibility", "resistance", "deescalation", "посев", "чувствительность"]),
    ("cap-crp-dynamics", 28, "3.2.1, комментарий", "Динамика СРБ на третий–четвёртый день", "Из лабораторных тестов целесообразно определение СРБ", "Всем госпитализированным пациентам", ["crp", "dynamics", "day3", "СРБ"]),
    ("cap-oral-switch-1", 28, "3.2.1, критерии клинической стабильности (начало)", "Переход на приём внутрь: все критерии обязательны", "Всем госпитализированным пациентам с ВП рекомендуется перевод", None, ["oral_switch", "clinical_stability", "temperature"]),
    ("cap-oral-switch-2", 29, "3.2.1, критерии клинической стабильности (продолжение)", "Переход на приём внутрь: дыхание, гемодинамика и всасывание", "- частота дыхания", "Уровень убедительности", ["oral_switch", "clinical_stability", "spo2", "absorption"]),
    ("cap-etiotropic", 30, "3.2.1.1", "Этиотропная антибактериальная терапия", "3.2.1.1. Этиотропная АБТ", "3.2.2.", ["review", "susceptibility", "pathogen", "etiotropic", "этиотропная"]),
    ("cap-amoxiclav-haemophilus", 61, "Приложение А3, характеристика основных классов ПМП", "Амоксициллин/клавуланат и H. influenzae", "Преимуществом амоксициллина+клавулановая кислота", "Ключевыми препаратами", ["amoxiclav", "haemophilus", "beta_lactamase", "H. influenzae"]),
    ("cap-fluoroquinolones", 62, "Приложение А3, фторхинолоны", "Респираторные фторхинолоны", "Среди препаратов данной группы наибольшее значение при ВП имеют левофлоксацин", "Хорошие микробиологические характеристики", ["levofloxacin", "fluoroquinolone", "левофлоксацин"]),
    ("cap-qt", 63, "Приложение А3, препараты других групп", "Удлинённый QTc ограничивает применение фторхинолонов", "Среди тетрациклинов", "Ванкомицин", ["qt", "safety", "fluoroquinolone", "contraindication"]),
    ("cap-dose-scope", 64, "Приложение А3, режимы дозирования АМП", "Область применимости таблицы доз", "Режимы дозирования АМП", "Наименование АМП Режим дозирования", ["dose", "renal", "hepatic", "normal_function"]),
    ("cap-dose-amoxiclav", 64, "Приложение А3, режимы дозирования АМП", "Доза амоксициллина/клавулановой кислоты", "Амоксициллин+клавулановая кислота** 0,5", "Ампициллин** 2,0", ["dose", "amoxiclav"]),
    ("cap-dose-ampicillin", 64, "Приложение А3, режимы дозирования АМП", "Доза ампициллина", "Ампициллин** 2,0", "ампициллин+сульбактам**", ["dose", "ampicillin"]),
    ("cap-dose-levofloxacin", 64, "Приложение А3, режимы дозирования АМП", "Доза левофлоксацина", "Левофлоксацин** 0,5", "Линезолид** 0,6", ["dose", "levofloxacin"]),
    ("cap-hydration", 68, "Приложение В, информация для пациентов", "Жидкость и ограничение чрезмерной нагрузки", "При пневмонии рекомендуется также", "Для того, чтобы предупредить", ["regimen", "hydration", "activity", "жидкость", "нагрузка"]),
]


def marker_position(raw: str, marker: str, offset: int = 0) -> int:


    exact = raw.find(marker, offset)
    if exact >= 0:
        return exact
    pattern = r"\s*".join(re.escape(char) for char in marker if not char.isspace())
    if marker[:1].isspace():
        pattern = r"\s+" + pattern
    if marker[-1:].isspace():
        pattern += r"\s+"
    match = re.search(pattern, raw[offset:])
    if match is None:
        raise ValueError(f"Source selector not found: {marker!r}")
    return offset + match.start()


def build_chunks(pages: list[dict]) -> list[dict]:
    page_map = {p["page"]: p["text"] for p in pages}
    chunks = []
    for cid, page, section, title, start, stop, tags in SPECS:
        raw = page_map[page]
        left = marker_position(raw, start)
        right = marker_position(raw, stop, left + 1) if stop else len(raw)
        excerpt = raw[left:right].strip()
        if not excerpt or excerpt not in raw:
            raise ValueError(f"Invalid exact source selector: {cid}")
        chunks.append({"id": cid, "document_id": DOCUMENT_ID, "title": title,
                       "section": section, "page": page, "text": excerpt,
                       "tags": tags, "source_url": SOURCE_URL,
                       "source_type": "guideline", "version": "2024"})
    return chunks


def catalog() -> list[dict]:
    items = []

    def add(cid, title, category, evidence_ids, description, requirements=(),
            drug_id=None, dose=None, route=None, frequency=None):
        items.append(dict(id=cid, title=title, category=category, drug_id=drug_id,
                          dose=dose, route=route, frequency=frequency,
                          evidence_ids=list(evidence_ids), requirements=list(requirements),
                          description=description))

    med_scope = ("adult", "nonsevere_inpatient", "normal_renal_function", "normal_hepatic_function")
    add("ampicillin_iv", "Ампициллин", "medication",
        ["cap-empiric", "cap-parenteral", "cap-beta-allergy", "cap-dose-ampicillin", "cap-dose-scope"],
        "Стартовый вариант нетяжёлой ВП без значимых сопутствующих заболеваний и факторов риска редких/полирезистентных возбудителей. Исключить немедленную аллергию на бета-лактамы. Доза из таблицы для нормальной функции печени и почек; схема не покрывает коррекцию при органной недостаточности.",
        med_scope + ("no_immediate_beta_lactam_allergy", "no_mdr_risk"),
        "ampicillin", "2,0 г", "в/в", "каждые 6 ч")
    add("amoxiclav_iv", "Амоксициллин + клавулановая кислота", "medication",
        ["cap-empiric", "cap-parenteral", "cap-dose-amoxiclav", "cap-dose-scope"],
        "Препарат первой линии при нетяжёлой ВП. При этиотропном выборе требуется документированная чувствительность выделенного клинически значимого возбудителя; учитывать качество посева и клиническую картину. Выбран интервал 8 ч из указанного в источнике диапазона 6–8 ч. 1,2 г — общая доза комбинации из таблицы.",
        med_scope + ("no_immediate_beta_lactam_allergy",),
        "amoxiclav", "1,2 г", "в/в", "каждые 8 ч")
    add("levofloxacin_iv", "Левофлоксацин", "medication",
        ["cap-beta-allergy", "cap-inpatient-table", "cap-fluoroquinolones", "cap-ecg", "cap-qt", "cap-dose-levofloxacin", "cap-dose-scope"],
        "Альтернатива при невозможности назначить пенициллины, включая немедленную аллергию на бета-лактамы, если нет собственных противопоказаний и лекарственных конфликтов. Проверить функцию почек, ЭКГ, QTc и сопутствующие препараты. В этой таблице 0,75 г каждые 24 ч относится только к приёму внутрь; внутривенно выбрана явно указанная схема 0,5 г каждые 12 ч.",
        med_scope + ("no_fluoroquinolone_allergy", "no_qt_risk"),
        "levofloxacin", "0,5 г", "в/в", "каждые 12 ч")
    add("cbc", "Общий (клинический) анализ крови", "investigation", ["cap-cbc"],
        "Эритроциты, гематокрит, лейкоциты, тромбоциты, лейкоцитарная формула; не считать назначение полученным результатом.")
    add("biochemistry", "Биохимический анализ крови", "investigation", ["cap-biochemistry"],
        "Мочевина, креатинин, билирубин, глюкоза, альбумин, натрий, калий, хлориды, АСТ и АЛТ. Учитывать результаты при выборе препарата и дозы.")
    add("crp", "С-реактивный белок", "investigation", ["cap-crp", "cap-crp-dynamics"],
        "Исходная оценка и повторная оценка на 3–4 день; динамика СРБ рассматривается вместе с клинической картиной.")
    add("ecg", "ЭКГ в стандартных отведениях", "investigation", ["cap-ecg"],
        "Для выявления осложнений, сопутствующих заболеваний и безопасного выбора антибиотика; при неизвестном QTc нельзя выдумывать нормальный результат.")
    add("sputum_culture", "Микроскопия и посев мокроты с чувствительностью", "investigation",
        ["cap-sputum-culture", "cap-sputum-before-antibiotics", "cap-culture-interpretation"],
        "При продуктивном кашле получить образец как можно раньше и до АБТ. Оценивать качество образца, микроскопию и клиническую значимость результата; не подменять назначение полученным посевом.")
    add("chest_xray", "Рентгенография органов грудной клетки", "investigation", ["cap-chest-xray"],
        "Для верификации диагноза, оценки распространённости и осложнений. При уже полученном подтверждающем снимке повторный снимок автоматически не назначается.")
    add("pulse_oximetry", "Пульсоксиметрия", "monitoring", ["cap-pulse-oximetry"],
        "Измерить SpO2 для выявления гипоксемии; не придумывать частоту измерений, не указанную в данном фрагменте.")
    add("vitals", "Контроль жизненных показателей", "monitoring", ["cap-vitals"],
        "Температура, ЧДД, ЧСС и артериальное давление; оценка вместе с состоянием сознания и жалобами.")
    add("hydration", "Достаточное питьё и ограничение чрезмерной нагрузки", "regimen", ["cap-hydration"],
        "Общая рекомендация из информации для пациента; источник не устанавливает объём жидкости. Индивидуальный объём и ограничения определяет врач с учётом состояния; внутривенные инфузии автоматически не назначаются.")
    add("reassess_72h", "Оценка эффективности и безопасности через 48–72 часа", "review",
        ["cap-reassess-72h", "cap-treatment-failure", "cap-crp-dynamics"],
        "Сопоставить температуру, интоксикацию, дыхательные симптомы, СРБ и переносимость. При ухудшении повторно оценить тяжесть, осложнения и уровень помощи.")
    add("susceptibility_review", "Пересмотр лечения по результату посева", "review",
        ["cap-culture-review", "cap-culture-interpretation", "cap-etiotropic"],
        "Использовать только реально полученный клинически значимый результат с чувствительностью. При подтверждённой устойчивости текущего препарата пересмотреть лечение; не выводить чувствительность из одного имени возбудителя.")
    add("oral_switch_review", "Проверка возможности перехода на приём внутрь", "review",
        ["cap-oral-switch-1", "cap-oral-switch-2"],
        "Требуются ВСЕ критерии: температура <37,8 при двух измерениях через 8ч, ясное сознание, ЧДД <24, ЧСС <100, систолическое АД >90, SpO2 >90% или PaO2 >60 и отсутствие нарушений всасывания. При нехватке данных зафиксировать неопределённость; одно измерение температуры не даёт права автоматически сменить путь введения.")
    next(item for item in items if item["id"] == "amoxiclav_iv")["supportive_evidence_ids"] = ["cap-etiotropic", "cap-culture-review", "cap-amoxiclav-haemophilus"]
    for item in items:
        if item["category"] == "medication":
            item["supportive_evidence_ids"] = list(dict.fromkeys(item.get("supportive_evidence_ids", []) + ["cap-inpatient-table", "cap-beta-allergy"]))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="Re-extract the bundled PDF with pypdf before building.")
    args = parser.parse_args()
    pdf = ROOT / "data/source/cap_2024.pdf"
    page_file = ROOT / "data/source/extracted_pages.json"
    manifest_path = ROOT / "data/source/manifest.json"
    parser_info = {"name": "pypdf", "version": "6.10.0", "method": "PdfReader.pages[n].extract_text()"}
    if manifest_path.exists():
        parser_info = json.loads(manifest_path.read_text(encoding="utf-8")).get("parser", parser_info)
    if args.extract:
        import pypdf
        from pypdf import PdfReader
        pages = [{"page": i + 1, "text": p.extract_text()}
                 for i, p in enumerate(PdfReader(pdf).pages)]
        parser_info = {"name": "pypdf", "version": pypdf.__version__, "method": "PdfReader.pages[n].extract_text()"}
    else:
        pages = json.loads(page_file.read_text(encoding="utf-8"))
    chunks = build_chunks(pages)
    if args.extract:
        
        write_json(page_file, pages)
    
    chunk_path = ROOT / "data/chunks.json"
    extras = []
    if chunk_path.exists():
        extras = [c for c in json.loads(chunk_path.read_text(encoding="utf-8"))
                  if c.get("document_id") != DOCUMENT_ID]
    write_json(chunk_path, chunks + extras)
    catalog_path = ROOT / "data/catalog.json"
    entries = catalog()
    ids = {x["id"] for x in entries}
    extra_entries = []
    if catalog_path.exists():
        extra_entries = [x for x in json.loads(catalog_path.read_text(encoding="utf-8")) if x["id"] not in ids]
    write_json(catalog_path, entries + extra_entries)
    manifest = {
        "document_id": DOCUMENT_ID,
        "title": "Внебольничная пневмония у взрослых",
        "version": "2024", "guideline_id": "654_2", "source_url": SOURCE_URL,
        "source_file": "data/source/cap_2024.pdf",
        "retrieved_at": "2026-09-27", "page_count": len(pages),
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "extracted_pages_sha256": hashlib.sha256(page_file.read_bytes()).hexdigest(),
        "parser": parser_info,
        "chunk_count": len(chunks), "page_numbering": "Physical PDF page, one-based; not an invented recommendation number.",
        "scope": "Selected exact excerpts for adult nonsevere inpatient community-acquired pneumonia; three educational synthetic scenarios, not full guideline coverage.",
        "dose_scope": "The dose table explicitly applies only to normal liver and kidney function.",
        "source_quality": "Raw extracted excerpts preserve PDF word-concatenation and displaced sub/superscripts. Original PDF is authoritative. Key allergy table p25 and dose table p64 visually checked against rendered pages.",
        "limitations": ["The guideline states review no later than 2026; this project pins the supplied 2024 edition and does not assert it remains the latest guideline.",
                       "Chunk titles and tags are retrieval metadata. Only text is a verbatim excerpt.",
                       "Drug-drug interaction labels and assignment-only procedure policies must have separate provenance."]
    }
    write_json(ROOT / "data/source/manifest.json", manifest)
    print(f"Built {len(chunks)} guideline excerpts and {len(entries)} action definitions.")


if __name__ == "__main__":
    main()
