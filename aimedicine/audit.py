from __future__ import annotations

from .schemas import AuditReport, Finding, Patient, Plan, PlanAction, Trace, latest


BETA_ALLERGIES = {"beta_lactam_immediate", "beta_lactam", "immediate_beta_lactam", "penicillin_anaphylaxis"}
ANTIBIOTICS = {"ampicillin", "amoxiclav", "levofloxacin"}
ACTIONABLE = {"active", "proposed"}
RULES = [
    "citation_exists", "catalog_support", "catalog_fields", "allergy_known",
    "beta_lactam_allergy", "fluoroquinolone_allergy", "dose_scope",
    "inpatient_scope", "mdr_scope", "comorbidity_scope", "qt_result_available", "potassium_available", "qt_drug_interaction",
    "culture_resistance_time_aware", "duplicate_antibiotics", "inpatient_diagnostics",
    "procedure_consent", "contrast_renal_precondition", "history_resolution", "antibiotic_present",
]


def patient_at(trace: Trace, day: int) -> Patient:

    eligible = [c for c in trace.checkpoints if c.day <= day]
    patient = (max(eligible, key=lambda c: c.day).patient if eligible else trace.patient).model_copy(deep=True)
    patient.observations = [o for o in patient.observations if o.day <= day]
    if patient.culture and patient.culture.available_day > day:
        patient.culture = None
    return patient


def _normal_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def _audit_plan(patient: Patient, plan: Plan, chunks: dict[str, dict], catalog: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []

    def emit(code: str, message: str, evidence: list[str], patient_evidence: str,
             recommendation: str, action: PlanAction | None = None,
             severity: str = "critical", category: str = "clinical") -> None:


        findings.append(Finding(
            code=code, severity=severity, category=category, plan_version=plan.version,
            action_id=action.id if action else None, message=message, evidence_ids=evidence,
            patient_evidence=patient_evidence, recommendation=recommendation, origin="rule",
        ))

    active_actions = [a for a in plan.actions if a.status in ACTIONABLE]
    medications = [a for a in active_actions if a.category == "medication"]
    stopped_drugs = {a.drug_id for a in plan.actions if a.status == "stopped" and a.drug_id}
    active_drugs = {a.drug_id for a in medications if a.drug_id}
    concurrent_drugs = (set(patient.current_medications) - stopped_drugs) | active_drugs

    for action in active_actions:
        expected = catalog.get(action.catalog_id)
        missing = [source for source in action.evidence_ids if source not in chunks]
        if missing:
            emit("citation_missing", "Назначение ссылается на отсутствующий фрагмент.", [],
                 f"Отсутствуют: {', '.join(missing)}", "Использовать существующий проверенный источник.",
                 action, category="citation")
        if expected is None:
            emit("action_outside_catalog", "Действие отсутствует в проверенном каталоге прототипа.", [],
                 f"catalog_id={action.catalog_id}", "Передать на ручную проверку; не считать действие обоснованным.",
                 action, category="citation")
        else:
            required = set(expected.get("evidence_ids", []))
            cited = set(action.evidence_ids)
            if not required or not required.issubset(cited):
                emit("citation_support_incomplete", "Ссылки не покрывают проверенные основания действия и его параметров.",
                     sorted(required), f"Не хватает оснований: {', '.join(sorted(required - cited)) or 'каталог не содержит оснований'}",
                     "Восстановить связку действие → условие применения → источник; проверить дозу отдельно.", action,
                     category="citation")
            unrelated = cited - required - set(expected.get("supportive_evidence_ids", []))
            if unrelated:
                emit("citation_unrelated", "К действию привязаны фрагменты без проверенной связи с ним в каталоге.",
                     sorted(required), f"Непроверенная связь: {', '.join(sorted(unrelated))}",
                     "Удалить нерелевантные ссылки либо вручную подтвердить и внести их связь в каталог.", action,
                     category="citation")
            mismatches = [field for field in ("category", "drug_id", "dose", "route", "frequency")
                          if _normal_text(getattr(action, field)) != _normal_text(expected.get(field))]
            if mismatches:
                differences = "; ".join(
                    f"{field}: фактически {getattr(action, field)!r}; в каталоге {expected.get(field)!r}"
                    for field in mismatches
                )
                emit("catalog_parameter_mismatch", "Параметры действия отличаются от проверенного варианта каталога.",
                     sorted(required), f"Не совпадают поля: {differences}",
                     "Не выводить новую дозировку из существующей ссылки; вернуть проверенный вариант или передать врачу.",
                     action, category="citation")

    for action in medications:
        drug = action.drug_id
        if drug not in ANTIBIOTICS:
            continue
        if patient.allergies is None:
            emit("allergy_unknown", "Аллергологический анамнез неизвестен; это не равно отсутствию аллергии.",
                 ["cap-beta-allergy"], "allergies=null", "Уточнить аллергологический анамнез до разрешения назначения.",
                 action, category="uncertainty")
        elif drug in {"ampicillin", "amoxiclav"} and BETA_ALLERGIES.intersection(patient.allergies):
            emit("beta_lactam_allergy", "Предложен бета-лактам при известной немедленной аллергии.",
                 ["cap-beta-allergy"], f"allergies={patient.allergies}; drug={drug}",
                 "Отменить несовместимое назначение и пересмотреть вариант с учётом противопоказания.", action)
        elif drug == "levofloxacin" and {"fluoroquinolone", "levofloxacin"}.intersection(patient.allergies):
            emit("fluoroquinolone_allergy", "Предложен левофлоксацин при известной аллергии на него/его класс.",
                 ["drug-levo-allergy"], f"allergies={patient.allergies}", "Блокировать назначение и передать на клинический пересмотр.", action)

        if patient.comorbidities:
            emit("comorbidity_outside_demo_scope", "Сопутствующие заболевания требуют оценки за пределами ограниченного набора демонстрации.",
                 ["cap-empiric", "demo-policy-comorbidity-scope"], f"comorbidities={patient.comorbidities}",
                 "Передать врачу для оценки влияния сопутствующих заболеваний; это граница прототипа, не универсальное противопоказание антибиотика.",
                 action, category="uncertainty")

        if not patient.hospitalized or patient.severity != "nonsevere":
            emit("patient_outside_scope", "Пациент не соответствует поддерживаемому сценарию нетяжёлой ВП в стационаре.",
                 ["cap-empiric"], f"hospitalized={patient.hospitalized}; severity={patient.severity}",
                 "Передать на ручной выбор соответствующего раздела КР.", action)
        if drug == "ampicillin" and patient.risk_resistant_pathogens:
            emit("mdr_risk_outside_scope", "Выбран ограниченный стартовый вариант при наличии факторов риска устойчивых возбудителей.",
                 ["cap-empiric"], "risk_resistant_pathogens=true", "Пересмотреть тактику за пределами данного демонстрационного варианта.", action)

        renal = {name: latest(patient, name, plan.day) for name in ("creatinine", "egfr")}
        if any(value is None for value in renal.values()):
            emit("renal_data_missing", "Не подтверждена функция почек для применения таблицы доз в заданных условиях.",
                 ["cap-biochemistry", "cap-dose-scope", "demo-policy-dose-scope"],
                 "Нет результата: " + ", ".join(k for k, v in renal.items() if v is None),
                 "Получить результаты; назначенный анализ не является полученным результатом.", action, category="uncertainty")
        elif renal["egfr"].value < 90:
            emit("renal_outside_demo_scope", "Функция почек за пределами консервативной области входных данных прототипа.",
                 ["cap-dose-scope", "demo-policy-dose-scope"], f"egfr={renal['egfr'].value}; область демонстрации egfr>=90",
                 "Передать врачу для проверки дозы; это ограничение прототипа, не универсальное противопоказание.", action,
                 category="uncertainty")
        hepatic = {name: latest(patient, name, plan.day) for name in ("alt", "ast")}
        if any(value is None for value in hepatic.values()):
            emit("hepatic_data_missing", "Нет данных для проверки ограниченной области применения доз по функции печени.",
                 ["cap-biochemistry", "cap-dose-scope", "demo-policy-dose-scope"],
                 "Нет результата: " + ", ".join(k for k, v in hepatic.items() if v is None),
                 "Получить данные и провести клиническую оценку; не объявлять функцию печени нормальной по умолчанию.",
                 action, category="uncertainty")
        elif any(value.value > 40 for value in hepatic.values()):
            emit("hepatic_outside_demo_scope", "Показатели за пределами консервативной области входных данных прототипа.",
                 ["cap-dose-scope", "demo-policy-dose-scope"], "Демо-граница: АЛТ/АСТ <=40 U/L; это не полная оценка функции печени.",
                 "Передать врачу для оценки функции печени и режима дозирования.", action, category="uncertainty")

        if drug == "levofloxacin":
            qtc = latest(patient, "qtc_ms", plan.day)
            if qtc is None:
                emit("qtc_unknown", "Для проверки QT-риска нет полученного результата ЭКГ/QTc.",
                     ["cap-ecg", "cap-qt", "drug-levo-qt"], "qtc_ms отсутствует на момент плана",
                     "Получить и оценить ЭКГ до разрешения назначения; одно назначение ЭКГ недостаточно.", action, category="uncertainty")
            elif qtc.value >= 450:
                emit("qt_risk", "QTc за пределами консервативной области входных данных прототипа.",
                     ["cap-qt", "drug-levo-qt", "demo-policy-qtc-scope"], f"qtc_ms={qtc.value}",
                     "Передать врачу: демонстрация ограничена QTc<450; это не универсальная граница противопоказания или полная оценка QT-риска.",
                     action, category="uncertainty")
            potassium = latest(patient, "potassium", plan.day)
            if potassium is None:
                emit("potassium_unknown", "Не получен уровень калия для проверки риска гипокалиемии.",
                     ["cap-biochemistry", "drug-levo-qt"], "potassium отсутствует на момент плана",
                     "Получить результат калия и оценить риск до разрешения левофлоксацина.", action, category="uncertainty")
            elif potassium.value < 3.5:
                emit("hypokalemia_qt_risk", "Низкий калий увеличивает риск при назначении левофлоксацина.",
                     ["drug-levo-qt", "demo-policy-qtc-scope"], f"potassium={potassium.value} mmol/L; демонстрационная граница <3.5",
                     "Передать врачу для коррекции и проверки назначения; прототип не назначает коррекцию электролитов.", action)
            if "amiodarone" in concurrent_drugs:
                emit("levofloxacin_amiodarone", "Сочетание левофлоксацина и амиодарона повышает риск удлинения QT.",
                     ["drug-levo-qt"], "Одновременно действующие препараты: levofloxacin, amiodarone",
                     "Не разрешать сочетание автоматически; требуется клинический пересмотр.", action, category="interaction")
            elif "amiodarone" in stopped_drugs:
                emit("amiodarone_washout_unknown", "Остановка амиодарона не доказывает исчезновение его длительного эффекта.",
                     ["drug-levo-qt"], "Амиодарон помечен stopped; время отмены и остаточный QT-риск не моделируются.",
                     "Проверить срок отмены и остаточный риск у врача; статус stopped не означает автоматически безопасный переход.",
                     action, severity="warning", category="uncertainty")

        culture = patient.culture
        if culture and culture.available_day <= plan.day and culture.susceptibility.get(drug) == "R":
            emit("known_culture_resistance", "План сохраняет препарат с уже известной устойчивостью выделенного возбудителя.",
                 ["cap-culture-review", "cap-etiotropic"],
                 f"{culture.organism}: {drug}=R; результат доступен на день {culture.available_day}, план на день {plan.day}",
                 "Пересмотреть терапию по реально полученной антибиотикограмме и клинической значимости результата.", action)

    antibiotic_actions = [a for a in medications if a.drug_id in ANTIBIOTICS]
    if not antibiotic_actions:
        emit("antibiotic_missing", "В текущем плане не разрешена антибактериальная терапия подтверждённой ВП.",
             ["cap-empiric"], "Нет active/proposed антибиотика поддерживаемого каталога",
             "Уточнить причины блокировки и передать врачу для решения; не подставлять препарат при неизвестных исходных данных.")
        if patient.comorbidities:
            emit("comorbidity_outside_demo_scope", "Сопутствующие заболевания требуют оценки за пределами ограниченного набора демонстрации.",
                 ["cap-empiric", "demo-policy-comorbidity-scope"], f"comorbidities={patient.comorbidities}",
                 "Передать врачу для оценки влияния сопутствующих заболеваний; это граница прототипа, не универсальное противопоказание антибиотика.",
                 category="uncertainty")
    if len(antibiotic_actions) > 1:
        emit("duplicate_antibiotics", "В данном демонстрационном сценарии одновременно разрешено несколько системных антибиотиков.",
             ["demo-policy-duplicate-antibiotics"], ", ".join(a.drug_id or "" for a in antibiotic_actions),
             "Проверить замещение: предыдущий препарат должен быть stopped. Это правило сценария, не запрет комбинированной терапии в медицине.",
             antibiotic_actions[-1], category="interaction")


    ordered = {a.catalog_id for a in active_actions}
    required = {
        "cbc": (["wbc"], "cap-cbc"),
        "biochemistry": (["creatinine", "alt", "ast"], "cap-biochemistry"),
        "crp": (["crp"], "cap-crp"),
        "ecg": (["qtc_ms"], "cap-ecg"),
        "pulse_oximetry": (["spo2"], "cap-pulse-oximetry"),
    }
    if patient.hospitalized:
        for action_id, (observations, source) in required.items():
            if action_id not in ordered and not all(latest(patient, name, plan.day) for name in observations):
                emit(f"required_{action_id}_missing", "Не отражено обязательное исследование/наблюдение либо его результат.",
                     [source], f"Нет {action_id} и достаточных доступных записей {observations}",
                     f"Добавить {action_id} или документировать уже выполненное исследование.", category="procedural", severity="warning")
        productive_cough = any("мокрот" in symptom.lower() for symptom in patient.symptoms)
        if productive_cough and "sputum_culture" not in ordered and patient.culture is None:
            emit("required_sputum_culture_missing", "При продуктивном кашле не отражено исследование мокроты.",
                 ["cap-sputum-culture", "cap-sputum-before-antibiotics"], "Кашель с мокротой; посев не назначен и результата нет",
                 "Организовать получение образца; не выдавать назначение за полученную антибиотикограмму.", category="procedural", severity="warning")

    for action in active_actions:
        if action.catalog_id in {"contrast_ct", "thoracentesis"}:
            consent = patient.consents.get(action.catalog_id)
            if consent is not True:
                emit("procedure_consent_missing", "Не подтверждено согласие для демонстрационной процедуры.",
                     ["demo-policy-consent"], f"consents[{action.catalog_id}]={consent!r}",
                     "Блокировать процедуру и получить/документировать согласие. Не генерировать согласие при исправлении.", action, category="procedural")
        if action.catalog_id == "contrast_ct":
            absent = [name for name in ("creatinine", "egfr") if latest(patient, name, plan.day) is None]
            if absent:
                emit("contrast_renal_precondition_missing", "Не получены данные для предусмотренной политикой проверки перед контрастом.",
                     ["demo-policy-renal-contrast"], "Нет результата: " + ", ".join(absent),
                     "Блокировать процедуру до получения и оценки данных; порядок — отдельная политика демонстрации, не цитата из КР ВП.",
                     action, category="procedural")
    return findings


def audit_trace(trace: Trace, chunks: list[dict], catalog: list[dict]) -> AuditReport:
    by_chunk = {chunk["id"]: chunk for chunk in chunks}
    by_action = {action["id"]: action for action in catalog}
    plans = sorted(trace.plans, key=lambda p: (p.day, p.version))
    all_findings: list[Finding] = []
    findings_per_plan: list[list[Finding]] = []
    for plan in plans:
        findings = _audit_plan(patient_at(trace, plan.day), plan, by_chunk, by_action)
        findings_per_plan.append(findings)
        all_findings.extend(findings)

    def key(finding: Finding, plan: Plan) -> tuple[str, str | None]:
        action = next((a for a in plan.actions if a.id == finding.action_id), None)
        return finding.code, action.catalog_id if action else None

    if plans:
        latest_keys = {key(f, plans[-1]) for f in findings_per_plan[-1]}
        for plan, findings in zip(plans[:-1], findings_per_plan[:-1]):
            for finding in findings:
                finding.status = "historical" if key(finding, plan) in latest_keys else "resolved"

    referenced = {source for finding in all_findings for source in finding.evidence_ids}
    unavailable = referenced - set(by_chunk)
    if unavailable and plans:
        all_findings.append(Finding(
            code="audit_source_missing", severity="critical", category="citation", plan_version=plans[-1].version,
            message="В загруженном корпусе отсутствуют основания одного или нескольких правил аудитора.",
            patient_evidence=", ".join(sorted(unavailable)), recommendation="Загрузить клинические и дополнительные policy-источники; отчёт пока неполон.",
            origin="rule",
        ))
    return AuditReport(findings=all_findings, checked_rules=RULES, limitations=[
        "Проверяется ограниченный каталог выбранного сценария; отсутствие находок не доказывает клиническую безопасность.",
        "Любой непустой список сопутствующих заболеваний находится за пределами текущих демонстрационных случаев и требует ручной оценки; это не правило о противопоказании антибиотиков при любых заболеваниях.",
        "Связь назначения с источником проверяется по вручную подтверждённому каталогу, а не универсальным доказательством логического следования текста.",
        "Проверка доз покрывает только заранее заданные режимы. eGFR>=90 и АЛТ/АСТ<=40 — консервативные границы демонстрационного набора, не универсальные противопоказания или полная оценка функции органов.",
        "Правило взаимодействий явно покрывает левофлоксацин+амиодарон; полной базы лекарственных взаимодействий нет.",
        "QTc<450 и калий>=3.5 — границы синтетического входного набора; они не доказывают отсутствие риска и не являются полной клинической стратификацией.",
        "КР, инструкция препарата и политики демонстрации имеют отдельные идентификаторы и разную область применимости.",
        "Наблюдения датируются с точностью до дня. При наличии синтетического execution_log его sequence отражает относительный порядок забора мокроты и начала препаратов внутри дня; без журнала этот порядок неизвестен.",
        "Наличие отдельного показателя — ограниченное свидетельство исследования; полнота всех полей анализа и качество образца здесь не валидируются.",
        "Historical отмечает повторяющуюся проблему старой версии; resolved сохраняет исправленную историческую находку. Только open описывает текущую нерешённую проблему.",
    ])
