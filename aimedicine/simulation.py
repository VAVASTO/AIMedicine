from __future__ import annotations

import random

from .schemas import Checkpoint, Culture, Event, Observation, Patient, Plan, latest


ANTIBIOTICS = {"ampicillin", "amoxiclav", "levofloxacin"}


def active_antibiotics(plan: Plan) -> set[str]:
    return {
        a.drug_id for a in plan.actions
        if a.category == "medication" and a.status == "active" and a.drug_id in ANTIBIOTICS
    }


def simulate(patient: Patient, plan: Plan, scenario: dict, seed: int = 42) -> Checkpoint:


    day = int(scenario.get("day", 3))
    if day <= plan.day:
        raise ValueError("Checkpoint must occur after the supplied treatment plan")
    rng = random.Random(seed)
    result = patient.model_copy(deep=True)
    active = active_antibiotics(plan)
    
    culture_order = next((a for a in plan.actions if a.catalog_id == "sputum_culture"
                          and a.category == "investigation" and a.status == "active"), None)
    execution_log = []
    if culture_order is not None:
        execution_log.append({"day": plan.day, "sequence": 1, "event": "sputum_collected",
                              "action_id": culture_order.id, "synthetic": True})
    for action in plan.actions:
        if action.category == "medication" and action.status == "active":
            execution_log.append({"day": plan.day, "sequence": len(execution_log) + 1,
                                  "event": "medication_started", "drug_id": action.drug_id,
                                  "action_id": action.id, "synthetic": True})
    hidden_culture = scenario.get("culture")
    culture_mode = hidden_culture.get("mode", "fixed") if hidden_culture else "fixed"
    if culture_mode not in {"fixed", "resistant_to_active"}:
        raise ValueError(f"Unknown scenario culture mode: {culture_mode}")
    susceptibility = dict(hidden_culture.get("susceptibility", {})) if hidden_culture else {}
    if culture_mode == "resistant_to_active":
        susceptibility = {drug: "R" if drug in active else "S" for drug in sorted(ANTIBIOTICS)}
    effective = bool(active) and any(susceptibility.get(drug, "S") == "S" for drug in active)
    
    allergy = set(patient.allergies or [])
    beta_allergy = bool(allergy & {"beta_lactam_immediate", "beta_lactam", "immediate_beta_lactam", "penicillin_anaphylaxis"})
    contraindicated = (beta_allergy and bool(active & {"ampicillin", "amoxiclav"})) or (
        bool({"fluoroquinolone", "levofloxacin"} & allergy) and "levofloxacin" in active
    )
    allergy_unknown = patient.allergies is None
    if contraindicated or allergy_unknown:
        effective = False


    jitter = rng.uniform(-0.05, 0.05)
    targets = ({"temperature": 37.0 + jitter, "spo2": 96.0, "respiratory_rate": 18.0,
                "heart_rate": 84.0, "crp": 42.0, "wbc": 9.0}
               if effective else
               {"temperature": 38.6 + jitter, "spo2": 93.0, "respiratory_rate": 25.0,
                "heart_rate": 106.0, "crp": 150.0, "wbc": 15.2})
    units = {"temperature": "°C", "spo2": "%", "respiratory_rate": "/min",
             "heart_rate": "/min", "crp": "mg/L", "wbc": "10^9/L"}
    for name, value in targets.items():
        result.observations.append(Observation(name=name, value=round(value, 2), unit=units[name], day=day))
    result.symptoms = (["Кашель сохраняется, выраженность меньше", "Одышка уменьшилась"]
                       if effective else ["Лихорадка сохраняется", "Кашель и одышка без улучшения"])

    if hidden_culture and culture_order is not None and plan.day < day:
        result.culture = Culture(
            organism=hidden_culture["organism"], collected_day=plan.day,
            available_day=day, susceptibility=susceptibility,
        )
    provisional = Checkpoint(day=day, patient=result, events=[], execution_log=execution_log, seed=seed, assumptions=[
        "Синтетический сценарий для проверки программной логики; физиологическая и прогностическая валидность не установлена.",
        "Изменения показателей заданы иллюстративными правилами; случайная вариация воспроизводима по seed.",
        "Экспозиция учитывает только active-препараты в переданном полном снимке плана.",
        "Посев доступен при выполненном active-назначении sputum_culture до контрольного дня.",
        "В синтетическом execution_log забор мокроты предшествует началу active-препаратов; согласия не создаются.",
        "Статус active означает действие внутри симуляции.",
    ])
    if culture_mode == "resistant_to_active":
        provisional.assumptions.append(
            "Условие сценария: возбудитель устойчив к каждому активному антибиотику из каталога "
            "и чувствителен к остальным. Это заданный результат посева, не прогноз возникновения резистентности."
        )
    provisional.events = needs_revision(patient, provisional, plan)
    if contraindicated:
        provisional.events.append(Event(day=day, kind="contraindicated_exposure",
                                        description="Симулятор обнаружил активный препарат при известной аллергии; требуется немедленная проверка плана."))
    if allergy_unknown and active:
        provisional.events.append(Event(day=day, kind="allergy_information_missing",
                                        description="Аллергологический анамнез неизвестен; симулятор не подтверждает безопасный ответ на назначенную терапию."))
    return provisional


def needs_revision(before: Patient, checkpoint: Checkpoint, plan: Plan) -> list[Event]:


    after, day = checkpoint.patient, checkpoint.day
    events: list[Event] = []
    t0, t1 = latest(before, "temperature", plan.day), latest(after, "temperature", day)
    crp0, crp1 = latest(before, "crp", plan.day), latest(after, "crp", day)
    spo2 = latest(after, "spo2", day)
    persistent = t1 is not None and t1.value >= 38.0 and (
        t0 is None or t0.value - t1.value < 0.5
    )
    inflammatory = crp0 is not None and crp1 is not None and crp1.value >= crp0.value
    if day - plan.day >= 3 and (persistent or inflammatory):
        events.append(Event(day=day, kind="lack_of_response",
                            description="К контрольной точке через 72 часа сохраняется лихорадка и/или отсутствует снижение воспалительного показателя: нужна переоценка.",
                            observation_names=["temperature", "crp"]))
    if spo2 is not None and spo2.value < 92:
        events.append(Event(day=day, kind="oxygenation_alert", description="Снижение SpO₂ требует оценки врачом.",
                            observation_names=["spo2"]))
    if not active_antibiotics(plan):
        events.append(Event(day=day, kind="no_active_antibiotic",
                            description="В текущем снимке плана нет активной антибактериальной терапии; ответ на неё не может быть продемонстрирован."))
    culture = after.culture
    if culture and culture.available_day <= day:
        resistant = sorted(drug for drug in active_antibiotics(plan) if culture.susceptibility.get(drug) == "R")
        if resistant:
            events.append(Event(day=day, kind="resistant_culture",
                                description=f"Доступен посев {culture.organism}: устойчивость к действующему препарату {', '.join(resistant)}; требуется пересмотр с учётом антибиотикограммы."))
    return events
