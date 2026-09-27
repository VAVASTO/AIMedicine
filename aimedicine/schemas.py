from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Observation(Record):
    name: str
    value: float
    unit: str
    day: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_known_units(self):
        units = {"temperature": "°C", "spo2": "%", "respiratory_rate": "/min", "heart_rate": "/min",
                 "crp": "mg/L", "wbc": "10^9/L", "creatinine": "µmol/L", "egfr": "mL/min/1.73m²",
                 "alt": "U/L", "ast": "U/L", "qtc_ms": "ms", "potassium": "mmol/L",
                 "chest_xray_infiltrate": "boolean"}
        if self.name in units and self.unit != units[self.name]:
            raise ValueError(f"Unsupported unit for {self.name}; expected {units[self.name]}")
        if self.value < 0 or (self.name == "spo2" and self.value > 100):
            raise ValueError("Invalid observation range")
        return self


class Culture(Record):
    organism: str
    collected_day: int
    available_day: int
    susceptibility: dict[str, Literal["S", "I", "R"]]

    @model_validator(mode="after")
    def ordered_dates(self):
        if self.collected_day < 0 or self.available_day < self.collected_day:
            raise ValueError("Culture result cannot precede collection")
        return self


class Patient(Record):
    id: str
    age: int = Field(ge=18, le=110)
    sex: Literal["female", "male"]
    diagnosis: str = "Внебольничная пневмония"
    severity: Literal["nonsevere", "severe"] = "nonsevere"
    hospitalized: bool = True
    hospitalization_reason: str
    risk_resistant_pathogens: bool = False
    allergies: list[str] | None = None
    comorbidities: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    observations: list[Observation]
    culture: Culture | None = None
    consents: dict[str, bool | None] = Field(default_factory=dict)
    symptoms: list[str] = Field(default_factory=list)
    synthetic: Literal[True] = True


def latest(patient: Patient, name: str, day: int | None = None) -> Observation | None:
    values = [x for x in patient.observations if x.name == name and (day is None or x.day <= day)]
    return max(values, key=lambda x: x.day) if values else None


class PlanAction(Record):
    id: str
    catalog_id: str
    title: str
    category: Literal["medication", "investigation", "monitoring", "regimen", "procedure", "review"]
    drug_id: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    status: Literal["proposed", "active", "blocked", "stopped"] = "proposed"
    indication: str
    evidence_ids: list[str]


class Plan(Record):
    version: int
    day: int
    actions: list[PlanAction]
    rationale: str
    missing_data: list[str] = Field(default_factory=list)
    origin: Literal["llm", "fixture", "repair", "rule"]


class Event(Record):
    day: int
    kind: str
    description: str
    observation_names: list[str] = Field(default_factory=list)


class Checkpoint(Record):
    day: int
    patient: Patient
    events: list[Event]
    execution_log: list[dict] = Field(default_factory=list)
    simulator: str = "scenario-rules-v1"
    seed: int
    assumptions: list[str]


class Finding(Record):
    code: str
    severity: Literal["critical", "warning", "info"]
    category: Literal["clinical", "interaction", "procedural", "citation", "uncertainty"]
    plan_version: int
    action_id: str | None = None
    message: str
    evidence_ids: list[str] = Field(default_factory=list)
    patient_evidence: str
    recommendation: str
    origin: Literal["rule", "llm"]
    status: Literal["open", "resolved", "historical"] = "open"


class AuditReport(Record):
    findings: list[Finding]
    checked_rules: list[str]
    limitations: list[str]
    llm_reviewed: bool = False
    llm_summary: str = ""
    llm_raw_summary: str = ""
    unverified_findings: list[Finding] = Field(default_factory=list)
    llm_validation: list[dict] = Field(default_factory=list)


class Trace(Record):
    patient: Patient
    plans: list[Plan]
    checkpoints: list[Checkpoint]
    audits: list[AuditReport] = Field(default_factory=list)
    retrievals: list[dict] = Field(default_factory=list)
    transitions: list[dict] = Field(default_factory=list)
    mode: Literal["live", "offline", "replay"]
    run_id: str
    metadata: dict = Field(default_factory=dict)


class PlanSelection(Record):
    selected_action_ids: list[str]
    rationale: str
    rationale_by_action: dict[str, str]
    evidence_by_action: dict[str, list[str]]
    missing_data: list[str] = Field(default_factory=list)


class ReplanSelection(PlanSelection):
    selected_action_ids: list[Literal[
        "ampicillin_iv", "amoxiclav_iv", "levofloxacin_iv", "cbc", "biochemistry", "crp",
        "ecg", "sputum_culture", "chest_xray", "pulse_oximetry", "vitals", "hydration",
        "reassess_72h", "susceptibility_review", "oral_switch_review",
    ]]


class LLMCandidate(Finding):
    code: Literal[
        "catalog_parameter_mismatch", "citation_missing", "citation_support_incomplete", "citation_unrelated",
        "action_outside_catalog", "allergy_unknown", "beta_lactam_allergy", "fluoroquinolone_allergy",
        "patient_outside_scope", "mdr_risk_outside_scope", "comorbidity_outside_demo_scope",
        "renal_data_missing", "renal_outside_demo_scope", "hepatic_data_missing", "hepatic_outside_demo_scope",
        "qtc_unknown", "qt_risk", "potassium_unknown", "hypokalemia_qt_risk",
        "levofloxacin_amiodarone", "amiodarone_washout_unknown", "known_culture_resistance",
        "duplicate_antibiotics", "antibiotic_missing", "required_cbc_missing", "required_biochemistry_missing",
        "required_crp_missing", "required_ecg_missing", "required_pulse_oximetry_missing",
        "required_sputum_culture_missing", "procedure_consent_missing", "contrast_renal_precondition_missing",
        "unverified_other"
    ]


class LLMAudit(Record):
    summary: str
    findings: list[LLMCandidate]
    limitations: list[str]
