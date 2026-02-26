from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ModalityName = Literal["text", "table", "figure", "supplement"]
SectionLabel = Literal["introduction", "methods", "results", "discussion", "conclusion", "unknown"]
SectionSource = Literal[
    "meta",
    "structured_abstract",
    "anchor",
    "statement_prefix",
    "category",
    "heading",
    "position",
    "lexical",
    "fallback",
]
ResultEvidenceType = Literal["text_primary", "media_support"]


class ModalityEvidence(BaseModel):
    finding_id: str
    modality: ModalityName
    anchor: str
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)
    value: float | None = None
    unit: str | None = None
    p_value: float | None = None
    effect_size: float | None = None
    category: str = "other"
    section_label: SectionLabel = "unknown"
    section_confidence: float = 0.0
    section_source: SectionSource = "fallback"
    result_evidence_type: ResultEvidenceType | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)

    @field_validator("section_confidence")
    @classmethod
    def _clamp_section_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)


class CrossModalClaim(BaseModel):
    claim_id: str
    claim: str
    evidence: list[str] = Field(default_factory=list)
    support_packets: list[str] = Field(default_factory=list)
    conflict_packets: list[str] = Field(default_factory=list)
    unresolved_packets: list[str] = Field(default_factory=list)
    status: Literal["supported", "contradicted", "partial", "unresolved"] = "unresolved"
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)


class DiscrepancyV2(BaseModel):
    claim: str
    reason: Literal["unsupported", "contradicted", "magnitude_mismatch", "missing_modality"]
    evidence: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"
    confidence: float = 0.0
    linked_packet_ids: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)


class ModalitySection(BaseModel):
    findings: list[ModalityEvidence] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)


class MethodologyDetail(BaseModel):
    statement: str
    category: str = "other"
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)


class MethodSlot(BaseModel):
    slot_key: str
    label: str
    statement: str
    status: Literal["found", "not_found", "access_limited"] = "not_found"
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)


class SectionSlot(BaseModel):
    section_key: Literal["introduction", "methods", "results", "discussion", "conclusion"]
    slot_key: str
    label: str
    statement: str
    status: Literal["found", "not_found", "access_limited"] = "not_found"
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)


class SectionExtractedBullet(BaseModel):
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    kind: str = "other"


class SectionEvidence(BaseModel):
    statement: str
    anchor: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    source_modality: ModalityName = "text"
    confidence: float = 0.0
    section_confidence: float = 0.0
    is_untrusted: bool = False
    flags: list[str] = Field(default_factory=list)
    result_evidence_type: ResultEvidenceType | None = None

    @field_validator("confidence", "section_confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)


class SectionBlock(BaseModel):
    items: list[SectionEvidence] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str | None = None


class DetailedSections(BaseModel):
    introduction: SectionBlock = Field(default_factory=SectionBlock)
    methods: SectionBlock = Field(default_factory=SectionBlock)
    results: SectionBlock = Field(default_factory=SectionBlock)
    discussion: SectionBlock = Field(default_factory=SectionBlock)
    conclusion: SectionBlock = Field(default_factory=SectionBlock)


class StructuredDossierV2(BaseModel):
    schema_version: int = 2
    sectioned_report_version: int = 3
    paper_meta: dict = Field(default_factory=dict)
    coverage: dict = Field(default_factory=dict)
    modalities: dict[str, ModalitySection] = Field(
        default_factory=lambda: {
            "text": ModalitySection(),
            "table": ModalitySection(),
            "figure": ModalitySection(),
            "supplement": ModalitySection(),
        }
    )
    cross_modal_claims: list[CrossModalClaim] = Field(default_factory=list)
    discrepancies: list[DiscrepancyV2] = Field(default_factory=list)
    executive_summary: str = ""
    extractive_evidence_version: int = 0
    extractive_evidence: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    presentation_evidence_version: int = 0
    presentation_evidence: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    executive_report_version: int = 0
    executive_report: dict[str, Any] = Field(default_factory=dict)
    methods_strengths: list[str] = Field(default_factory=list)
    methods_weaknesses: list[str] = Field(default_factory=list)
    methods_compact: list[MethodSlot] = Field(default_factory=list)
    methods_compact_version: int = 1
    sections_compact: dict[str, list[SectionSlot]] = Field(default_factory=dict)
    sections_compact_version: int = 1
    sections_extracted: dict[str, list[SectionExtractedBullet]] = Field(default_factory=dict)
    sections_extracted_version: int = 0
    methodology_details: list[MethodologyDetail] = Field(default_factory=list)
    sections: DetailedSections = Field(default_factory=DetailedSections)
    section_diagnostics: dict = Field(default_factory=dict)
    sections_fallback_used: bool = False
    sections_fallback_notes: list[str] = Field(default_factory=list)
    coverage_snapshot_line: str = ""
    reproducibility_ethics: list[str] = Field(default_factory=list)
    uncertainty_gaps: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.0

    @field_validator("overall_confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return float(value)
