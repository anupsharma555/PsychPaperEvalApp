from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from scripts.compare_evidence_to_gold import (
    _build_artifact_stage_diagnostics,
    compare_evidence_to_gold,
    evidence_metadata_from_payload,
    evidence_packets_from_payload,
)
from scripts.validate_gold_standards import load_gold_standard, validate_gold_standard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARMA_GOLD = PROJECT_ROOT / "benchmarks" / "gold_standards" / "sharma_2017_reward_deficits.json"


def test_evidence_packets_are_compatible_with_reviewed_gold_claims() -> None:
    gold = load_gold_standard(SHARMA_GOLD)
    packets = [
        {
            "finding_id": "methods-sample",
            "modality": "text",
            "section_label": "methods",
            "statement": (
                "The final analytic sample included 225 adults across major depressive disorder, "
                "bipolar disorder, schizophrenia, psychosis risk, and healthy controls."
            ),
            "evidence_refs": ["section:Methods:1"],
            "confidence": 0.92,
        },
        {
            "finding_id": "methods-bas",
            "modality": "text",
            "section_label": "methods",
            "statement": (
                "Reward responsivity was measured dimensionally with the Behavioral Activation Scale "
                "reward sensitivity subscale and resting-state functional connectivity."
            ),
            "evidence_refs": ["section:Methods:2"],
            "confidence": 0.9,
        },
        {
            "finding_id": "results-network",
            "modality": "figure",
            "section_label": "results",
            "statement": (
                "Reward deficits were linked to nucleus accumbens, default mode network, and "
                "cingulo-opercular network dysconnectivity."
            ),
            "source_excerpt": "Figure results describe network dysconnectivity associated with reward deficits.",
            "evidence_refs": ["figure:2"],
            "confidence": 0.87,
        },
    ]

    comparison = compare_evidence_to_gold(
        packets,
        gold,
        min_critical_claim_candidate_rate=1.0,
        min_section_coverage_rate=1.0,
    )

    assert comparison["compatible"] is True
    assert comparison["critical_claim_candidate_rate"] == 1.0
    assert comparison["expected_entity_observability_rate"] == 1.0
    assert comparison["expected_number_observability_rate"] == 1.0
    assert comparison["expected_detail_type_observability_rate"] == 1.0
    assert comparison["overall_benchmark_score"] == 1.0
    assert comparison["benchmark_content_score"] == 1.0
    assert comparison["claim_requirement_gaps"] == []
    assert comparison["schema_gaps"] == []


def test_overall_benchmark_score_counts_required_content_slots_without_extra_content_penalty() -> None:
    gold = {
        "case_id": "score-case",
        "critical_claims": [
            {
                "claim_id": "score-001",
                "section": "methods",
                "claim": "Sertraline 20 mg was assessed with MADRS.",
                "expected_entities": ["sertraline", "MADRS"],
                "expected_numbers": [{"label": "dose", "value": 20, "unit": "mg"}],
                "expected_detail_types": ["medication_or_therapeutic"],
            }
        ],
        "report_should_not_claim": [],
    }
    full = compare_evidence_to_gold(
        [
            {
                "finding_id": "full",
                "section_label": "methods",
                "modality": "text",
                "statement": "Sertraline 20 mg was assessed with MADRS; extra exploratory outcomes were also described.",
                "evidence_refs": ["section:Methods:1"],
            }
        ],
        gold,
    )
    partial = compare_evidence_to_gold(
        [
            {
                "finding_id": "partial",
                "section_label": "methods",
                "modality": "text",
                "statement": "Sertraline 20 mg was administered.",
                "evidence_refs": ["section:Methods:1"],
            }
        ],
        gold,
    )

    assert full["overall_benchmark_score"] == 1.0
    assert full["benchmark_content_score_basis"]["matched_slots"] == 6
    assert full["benchmark_content_score_basis"]["expected_slots"] == 6
    assert full["benchmark_content_score_basis"]["extra_content_penalized"] is False
    assert partial["overall_benchmark_score"] == 0.833
    assert partial["benchmark_content_score_basis"]["matched_slots"] == 5
    assert partial["claim_requirement_gaps"][0]["missing_entities"] == ["MADRS"]


def test_evidence_gold_compatibility_fails_when_packets_cannot_be_scored() -> None:
    gold = load_gold_standard(SHARMA_GOLD)
    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "unanchored",
                "statement": "The paper discusses reward deficits.",
                "section_label": "unknown",
                "evidence_refs": [],
                "modality": "text",
            }
        ],
        gold,
    )

    assert comparison["compatible"] is False
    assert comparison["usable_packets"] == 0
    assert "no evidence_refs" in comparison["schema_gaps"]
    assert "no known section_label values" in comparison["schema_gaps"]
    assert "schema gap: no evidence_refs" in comparison["failure_reasons"]


def test_evidence_gold_compatibility_requires_expected_entities() -> None:
    gold = load_gold_standard(SHARMA_GOLD)
    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "vague-methods",
                "statement": "The final analytic sample included 225 adults across several groups.",
                "section_label": "methods",
                "evidence_refs": ["section:Methods:1"],
                "modality": "text",
            },
            {
                "finding_id": "vague-bas",
                "statement": "Reward responsivity was measured dimensionally.",
                "section_label": "methods",
                "evidence_refs": ["section:Methods:2"],
                "modality": "text",
            },
            {
                "finding_id": "vague-results",
                "statement": "Reward deficits were linked to network dysconnectivity.",
                "section_label": "results",
                "evidence_refs": ["section:Results:1"],
                "modality": "text",
            },
        ],
        gold,
    )

    assert comparison["compatible"] is False
    assert comparison["expected_entity_observability_rate"] < 0.5
    assert comparison["expected_number_observability_rate"] == 1.0
    assert comparison["claim_requirement_gaps"][0]["claim_id"] == "sharma-001"
    assert "major depressive disorder" in comparison["claim_requirement_gaps"][0]["missing_entities"]
    assert any("expected entity observability rate" in reason for reason in comparison["failure_reasons"])


def test_evidence_gold_compatibility_requires_expected_detail_types() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "results-adverse-event",
            "section": "results",
            "importance": "P1",
            "claim": "Reward deficits were linked to nucleus accumbens, default mode network, and cingulo-opercular network dysconnectivity.",
            "source_anchor": "Abstract Results",
            "expected_entities": ["nucleus accumbens", "default mode network", "cingulo-opercular network"],
            "expected_numbers": [],
            "expected_detail_types": ["adverse_event"],
        }
    ]
    packets = [
        {
            "finding_id": "results-network",
            "modality": "figure",
            "section_label": "results",
            "statement": (
                "Reward deficits were linked to nucleus accumbens, default mode network, and "
                "cingulo-opercular network dysconnectivity."
            ),
            "evidence_refs": ["figure:2"],
        },
    ]

    missing_detail_types = compare_evidence_to_gold(
        packets,
        gold,
        min_critical_claim_candidate_rate=1.0,
        min_section_coverage_rate=1.0,
    )
    packets[0]["detail_types"] = ["adverse_event"]
    matched_detail_types = compare_evidence_to_gold(
        packets,
        gold,
        min_critical_claim_candidate_rate=1.0,
        min_section_coverage_rate=1.0,
    )

    assert missing_detail_types["compatible"] is False
    assert missing_detail_types["expected_detail_type_observability_rate"] == 0.0
    assert missing_detail_types["claim_requirement_gaps"][0]["missing_detail_types"] == [
        "adverse_event",
    ]
    assert any("expected detail-type observability rate" in reason for reason in missing_detail_types["failure_reasons"])
    assert matched_detail_types["compatible"] is True
    assert matched_detail_types["expected_detail_type_observability_rate"] == 1.0
    assert matched_detail_types["claim_matches"][0]["matched_detail_type_values"] == ["adverse_event"]


def test_evidence_gold_compatibility_reports_missing_number_gaps() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "missing-dose",
            "section": "methods",
            "importance": "P1",
            "claim": "Participants received fluoxetine 20 mg daily.",
            "source_anchor": "Methods",
            "expected_entities": ["fluoxetine"],
            "expected_numbers": [{"label": "dose_mg", "value": 20, "unit": "mg"}],
        }
    ]

    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "methods-medication",
                "modality": "text",
                "section_label": "methods",
                "statement": "Participants received fluoxetine daily.",
                "evidence_refs": ["section:Methods:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
    )

    assert comparison["compatible"] is False
    assert comparison["claim_requirement_gaps"] == [
        {
            "claim_id": "missing-dose",
            "section": "methods",
            "best_packet_id": "methods-medication",
            "candidate_missing": False,
            "missing_entities": [],
            "missing_numbers": ["dose_mg"],
            "missing_detail_types": [],
        }
    ]


def test_evidence_gold_compatibility_requires_expected_number_units() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "wrong-unit-dose",
            "section": "methods",
            "importance": "P1",
            "claim": "Participants received fluoxetine 20 mg daily.",
            "source_anchor": "Methods",
            "expected_entities": ["fluoxetine"],
            "expected_numbers": [{"label": "dose_mg", "value": 20, "unit": "mg"}],
        }
    ]

    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "methods-wrong-unit",
                "modality": "text",
                "section_label": "methods",
                "statement": "Participants received fluoxetine for 20 weeks before outcome assessment.",
                "evidence_refs": ["section:Methods:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
    )

    assert comparison["compatible"] is False
    assert comparison["claim_matches"][0]["matched_number_values"] == []
    assert comparison["claim_requirement_gaps"][0]["missing_numbers"] == ["dose_mg"]


def test_evidence_gold_compatibility_matches_expected_number_unit_aliases() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "alias-dose",
            "section": "methods",
            "importance": "P1",
            "claim": "Participants received fluoxetine 20 mg daily.",
            "source_anchor": "Methods",
            "expected_entities": ["fluoxetine"],
            "expected_numbers": [{"label": "dose_mg", "value": 20, "unit": "mg"}],
        }
    ]

    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "methods-dose-alias",
                "modality": "text",
                "section_label": "methods",
                "statement": "Participants received fluoxetine at a 20 milligrams daily dose.",
                "evidence_refs": ["section:Methods:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
    )

    assert comparison["compatible"] is True
    assert comparison["claim_matches"][0]["matched_number_values"] == ["dose_mg"]


def test_evidence_gold_compatibility_matches_scientific_unit_aliases() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "scientific-units",
            "section": "results",
            "importance": "P1",
            "claim": "The assay had an IC50 of 9.8 uM and treatment dose of 7.25 mg/kg/day.",
            "source_anchor": "Results",
            "expected_entities": ["assay", "treatment"],
            "expected_numbers": [
                {"label": "ic50_um", "value": 9.8, "unit": "uM"},
                {"label": "dose_mgkgday", "value": 7.25, "unit": "mg/kg/day"},
            ],
            "expected_detail_types": ["assay_readout", "dose_schedule"],
        }
    ]

    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "results-scientific-units",
                "modality": "text",
                "section_label": "results",
                "statement": "The assay IC50 was 9.8 micromolar; treatment was administered at 7.25 mg/kg/d.",
                "detail_types": ["assay_readout", "dose_schedule"],
                "evidence_refs": ["section:Results:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )

    assert comparison["compatible"] is True
    assert comparison["claim_matches"][0]["matched_number_values"] == ["ic50_um", "dose_mgkgday"]


def test_evidence_gold_compatibility_matches_statistic_labels_before_values() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "statistic-prefix-units",
            "section": "results",
            "importance": "P1",
            "claim": "The association had OR 1.26 and model discrimination D prime 0.991.",
            "source_anchor": "Results",
            "expected_entities": ["association", "model discrimination"],
            "expected_numbers": [
                {"label": "odds_ratio", "value": 1.26, "unit": "odds_ratio"},
                {"label": "d_prime", "value": 0.991, "unit": "D_prime"},
            ],
            "expected_detail_types": ["statistical_result"],
        }
    ]

    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "results-statistic-prefixes",
                "modality": "table",
                "section_label": "results",
                "statement": "The association was significant: OR = 1.26, and model discrimination was D' = 0.991.",
                "detail_types": ["statistical_result"],
                "evidence_refs": ["table:2"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )

    assert comparison["compatible"] is True
    assert comparison["claim_matches"][0]["matched_number_values"] == ["odds_ratio", "d_prime"]


def test_evidence_gold_compatibility_matches_expected_number_comparators() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "p-value-threshold",
            "section": "results",
            "importance": "P1",
            "claim": "The primary endpoint improved with p < 0.05.",
            "source_anchor": "Results",
            "expected_entities": ["primary endpoint"],
            "expected_numbers": [{"label": "p_threshold", "value": 0.05, "comparator": "<"}],
            "expected_detail_types": ["statistical_result"],
        }
    ]

    matched = compare_evidence_to_gold(
        [
            {
                "finding_id": "results-p-threshold",
                "modality": "table",
                "section_label": "results",
                "statement": "The primary endpoint improved significantly (p < 0.05).",
                "evidence_refs": ["table:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )
    matched_leading_decimal = compare_evidence_to_gold(
        [
            {
                "finding_id": "results-p-leading-decimal",
                "modality": "table",
                "section_label": "results",
                "statement": "The primary endpoint improved significantly (p < .05).",
                "evidence_refs": ["table:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )
    wrong_comparator = compare_evidence_to_gold(
        [
            {
                "finding_id": "results-p-equal",
                "modality": "table",
                "section_label": "results",
                "statement": "The primary endpoint had p = 0.05.",
                "evidence_refs": ["table:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )

    assert matched["compatible"] is True
    assert matched["claim_matches"][0]["matched_number_values"] == ["p_threshold"]
    assert matched_leading_decimal["compatible"] is True
    assert matched_leading_decimal["claim_matches"][0]["matched_number_values"] == ["p_threshold"]
    assert wrong_comparator["compatible"] is False
    assert wrong_comparator["claim_matches"][0]["missing_number_values"] == ["p_threshold"]


def test_evidence_gold_compatibility_matches_approximate_unit_values() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "approx-dose",
            "section": "methods",
            "importance": "P1",
            "claim": "Participants received approximately 10 mg of fluoxetine.",
            "source_anchor": "Methods",
            "expected_entities": ["fluoxetine"],
            "expected_numbers": [{"label": "approx_dose", "value": 10, "unit": "mg", "comparator": "~"}],
            "expected_detail_types": ["medication_or_therapeutic", "dose_schedule"],
        }
    ]

    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "methods-approx-dose",
                "modality": "text",
                "section_label": "methods",
                "statement": "Participants received fluoxetine at about 9.8 mg daily.",
                "evidence_refs": ["section:Methods:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )

    assert comparison["compatible"] is True
    assert comparison["claim_matches"][0]["matched_number_values"] == ["approx_dose"]


def test_evidence_gold_compatibility_infers_detail_types_from_packet_text() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "trial-medication",
            "section": "methods",
            "importance": "P1",
            "claim": "Participants received fluoxetine 20 mg daily for symptom outcomes.",
            "source_anchor": "Methods",
            "expected_entities": ["fluoxetine"],
            "expected_numbers": [{"label": "dose", "value": 20, "unit": "mg"}],
            "expected_detail_types": ["medication_or_therapeutic", "dose_schedule", "outcome_measure"],
        }
    ]
    packets = [
        {
            "finding_id": "methods-medication",
            "modality": "text",
            "section_label": "methods",
            "statement": "Participants received fluoxetine 20 mg daily and PHQ-9 symptom outcome was measured.",
            "evidence_refs": ["section:Methods:1"],
        }
    ]

    comparison = compare_evidence_to_gold(
        packets,
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )

    assert comparison["compatible"] is True
    assert comparison["expected_detail_type_observability_rate"] == 1.0
    assert comparison["claim_matches"][0]["matched_detail_type_values"] == [
        "medication_or_therapeutic",
        "dose_schedule",
        "outcome_measure",
    ]


def test_evidence_gold_compatibility_supports_split_claim_requirements_across_packets() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "trial-split-methods",
            "section": "methods",
            "importance": "P1",
            "claim": "Participants received fluoxetine 20 mg daily and PHQ-9 symptom outcomes were measured.",
            "source_anchor": "Methods",
            "expected_entities": ["fluoxetine", "PHQ-9"],
            "expected_numbers": [{"label": "dose", "value": 20, "unit": "mg"}],
            "expected_detail_types": ["medication_or_therapeutic", "dose_schedule", "outcome_measure"],
        }
    ]

    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "methods-medication",
                "modality": "text",
                "section_label": "methods",
                "statement": "Participants received fluoxetine as the active medication.",
                "evidence_refs": ["section:Methods:1"],
            },
            {
                "finding_id": "methods-dose",
                "modality": "text",
                "section_label": "methods",
                "statement": "The active arm used a 20 mg daily oral dose.",
                "evidence_refs": ["section:Methods:2"],
            },
            {
                "finding_id": "methods-outcome",
                "modality": "text",
                "section_label": "methods",
                "statement": "PHQ-9 symptom outcome scores were measured at follow-up.",
                "evidence_refs": ["section:Methods:3"],
            },
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )

    assert comparison["compatible"] is True
    match = comparison["claim_matches"][0]
    assert match["matched_entity_values"] == ["fluoxetine", "PHQ-9"]
    assert match["matched_number_values"] == ["dose"]
    assert match["matched_detail_type_values"] == [
        "medication_or_therapeutic",
        "dose_schedule",
        "outcome_measure",
    ]
    assert set(match["supporting_packet_ids"]) == {
        "methods-medication",
        "methods-dose",
        "methods-outcome",
    }


def test_gold_standard_validator_accepts_expected_detail_types() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"][0]["expected_detail_types"] = [
        "data_source_or_design",
        "rationale_or_objective",
        "interpretation_or_implication",
        "limitation_or_caution",
        "conclusion_or_takeaway",
    ]

    assert validate_gold_standard(gold) == []


def test_evidence_gold_compatibility_infers_reasoning_detail_types() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"] = [
        {
            "claim_id": "discussion-caution",
            "section": "discussion",
            "importance": "P1",
            "claim": "The authors interpreted the findings cautiously because cross-sectional design limits causal inference.",
            "source_anchor": "Discussion",
            "expected_entities": [],
            "expected_numbers": [],
            "expected_detail_types": ["interpretation_or_implication", "limitation_or_caution"],
        }
    ]

    comparison = compare_evidence_to_gold(
        [
            {
                "finding_id": "discussion-caution",
                "modality": "text",
                "section_label": "discussion",
                "statement": (
                    "The authors interpreted the findings cautiously because cross-sectional design "
                    "limits causal inference and future research is needed."
                ),
                "evidence_refs": ["section:Discussion:1"],
            }
        ],
        gold,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )

    assert comparison["compatible"] is True
    assert comparison["claim_matches"][0]["matched_detail_type_values"] == [
        "interpretation_or_implication",
        "limitation_or_caution",
    ]


def test_gold_standard_validator_rejects_unknown_expected_detail_type() -> None:
    gold = copy.deepcopy(load_gold_standard(SHARMA_GOLD))
    gold["critical_claims"][0]["expected_detail_types"] = ["magic_signal"]

    errors = validate_gold_standard(gold)

    assert any("expected_detail_types[0] must be one of" in error for error in errors)


def test_evidence_gold_compatibility_rejects_forbidden_claims() -> None:
    gold = load_gold_standard(SHARMA_GOLD)
    packets = [
        {
            "finding_id": "methods-sample",
            "modality": "text",
            "section_label": "methods",
            "statement": (
                "The final analytic sample included 225 adults across major depressive disorder, "
                "bipolar disorder, schizophrenia, psychosis risk, and healthy controls."
            ),
            "evidence_refs": ["section:Methods:1"],
        },
        {
            "finding_id": "methods-bas",
            "modality": "text",
            "section_label": "methods",
            "statement": (
                "Reward responsivity was measured dimensionally with the Behavioral Activation Scale "
                "reward sensitivity subscale and resting-state functional connectivity."
            ),
            "evidence_refs": ["section:Methods:2"],
        },
        {
            "finding_id": "results-network",
            "modality": "figure",
            "section_label": "results",
            "statement": (
                "Reward deficits were linked to nucleus accumbens, default mode network, and "
                "cingulo-opercular network dysconnectivity."
            ),
            "evidence_refs": ["figure:2"],
        },
        {
            "finding_id": "forbidden-treatment",
            "modality": "text",
            "section_label": "conclusion",
            "statement": "The study tested a treatment and longitudinal outcome for reward deficits.",
            "evidence_refs": ["section:Conclusion:1"],
        },
    ]

    comparison = compare_evidence_to_gold(
        packets,
        gold,
        min_critical_claim_candidate_rate=1.0,
        min_section_coverage_rate=1.0,
    )

    assert comparison["compatible"] is False
    assert comparison["forbidden_claim_violations"]
    assert comparison["forbidden_claim_violations"][0]["best_packet_id"] == "forbidden-treatment"
    assert comparison["failure_reasons"] == ["forbidden claim violations: 1"]


def test_evidence_packet_loader_accepts_structured_report_payload() -> None:
    payload = {
        "modalities": {
            "figure": {
                "findings": [
                    {
                        "finding_id": "fig-1",
                        "statement": "Figure 1 reports the primary effect.",
                        "evidence_refs": ["figure:1"],
                        "section_label": "results",
                    }
                ]
            }
        },
        "scientific_details": [
            {
                "statement": "Participants received fluoxetine 20 mg daily.",
                "source_excerpt": "Methods detail: fluoxetine 20 mg daily.",
                "evidence_refs": ["section:Methods:1"],
                "source_modality": "text",
                "section_label": "methods",
                "category": "medication",
                "detail_types": ["medication_or_therapeutic"],
            }
        ],
        "sections_extracted": {
            "discussion": [
                {
                    "statement": "Discussion interpreted the finding cautiously.",
                    "evidence_refs": ["section:Discussion:1"],
                }
            ]
        },
    }

    packets = evidence_packets_from_payload(payload)

    assert [packet["section_label"] for packet in packets] == ["results", "methods", "discussion"]
    assert [packet["modality"] for packet in packets] == ["figure", "text", "text"]
    assert packets[1]["detail_types"] == ["medication_or_therapeutic"]


def test_evidence_metadata_loader_preserves_synthesis_evidence_diagnostics() -> None:
    payload = {
        "summary_json": {
            "modalities": {
                "text": {
                    "findings": [
                        {
                            "finding_id": "methods-med",
                            "statement": "Participants received fluoxetine 20 mg daily.",
                            "evidence_refs": ["section:Methods:1"],
                            "section_label": "methods",
                        }
                    ]
                }
            },
            "section_diagnostics": {
                "evidence_packet_coverage": {
                    "packet_total": 1,
                    "usable_packets": 1,
                    "usable_packet_rate": 1.0,
                    "sections_present": ["methods"],
                    "missing_core_sections": ["results", "discussion"],
                    "by_section": {"methods": 1, "unknown": 4},
                    "by_modality": {"text": 1, "unknown": 3},
                    "by_detail_type": {"medication_or_therapeutic": 1, "dose_schedule": 1},
                    "cross_modal_packet_count": 0,
                    "typed_packet_count": 1,
                    "quality_flags": ["no_cross_modal_packets"],
                },
                "synthesis_evidence_warnings": [
                    "Critical synthesis evidence not found for this paper type: Safety or Adverse Events."
                ],
                "synthesis_evidence_plan": {
                    "missing_focus_slot_count": 3,
                    "quality_flags": ["critical_focus_slots_missing", "safety_or_adverse_events_missing"],
                    "critical_missing_focus_slots": [
                        {
                            "slot_key": "safety_or_adverse_events",
                            "label": "Safety or Adverse Events",
                            "reason": "No selected scientific detail matched this focus slot.",
                        }
                    ],
                },
            },
        }
    }

    metadata = evidence_metadata_from_payload(payload)
    comparison = compare_evidence_to_gold(
        evidence_packets_from_payload(payload),
        {
            "case_id": "trial",
            "critical_claims": [
                {
                    "claim_id": "med",
                    "section": "methods",
                    "claim": "Participants received fluoxetine 20 mg daily.",
                    "expected_entities": ["fluoxetine"],
                    "expected_numbers": [{"label": "dose", "value": 20, "unit": "mg"}],
                    "expected_detail_types": ["medication_or_therapeutic", "dose_schedule"],
                }
            ],
            "report_should_not_claim": [],
        },
        evidence_metadata=metadata,
        min_usable_packet_rate=1.0,
        min_section_coverage_rate=1.0,
        min_critical_claim_candidate_rate=1.0,
        min_expected_entity_observability_rate=1.0,
        min_expected_number_observability_rate=1.0,
        min_expected_detail_type_observability_rate=1.0,
    )

    diagnostics = comparison["synthesis_evidence_diagnostics"]
    assert diagnostics["has_synthesis_evidence_plan"] is True
    assert diagnostics["missing_focus_slot_count"] == 3
    assert diagnostics["critical_missing_focus_slot_count"] == 1
    assert diagnostics["critical_missing_focus_slots"][0]["slot_key"] == "safety_or_adverse_events"
    assert "safety_or_adverse_events_missing" in diagnostics["synthesis_quality_flags"]
    packet_coverage = diagnostics["evidence_packet_coverage"]
    assert packet_coverage["available"] is True
    assert packet_coverage["usable_packets"] == 1
    assert packet_coverage["by_section"] == {"methods": 1}
    assert packet_coverage["by_modality"] == {"text": 1}
    assert packet_coverage["missing_core_sections"] == ["results", "discussion"]
    assert "no_cross_modal_packets" in packet_coverage["quality_flags"]
    assert comparison["compatible"] is True


def test_evidence_packet_loader_accepts_compare_run_json_wrapper() -> None:
    payload = {
        "run_mode": "pipeline",
        "summary_json": {
            "modalities": {
                "table": {
                    "findings": [
                        {
                            "finding_id": "table-1",
                            "statement": "Table 1 reports 225 participants.",
                            "evidence_refs": ["table:1"],
                            "section_label": "methods",
                        }
                    ]
                }
            }
        },
    }

    packets = evidence_packets_from_payload(payload)

    assert len(packets) == 1
    assert packets[0]["modality"] == "table"
    assert packets[0]["section_label"] == "methods"


def test_evidence_packet_loader_accepts_report_section_bullets() -> None:
    payload = {
        "v2_report": {
            "sections": [
                {
                    "section": "methods",
                    "bullets": [
                        {
                            "text": "The cohort study analyzed 225 participants.",
                            "anchors": ["section:Methods:1"],
                            "source": "sections_extracted",
                        }
                    ],
                },
                {
                    "section": "results",
                    "bullets": [
                        {
                            "text": "Figure results showed reward-network dysconnectivity.",
                            "anchors": ["figure:2"],
                        }
                    ],
                },
            ]
        }
    }

    packets = evidence_packets_from_payload(payload)

    assert len(packets) == 2
    assert packets[0]["finding_id"] == "v2_report-1-1"
    assert packets[0]["section_label"] == "methods"
    assert packets[0]["modality"] == "text"
    normalized_report = compare_evidence_to_gold(
        packets,
        {
            "case_id": "report",
            "critical_claims": [
                {
                    "claim_id": "sample",
                    "section": "methods",
                    "claim": "The cohort study analyzed 225 participants.",
                    "expected_entities": ["participants"],
                    "expected_numbers": [{"label": "sample", "value": 225}],
                    "expected_detail_types": ["data_source_or_design"],
                }
            ],
            "report_should_not_claim": [],
        },
        min_expected_detail_type_observability_rate=1.0,
    )
    assert normalized_report["claim_matches"][0]["matched_numbers"] == 1
    assert normalized_report["claim_matches"][0]["matched_detail_type_values"] == ["data_source_or_design"]
    assert packets[1]["section_label"] == "results"
    assert packets[1]["modality"] == "figure"


def test_evidence_gold_cli_accepts_threshold_overrides(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            [
                {
                    "finding_id": "methods-sample",
                    "modality": "text",
                    "section_label": "methods",
                    "statement": (
                        "The final analytic sample included 225 adults across major depressive disorder, "
                        "bipolar disorder, schizophrenia, psychosis risk, and healthy controls."
                    ),
                    "evidence_refs": ["section:Methods:1"],
                },
                {
                    "finding_id": "methods-bas",
                    "modality": "text",
                    "section_label": "methods",
                    "statement": (
                        "Reward responsivity was measured dimensionally with the Behavioral Activation Scale "
                        "reward sensitivity subscale and resting-state functional connectivity."
                    ),
                    "evidence_refs": ["section:Methods:2"],
                },
                {
                    "finding_id": "results-network",
                    "modality": "figure",
                    "section_label": "results",
                    "statement": (
                        "Reward deficits were linked to nucleus accumbens, default mode network, and "
                        "cingulo-opercular network dysconnectivity."
                    ),
                    "evidence_refs": ["figure:2"],
                },
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_evidence_to_gold.py",
            "--evidence-json",
            str(evidence_path),
            "--gold-standard",
            str(SHARMA_GOLD),
            "--min-critical-claim-candidate-rate",
            "1.0",
            "--fail-on-incompatible",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    comparison = json.loads(completed.stdout)

    assert comparison["compatible"] is True
    assert comparison["thresholds"]["min_critical_claim_candidate_rate"] == 1.0


def test_evidence_gold_cli_summary_prints_overall_score_and_gaps(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "summary_json": {
                    "modalities": {
                        "text": {
                            "findings": [
                                {
                                    "finding_id": "partial",
                                    "section_label": "methods",
                                    "modality": "text",
                                    "statement": "Sertraline 20 mg was administered.",
                                    "evidence_refs": ["section:Methods:1"],
                                }
                            ]
                        }
                    },
                    "synthesis_evidence_plan": {
                        "focus_slots": [
                            {
                                "slot_key": "methods",
                                "statement": "Sertraline 20 mg was administered.",
                            }
                        ]
                    },
                    "executive_summary": "Sertraline 20 mg was administered.",
                }
            }
        ),
        encoding="utf-8",
    )
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "score-case",
                "source_pdf": "score.pdf",
                "source_sha256": "0" * 64,
                "gold_standard_type": "final_report_expectations",
                "authoring": {
                    "method": "codex_assisted_source_review",
                    "review_status": "codex_drafted_needs_review",
                    "created_at": "2026-06-21",
                    "source_material": ["fixture"],
                },
                "paper_identity": {
                    "title": "Score fixture",
                    "domain": "test",
                    "study_type": "test",
                },
                "final_report_expectations": {
                    "research_question": "Does the score summary expose missing content?",
                    "study_design": "test fixture",
                    "population_or_materials": "test evidence packet",
                    "methods": ["Sertraline dose extraction"],
                    "primary_findings": ["Sertraline 20 mg was assessed with MADRS."],
                    "secondary_findings": ["None"],
                    "sensitivity_analysis": ["None"],
                    "statistical_tests_used": ["None"],
                    "interpretation": ["Partial evidence should miss MADRS."],
                    "limitations": ["Fixture only"],
                    "tables_figures_supplements": ["None"],
                    "uniqueness": ["Small comparator fixture"],
                    "supplement_availability": "No supplement.",
                },
                "scoring_focus": ["score summary"],
                "critical_claims": [
                    {
                        "claim_id": "score-001",
                        "section": "methods",
                        "importance": "P1",
                        "claim": "Sertraline 20 mg was assessed with MADRS.",
                        "source_anchor": "fixture",
                        "expected_entities": ["sertraline", "MADRS"],
                        "expected_numbers": [{"label": "dose", "value": 20, "unit": "mg"}],
                        "expected_detail_types": ["medication_or_therapeutic"],
                    }
                ],
                "report_should_not_claim": [],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_evidence_to_gold.py",
            "--evidence-json",
            str(evidence_path),
            "--gold-standard",
            str(gold_path),
            "--summary",
            "--stage-diagnostics",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "case_id: score-case" in completed.stdout
    assert "overall_benchmark_score: 0.833" in completed.stdout
    assert "matched_slots: 5" in completed.stdout
    assert "- expected_entities: 1/2" in completed.stdout
    assert "benchmark_gaps:" in completed.stdout
    assert "- score-001 (methods): missing entities: MADRS" in completed.stdout
    assert "stage_diagnostics:" in completed.stdout
    assert "- evidence_packets: 0/1 expected items present" in completed.stdout
    assert "- failure_points:" in completed.stdout
    assert "  - absent_from_saved_artifact: 1" in completed.stdout


def test_stage_diagnostics_trace_ordered_phrase_variants_without_changing_score() -> None:
    evidence_payload = {
        "summary_json": {
            "modalities": {
                "text": {
                    "findings": [
                        {
                            "finding_id": "triage-payment",
                            "section_label": "methods",
                            "modality": "text",
                            "statement": (
                                "Hospital admissions could be made dependent on willingness "
                                "(and ability) to pay."
                            ),
                            "evidence_refs": ["section:Methods:1"],
                        }
                    ]
                }
            }
        }
    }
    gold = {
        "case_id": "phrase-case",
        "critical_claims": [
            {
                "claim_id": "phrase-001",
                "section": "methods",
                "claim": "The model discusses willingness-to-pay triage.",
                "expected_entities": ["willingness to pay"],
                "expected_numbers": [],
            }
        ],
        "report_should_not_claim": [],
    }

    packets = evidence_packets_from_payload(evidence_payload)
    comparison = compare_evidence_to_gold(packets, gold)
    assert comparison["expected_entity_observability_rate"] == 0.0

    diagnostics = _build_artifact_stage_diagnostics(evidence_payload, gold, comparison)
    item = diagnostics["missing_items"][0]
    assert item["term"] == "willingness to pay"
    assert item["failure_point"] == "dropped_before_synthesis_selection"
    assert item["present_in"] == ["evidence_packets"]
    assert item["source_visibility"]["classification"] == "exact_present"
    assert "willingness (and ability) to pay" in item["stage_matches"]["evidence_packets"][0]["snippet"]


def test_stage_diagnostics_flags_weak_source_visibility_candidates() -> None:
    evidence_payload = {
        "summary_json": {
            "modalities": {
                "figure": {
                    "findings": [
                        {
                            "finding_id": "figure-signal",
                            "section_label": "results",
                            "modality": "figure",
                            "statement": "Figure evidence mentions C/EBP/AEP signaling in the model.",
                            "source_excerpt": "Immunoblot showing p-C/EBP, C/EBP, AEP, APP, and Tau.",
                            "evidence_refs": ["figure:7"],
                        }
                    ]
                }
            }
        }
    }
    gold = {
        "case_id": "visibility-case",
        "critical_claims": [
            {
                "claim_id": "visibility-001",
                "section": "results",
                "claim": "The figure reports C/EBP beta signaling.",
                "expected_entities": ["C/EBP beta"],
                "expected_numbers": [],
            }
        ],
        "report_should_not_claim": [],
    }

    packets = evidence_packets_from_payload(evidence_payload)
    comparison = compare_evidence_to_gold(packets, gold)
    assert comparison["expected_entity_observability_rate"] == 0.0

    diagnostics = _build_artifact_stage_diagnostics(evidence_payload, gold, comparison)
    item = diagnostics["missing_items"][0]
    assert item["failure_point"] == "absent_from_saved_artifact"
    assert item["source_visibility"]["classification"] == "weak_term_candidate"
    assert item["source_visibility"]["term_score"] > 0.0
    assert diagnostics["source_visibility_counts"]["weak_term_candidate"] == 1
