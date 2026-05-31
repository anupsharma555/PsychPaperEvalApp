from __future__ import annotations

import json

from app.services.analysis.section_ledger import apply_section_boundary_ledger_to_dicts


def _chunk(title: str, text: str, section: str = "unknown") -> dict:
    return {
        "anchor": f"section:{title}:1",
        "modality": "text",
        "content": text,
        "meta": json.dumps(
            {
                "section_raw_title": title,
                "section_norm": section,
                "section_source": "grobid",
            }
        ),
    }


def _section(row: dict) -> str:
    meta = row["meta"] if isinstance(row["meta"], dict) else json.loads(row["meta"])
    return meta["section_norm"]


def test_section_boundary_ledger_relabels_sharma_like_body_results_prefix() -> None:
    rows = [
        _chunk("body", "Objective: Anhedonia is central to multiple psychiatric disorders."),
        _chunk("body", "Method: Participants completed reward sensitivity measures."),
        _chunk("Results:", "Both animal and human studies implicate mesolimbic reward systems.", "results"),
        _chunk("Participants", "For this study, 244 participants were assessed at two visits.", "methods"),
        _chunk("DISCUSSION", "We used a fully data-driven survey of the connectome.", "discussion"),
        _chunk("CONCLUSIONS", "Our results corroborate previous research.", "conclusion"),
    ]

    relabeled = apply_section_boundary_ledger_to_dicts(rows)

    assert [_section(row) for row in relabeled] == [
        "introduction",
        "introduction",
        "introduction",
        "methods",
        "discussion",
        "conclusion",
    ]
    meta = relabeled[2]["meta"]
    assert meta["section_original_norm"] == "results"
    assert meta["section_source"].startswith("section_boundary_ledger:")


def test_section_boundary_ledger_allows_late_explicit_methods() -> None:
    rows = [
        _chunk("abstract", "Overexpression of two genes was studied in mammalian cells.", "introduction"),
        _chunk("Results", "The approach generated stable double-transfected cell lines.", "results"),
        _chunk("Discussion", "The method can be modified for other genes.", "discussion"),
        _chunk("Methods", "Cells were transfected with expression vectors and selected.", "methods"),
        _chunk("Generation of second vector", "PCR and Sanger sequencing were used to validate constructs.", "methods"),
    ]

    relabeled = apply_section_boundary_ledger_to_dicts(rows)

    assert [_section(row) for row in relabeled] == [
        "introduction",
        "introduction",
        "discussion",
        "methods",
        "methods",
    ]


def test_section_boundary_ledger_keeps_methods_subsections_after_study_design() -> None:
    rows = [
        _chunk("Abstract", "This study evaluated pediatric pneumococcal pneumonia.", "introduction"),
        _chunk(
            "Study design and sample recruitment",
            "Children under 18 years with community-acquired pneumonia were recruited prospectively.",
            "methods",
        ),
        _chunk(
            "Ethical considerations",
            "The protocol was approved by institutional review boards and informed consent was obtained.",
            "discussion",
        ),
        _chunk(
            "Specimen and pathogen identification",
            "Multiplex PCR of pleural effusion was performed to identify bacterial pathogens.",
            "discussion",
        ),
        _chunk(
            "Statistical analysis",
            "Student's t test and Fisher's exact test were used for group comparisons.",
            "methods",
        ),
        _chunk("Results", "A total of 983 patients fulfilled pneumococcal pneumonia criteria.", "results"),
    ]

    relabeled = apply_section_boundary_ledger_to_dicts(rows)

    assert [_section(row) for row in relabeled] == [
        "introduction",
        "methods",
        "methods",
        "methods",
        "methods",
        "results",
    ]
