from __future__ import annotations

import json

from app.services.analysis import text_analysis


def _chunk(anchor: str, content: str, section: str) -> dict:
    return {
        "anchor": anchor,
        "content": content,
        "meta": json.dumps({"section_norm": section, "section_confidence": 0.9, "section_source": "heading"}),
    }


def test_local_text_preselection_reduces_llm_chunks_while_preserving_scientific_signal(monkeypatch) -> None:
    monkeypatch.setattr(text_analysis.settings, "llm_provider", "local")
    monkeypatch.setattr(text_analysis.settings, "analysis_local_text_preselection_enabled", True)
    monkeypatch.setattr(text_analysis.settings, "analysis_local_text_preselection_max_chunks", 10)
    monkeypatch.setattr(text_analysis.settings, "analysis_local_text_llm_batch_max_chars", 12000)

    prompts: list[str] = []

    def _fake_chat_text_fast(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        prompts.append(prompt)
        return "{}"

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat_text_fast)

    chunks: list[dict] = []
    for idx in range(12):
        chunks.append(_chunk(f"I{idx}", f"Background context paragraph {idx} about the disorder.", "introduction"))
    for idx in range(12):
        content = f"Methods paragraph {idx} describes recruitment and routine analysis details."
        if idx == 3:
            content = "Participants received sertraline 50 mg daily for 8 weeks versus placebo."
        chunks.append(_chunk(f"M{idx}", content, "methods"))
    for idx in range(12):
        content = f"Results paragraph {idx} reports a measured clinical outcome."
        if idx == 4:
            content = "MADRS response was higher with sertraline 50 mg than placebo at week 8, p=0.03."
        chunks.append(_chunk(f"R{idx}", content, "results"))
    for idx in range(8):
        chunks.append(_chunk(f"D{idx}", f"Discussion paragraph {idx} interprets the findings.", "discussion"))
    for idx in range(4):
        chunks.append(_chunk(f"C{idx}", f"Conclusion paragraph {idx} summarizes future research.", "conclusion"))

    report = text_analysis.analyze_text(chunks)
    diagnostics = report["diagnostics"]
    combined_prompt = "\n".join(prompts)

    assert diagnostics["text_preselection"]["enabled"] is True
    assert diagnostics["text_preselection"]["original_chunks"] == len(chunks)
    assert diagnostics["text_preselection"]["selected_chunks"] <= 10
    assert diagnostics["text_preselection"]["skipped_chunks"] >= len(chunks) - 10
    assert diagnostics["llm_input_chunks"] == diagnostics["text_preselection"]["selected_chunks"]
    assert "sertraline 50 mg daily" in combined_prompt
    assert "p=0.03" in combined_prompt


def test_text_preselection_is_not_applied_to_openai_provider(monkeypatch) -> None:
    monkeypatch.setattr(text_analysis.settings, "llm_provider", "openai")
    monkeypatch.setattr(text_analysis.settings, "analysis_local_text_preselection_enabled", True)

    chunks = [_chunk(f"M{idx}", f"Participants completed procedure {idx}.", "methods") for idx in range(14)]
    selected, diagnostics = text_analysis._preselect_text_llm_chunks(chunks)  # noqa: SLF001

    assert selected == chunks
    assert diagnostics["enabled"] is False
    assert diagnostics["selected_chunks"] == len(chunks)


def test_local_text_batch_budget_packs_medium_scientific_chunks(monkeypatch) -> None:
    monkeypatch.setattr(text_analysis.settings, "llm_provider", "local")
    monkeypatch.setattr(text_analysis.settings, "analysis_local_text_preselection_enabled", False)
    monkeypatch.setattr(text_analysis.settings, "analysis_local_text_llm_batch_max_chars", 9000)

    prompt_blocks: list[int] = []

    def _fake_chat_text_fast(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        prompt_blocks.append(prompt.count("\n[") + (1 if prompt.startswith("[") else 0))
        return "{}"

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat_text_fast)

    repeated_detail = (
        "Participants received sertraline 50 mg daily and completed MADRS, adverse-event, "
        "and response assessments at week 8. "
    )
    chunks = [
        _chunk(f"M{idx}", repeated_detail * 8, "methods" if idx < 2 else "results")
        for idx in range(4)
    ]

    report = text_analysis.analyze_text(chunks)

    assert report["diagnostics"]["llm_batches"] == 1
    assert report["diagnostics"]["llm_prompt_blocks"] == [4]
    assert prompt_blocks == [4]


def test_result_style_headings_override_parser_methods_label() -> None:
    chunk = _chunk(
        "section:Gut dysbiosis occurs in 5xFAD mice in an age-dependent manner:4",
        (
            "Previous studies have indicated an association between alterations in the intestinal "
            "microbial community and Alzheimer's disease. We found that gut dysbiosis occurs in "
            "5xFAD mice with age and is associated with increased C/EBP beta/AEP pathway activity."
        ),
        "methods",
    )

    section = text_analysis._infer_chunk_section(chunk, idx=5, total_chunks=38)  # noqa: SLF001

    assert section == "results"
