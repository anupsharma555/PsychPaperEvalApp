from __future__ import annotations

import pytest

from app.services.analysis import llm, openai_usage, reconcile, runner, synthesis, text_analysis


def test_text_analysis_uses_fast_text_model(monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return '{"evidence_packets": [], "findings": [], "claims": []}'

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat)
    result = text_analysis.analyze_text(
        [
            {
                "anchor": "section:methods:1",
                "content": "Randomized trial with clear inclusion criteria.",
            }
        ]
    )
    assert calls
    assert "strict section fidelity" in calls[0]["prompt"].lower()
    assert result["evidence_packets"] == []
    assert result["analysis_notes"] == []


def test_text_analysis_backfills_sparse_discussion_and_conclusion(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return (
            '{"evidence_packets":['
            '{"finding_id":"text-1","anchor":"section:Methods:1","statement":"Cells were transfected with plasmids.","evidence_refs":["section:Methods:1"],"confidence":0.9,"category":"methods"},'
            '{"finding_id":"text-2","anchor":"section:Results:2","statement":"OCT1 expression increased proguanil uptake.","evidence_refs":["section:Results:2"],"confidence":0.9,"category":"results"}'
            '],"findings":[],"claims":[]}'
        )

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat)
    result = text_analysis.analyze_text(
        [
            {
                "anchor": "section:Methods:1",
                "content": "Cells were transfected with plasmids and selected with antibiotics.",
            },
            {
                "anchor": "section:Results:2",
                "content": "OCT1 expression increased proguanil uptake in engineered cells.",
            },
            {
                "anchor": "section:Discussion:3",
                "content": "We established a robust transfection system for simultaneous overexpression of two different genes.",
            },
            {
                "anchor": "section:Discussion:4",
                "content": "Limitations of the method, such as treatment with two antibiotics simultaneously, could apply to very sensitive cell lines.",
            },
            {
                "anchor": "section:Discussion:5",
                "content": "Overall we believe this technique will be useful when studying gene interactions.",
            },
        ]
    )

    packets = result["evidence_packets"]
    section_labels = [packet.get("section_label") for packet in packets]
    assert section_labels.count("discussion") >= 2
    assert section_labels.count("conclusion") >= 1
    assert any("section_backfill" in packet.get("quality_flags", []) for packet in packets)


def test_report_guidance_is_loaded_into_llm_prompts() -> None:
    assert "Define or briefly explain study-specific terms" in synthesis.SECTION_EXTRACTION_SYSTEM
    assert "Flip-It" in synthesis.EXECUTIVE_REPORT_SYNTHESIS_SYSTEM
    assert "Adjust section expectations to the inferred paper type" in synthesis.SECTION_EXTRACTION_SYSTEM


def test_paper_type_inference_detects_review_and_lab_paper() -> None:
    assert synthesis._infer_paper_type(  # noqa: SLF001 - prompt payload helper coverage
        {
            "text_packets": [
                {
                    "statement": "This systematic review searched databases and applied inclusion criteria.",
                    "category": "methods",
                }
            ]
        }
    ) == "review"
    assert synthesis._infer_paper_type(  # noqa: SLF001 - prompt payload helper coverage
        {
            "text_packets": [
                {
                    "statement": "Cell clones were screened by multiplex PCR after plasmid transfection.",
                    "category": "methods",
                }
            ]
        }
    ) == "laboratory/preclinical study"
    assert synthesis._infer_paper_type(  # noqa: SLF001 - prompt payload helper coverage
        {
            "text_chunk_records": [
                {
                    "content": (
                        "For qualitative analysis and comparison of fluorescent protein expression, "
                        "HEK293-Flp-In cells were transfected with plasmids and analyzed by microscopy."
                    )
                }
            ]
        }
    ) == "laboratory/preclinical study"


def test_openai_provider_routes_text_without_temperature(monkeypatch) -> None:
    captured: dict = {}

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "id": "resp_test",
                "output_text": "{}",
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            }

    def _fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm.settings, "psychpaper_openai_api_key", "OPENAI_API_KEY_FOR_TESTS")
    monkeypatch.setattr(llm.settings, "openai_api_key", "OPENAI_API_KEY_FOR_TESTS")
    monkeypatch.setattr(llm.settings, "openai_api_mode", "responses")
    monkeypatch.setattr(llm.settings, "openai_usage_guardrails_enabled", False)
    monkeypatch.setattr(llm.settings, "openai_text_model", "gpt-5-mini")
    monkeypatch.setattr(llm.settings, "openai_send_temperature", False)
    monkeypatch.setattr(llm.httpx, "post", _fake_post)

    result = llm.chat_text_fast("Analyze this.", system="Return JSON only.")

    assert result == "{}"
    assert captured["url"].endswith("/responses")
    assert captured["json"]["model"] == "gpt-5-mini"
    assert captured["json"]["text"] == {"format": {"type": "json_object"}}
    assert captured["json"]["instructions"] == "Return JSON only."
    content = captured["json"]["input"][-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "input_text", "text": "Analyze this."}
    assert {"type": "input_text", "text": "Return JSON."} in content
    assert "temperature" not in captured["json"]
    assert captured["headers"]["Authorization"] == "Bearer OPENAI_API_KEY_FOR_TESTS"


def test_openai_provider_routes_vision_images(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
        b"\x00\x05\xfe\x02\xfeA\xdd\x9a\x1b\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    captured: dict = {}

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "id": "resp_test",
                "output_text": "ok",
                "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
            }

    def _fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _Response()

    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm.settings, "psychpaper_openai_api_key", "OPENAI_API_KEY_FOR_TESTS")
    monkeypatch.setattr(llm.settings, "openai_api_key", "OPENAI_API_KEY_FOR_TESTS")
    monkeypatch.setattr(llm.settings, "openai_api_mode", "responses")
    monkeypatch.setattr(llm.settings, "openai_usage_guardrails_enabled", False)
    monkeypatch.setattr(llm.settings, "openai_vision_model", "gpt-5-mini")
    monkeypatch.setattr(llm.httpx, "post", _fake_post)

    assert llm.chat_with_images("Describe.", [str(image_path)]) == "ok"
    content = captured["json"]["input"][-1]["content"]
    assert captured["json"]["model"] == "gpt-5-mini"
    assert content[0] == {"type": "input_text", "text": "Describe."}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/")


def test_openai_usage_budget_blocks_expensive_reservation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(openai_usage.settings, "llm_provider", "openai")
    monkeypatch.setattr(openai_usage.settings, "openai_usage_guardrails_enabled", True)
    monkeypatch.setattr(openai_usage.settings, "openai_usage_log_path", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(openai_usage.settings, "openai_max_cost_per_run_usd", 0.001)
    monkeypatch.setattr(openai_usage.settings, "openai_max_cost_per_day_usd", 1.0)
    monkeypatch.setattr(openai_usage.settings, "openai_max_calls_per_run", 10)
    monkeypatch.setattr(openai_usage.settings, "openai_max_output_tokens_per_run", 100000)

    openai_usage.set_usage_context(job_id=999, document_id=1000, stage="text")

    with pytest.raises(openai_usage.OpenAIBudgetExceeded):
        openai_usage.reserve_openai_call(
            model="gpt-5-mini",
            modality="text",
            max_output_tokens=1000,
            estimated_input_tokens=1000,
        )


def test_openai_usage_summary_groups_actuals(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(openai_usage.settings, "llm_provider", "openai")
    monkeypatch.setattr(openai_usage.settings, "openai_usage_guardrails_enabled", True)
    monkeypatch.setattr(openai_usage.settings, "openai_usage_log_path", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(openai_usage.settings, "openai_max_cost_per_run_usd", 0.10)
    monkeypatch.setattr(openai_usage.settings, "openai_max_cost_per_day_usd", 1.0)

    openai_usage.set_usage_context(job_id=123, document_id=456, stage="figure")
    reservation_id = openai_usage.reserve_openai_call(
        model="gpt-5-mini",
        modality="vision",
        max_output_tokens=10,
        estimated_input_tokens=20,
    )
    openai_usage.record_openai_result(
        reservation_id=reservation_id,
        model="gpt-5-mini",
        modality="vision",
        usage={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
    )

    summary = openai_usage.summarize_openai_usage(job_id=123, document_id=456)

    assert summary["calls"] == 1
    assert summary["input_tokens"] == 20
    assert summary["output_tokens"] == 10
    assert summary["total_tokens"] == 30
    assert summary["estimated_cost_usd"] > 0
    assert summary["by_stage"]["figure"]["calls"] == 1


def test_openai_usage_marks_abandoned_reservations_failed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(openai_usage.settings, "llm_provider", "openai")
    monkeypatch.setattr(openai_usage.settings, "openai_usage_guardrails_enabled", True)
    monkeypatch.setattr(openai_usage.settings, "openai_usage_log_path", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(openai_usage.settings, "openai_max_cost_per_run_usd", 0.013)
    monkeypatch.setattr(openai_usage.settings, "openai_max_cost_per_day_usd", 1.0)
    monkeypatch.setattr(openai_usage.settings, "openai_max_calls_per_run", 10)
    monkeypatch.setattr(openai_usage.settings, "openai_max_output_tokens_per_run", 100000)

    openai_usage.set_usage_context(job_id=124, document_id=457, stage="text")
    openai_usage.reserve_openai_call(
        model="gpt-5-mini",
        modality="text",
        max_output_tokens=6000,
        estimated_input_tokens=1000,
    )
    marked = openai_usage.mark_unmatched_openai_reservations_failed(
        job_id=124,
        document_id=457,
        stage="text",
        error="test timeout",
    )

    assert marked == 1
    openai_usage.reserve_openai_call(
        model="gpt-5-mini",
        modality="text",
        max_output_tokens=6000,
        estimated_input_tokens=1000,
    )


def test_openai_mode_rejects_text_fallback_report(monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "llm_provider", "openai")

    with pytest.raises(RuntimeError, match="refusing to create a fallback report"):
        runner._enforce_openai_stage_success(
            "text",
            chunks=[{"anchor": "section:body:1", "content": "Paper text."}],
            usage={"text_calls": 1, "text_errors": 1},
            fallback_reason="subprocess_fallback:OpenAI request failed with HTTP 401",
        )


def test_openai_mode_rejects_vision_fallback_report(monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "llm_provider", "openai")

    with pytest.raises(RuntimeError, match="OpenAI figure analysis failed"):
        runner._enforce_openai_stage_success(
            "figure",
            chunks=[{"anchor": "figure:1", "meta": "{}"}],
            usage={"vision_calls": 1, "vision_errors": 1},
            report={"diagnostics": {"vision_failures": 1, "ocr_fallback_calls": 0}},
        )


def test_local_mode_allows_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(runner.settings, "llm_provider", "local")

    runner._enforce_openai_stage_success(
        "text",
        chunks=[{"anchor": "section:body:1", "content": "Paper text."}],
        usage={"text_calls": 1, "text_errors": 1},
        fallback_reason="subprocess_fallback:local model crashed",
    )


def test_text_analysis_drops_invalid_anchor_absence_spam(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return (
            '{"evidence_packets":[{"finding_id":"text-1","anchor":"section:methods:3",'
            '"statement":"Statistical models are not described.","evidence_refs":["section:methods:3"],'
            '"confidence":0.0,"category":"stats"}],'
            '"findings":[],"claims":[]}'
        )

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat)
    result = text_analysis.analyze_text(
        [
            {
                "anchor": "html:para:0",
                "content": "Randomized participants were assigned to placebo or active treatment.",
            }
        ]
    )
    assert result["evidence_packets"] == []


def test_text_analysis_adds_access_limited_note(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return '{"evidence_packets": [], "findings": [], "claims": []}'

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat)
    result = text_analysis.analyze_text(
        [
            {"anchor": "html:para:1", "content": "Already a subscriber? Access your subscription credentials."},
            {"anchor": "html:para:2", "content": "Purchase this article to access the full text."},
        ]
    )
    assert len(result["analysis_notes"]) == 1
    assert "access-limited" in result["analysis_notes"][0]


def test_text_analysis_heuristic_fallback_recovers_section_packets(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return '{"evidence_packets": [], "findings": [], "claims": []}'

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat)
    result = text_analysis.analyze_text(
        [
            {
                "anchor": "section:body:0",
                "content": "Objective: Reward dysfunction motivates a transdiagnostic framework.",
                "meta": '{"section_norm":"introduction"}',
            },
            {
                "anchor": "section:Participants:1",
                "content": "Participants (n=225) underwent MRI and MDMR analyses with covariate adjustment.",
                "meta": '{"section_norm":"methods"}',
            },
            {
                "anchor": "section:Results:2",
                "content": "Results showed higher default-mode connectivity associated with lower reward responsivity (p<0.05).",
                "meta": '{"section_norm":"results"}',
            },
            {
                "anchor": "section:Discussion:3",
                "content": "These findings suggest a shared mechanism and highlight limitations in generalizability.",
                "meta": '{"section_norm":"discussion"}',
            },
        ]
    )
    packets = result["evidence_packets"]
    assert packets
    labels = {packet.get("section_label") for packet in packets}
    assert "methods" in labels
    assert "results" in labels


def test_text_analysis_uses_chunk_fallback_when_llm_returns_empty(monkeypatch) -> None:
    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        return '{"evidence_packets": [], "findings": [], "claims": []}'

    monkeypatch.setattr(text_analysis, "chat_text_fast", _fake_chat)
    result = text_analysis.analyze_text(
        [
            {
                "anchor": "section:body:0",
                "content": "Background: Reward dysfunction spans mood and psychotic disorders and motivates transdiagnostic analysis.",
            },
            {
                "anchor": "section:Participants:1",
                "content": "Participants were recruited across diagnostic groups and assessed with dimensional reward measures.",
            },
            {
                "anchor": "section:Results:2",
                "content": "Results showed increased default mode connectivity associated with lower reward responsivity.",
            },
            {
                "anchor": "section:DISCUSSION:3",
                "content": "These findings suggest shared network-level mechanisms and highlight limitations of cross-sectional inference.",
            },
            {
                "anchor": "section:CONCLUSIONS:4",
                "content": "In conclusion, corticostriatal dysconnectivity may underlie transdiagnostic reward impairment.",
            },
        ]
    )
    assert len(result["evidence_packets"]) >= 4
    section_labels = {packet.get("section_label") for packet in result["evidence_packets"]}
    assert "methods" in section_labels
    assert "results" in section_labels


def test_reconcile_unresolved_uses_deep_model(monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return '{"discrepancies": []}'

    monkeypatch.setattr(reconcile, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(reconcile.settings, "analysis_narrative_overrides_subprocess_guard_enabled", False)
    result = reconcile._llm_reconcile_unresolved(
        [
            {
                "claim_id": "claim-1",
                "claim": "Intervention reduced symptoms",
                "evidence": ["section:results:2"],
                "related_packets": [],
            }
        ]
    )
    assert calls
    assert result == []


def test_synthesis_overrides_uses_deep_model(monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return (
            '{"executive_summary":"Strong trial signal.","methods_strengths":["Randomized design"],'
            '"methods_weaknesses":[],"reproducibility_ethics":[],"uncertainty_gaps":[],'
            '"interpretation":"Effect appears clinically relevant but requires replication."}'
        )

    monkeypatch.setattr(synthesis, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_subprocess_guard_enabled", False)
    overrides = synthesis._llm_synthesis_overrides(
        {
            "paper_meta": {},
            "coverage": {},
            "text_packets": [],
            "table_packets": [],
            "figure_packets": [],
            "supp_packets": [],
            "discrepancies": [],
            "cross_modal_claims": [],
        }
    )
    assert calls
    assert overrides["executive_summary"] == "Strong trial signal."


def test_synthesis_verifier_uses_deep_model(monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_chat(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return (
            '{"executive_summary":"Verified summary","methods_strengths":["Verified strength"],'
            '"methods_weaknesses":["Verified weakness"],"reproducibility_ethics":[],"uncertainty_gaps":[],'
            '"interpretation":"Verified interpretation"}'
        )

    monkeypatch.setattr(synthesis, "chat_text_deep", _fake_chat)
    monkeypatch.setattr(synthesis.settings, "analysis_narrative_overrides_subprocess_guard_enabled", False)
    overrides = synthesis._llm_verifier_overrides(
        {
            "paper_meta": {},
            "coverage": {},
            "text_packets": [],
            "table_packets": [],
            "figure_packets": [],
            "supp_packets": [],
            "discrepancies": [],
            "cross_modal_claims": [],
        },
        {
            "executive_summary": "Draft",
            "interpretation": "",
            "methods_strengths": [],
            "methods_weaknesses": [],
            "reproducibility_ethics": [],
            "uncertainty_gaps": [],
        },
    )
    assert calls
    assert overrides["executive_summary"] == "Verified summary"
