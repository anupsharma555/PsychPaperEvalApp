from __future__ import annotations

from app.services.analysis import reconcile, synthesis, text_analysis


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
