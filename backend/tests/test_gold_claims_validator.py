from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_gold_claims import GoldClaimValidationError, load_gold_claims


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARMA_GOLD_CLAIMS = PROJECT_ROOT / "benchmarks" / "sharma_2017_gold_claims.jsonl"


def _valid_claim(**overrides: object) -> dict[str, object]:
    claim: dict[str, object] = {
        "claim_id": "claim-001",
        "section": "methods",
        "claim_type": "sample",
        "importance": "high",
        "evidence_quote": "The study initially assessed 244 participants.",
        "page_or_anchor": "test/text/sharma_2017_chatgpt_extraction.md#METHODS",
        "expected_entities": ["participants"],
        "expected_numbers": [{"label": "initial_sample", "value": 244, "unit": "participants"}],
        "priority": "P1",
    }
    claim.update(overrides)
    return claim


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_sharma_gold_claims_fixture_is_machine_checkable() -> None:
    claims = load_gold_claims(SHARMA_GOLD_CLAIMS)

    assert len(claims) >= 5
    assert {claim["section"] for claim in claims} == {
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    }
    assert any(claim["expected_numbers"] for claim in claims)
    assert all(claim["evidence_quote"] for claim in claims)


def test_gold_claim_validator_fails_on_duplicate_ids(tmp_path: Path) -> None:
    fixture = _write_jsonl(
        tmp_path / "gold_claims.jsonl",
        [_valid_claim(), _valid_claim(section="results", evidence_quote="A second quote.")],
    )

    with pytest.raises(GoldClaimValidationError, match="duplicate claim_id"):
        load_gold_claims(fixture)


def test_gold_claim_validator_fails_on_missing_sections(tmp_path: Path) -> None:
    fixture = _write_jsonl(tmp_path / "gold_claims.jsonl", [_valid_claim(section="")])

    with pytest.raises(GoldClaimValidationError, match="section must be one of"):
        load_gold_claims(fixture)


def test_gold_claim_validator_fails_on_missing_evidence(tmp_path: Path) -> None:
    fixture = _write_jsonl(tmp_path / "gold_claims.jsonl", [_valid_claim(evidence_quote="")])

    with pytest.raises(GoldClaimValidationError, match="evidence_quote must be a non-empty string"):
        load_gold_claims(fixture)


def test_gold_claim_validator_fails_on_invalid_priorities(tmp_path: Path) -> None:
    fixture = _write_jsonl(tmp_path / "gold_claims.jsonl", [_valid_claim(priority="urgent")])

    with pytest.raises(GoldClaimValidationError, match="priority must be one of"):
        load_gold_claims(fixture)


def test_gold_claim_validator_fails_on_malformed_expected_numeric_fields(tmp_path: Path) -> None:
    fixture = _write_jsonl(
        tmp_path / "gold_claims.jsonl",
        [_valid_claim(expected_numbers=[{"label": "initial_sample", "value": "244"}])],
    )

    with pytest.raises(GoldClaimValidationError, match="expected_numbers\\[0\\].value must be a number"):
        load_gold_claims(fixture)
