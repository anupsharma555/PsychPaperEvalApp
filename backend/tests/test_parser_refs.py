from __future__ import annotations

from app.services.analysis.utils import extract_expected_refs, extract_refs_from_text
from app.services.parser import _safe_zip_name


def test_reference_extraction_handles_ranges_and_supplement_refs() -> None:
    text = "See Figure 1-3 and Fig S1 plus Table 2."
    refs = extract_refs_from_text(text)
    assert {"1", "2", "3", "S1"} <= refs

    expected = extract_expected_refs([text])
    assert "2" in expected["table_refs"]
    assert "S1" in expected["figure_refs"]


def test_safe_zip_name_blocks_path_traversal() -> None:
    assert _safe_zip_name("../etc/passwd") == ""
    assert _safe_zip_name("nested/../../escape.txt") == ""
    assert _safe_zip_name("supp/data/table.csv") == "supp/data/table.csv"
