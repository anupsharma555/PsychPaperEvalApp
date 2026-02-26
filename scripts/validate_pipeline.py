from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings  # noqa: E402
from app.services.validated_pipeline import (  # noqa: E402
    _grobid_fulltext,
    _ocr_text_doctr,
    _pdffigures2_extract,
    _table_from_image,
    _tei_to_text_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to a PDF for validated pipeline testing")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    print("Validating GROBID extraction...")
    tei_xml = _grobid_fulltext(pdf_path)
    text_chunks = _tei_to_text_chunks(tei_xml)
    print(f"Text chunks: {len(text_chunks)}")

    print("Running PDFFigures2 extraction...")
    doc_id = int(time.time())
    figures = _pdffigures2_extract(pdf_path, doc_id)
    print(f"Figures extracted: {len(figures)}")

    table_count = 0
    ocr_nonempty = 0

    for fig in figures:
        if fig.fig_type and fig.fig_type.lower() == "table":
            try:
                table_df = _table_from_image(fig.path)
            except Exception as exc:
                print(f"Table extraction error for {fig.path.name}: {exc}")
                continue
            if table_df is not None and not table_df.empty:
                table_count += 1
            continue

        try:
            ocr_text = _ocr_text_doctr(fig.path)
        except Exception as exc:
            print(f"OCR error for {fig.path.name}: {exc}")
            continue
        if ocr_text.strip():
            ocr_nonempty += 1

    print(f"Tables parsed: {table_count}")
    print(f"Figures with OCR text: {ocr_nonempty}")

    if not text_chunks:
        print("WARNING: No text chunks extracted.")
    if figures and ocr_nonempty == 0:
        print("WARNING: OCR produced no text from extracted figures.")
    if figures and table_count == 0:
        print("WARNING: No tables parsed from extracted figures.")


if __name__ == "__main__":
    main()
