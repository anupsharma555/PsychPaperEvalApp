from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
import pandas as pd
from PIL import Image
from sqlmodel import Session

from app.core.config import settings
from app.db.models import Asset, Chunk
from app.services.analysis.ocr import ocr_image_text, ocr_image_words
from app.services.analysis.utils import extract_refs_from_text
from app.services.storage import artifacts_dir


@dataclass
class FigureAsset:
    path: Path
    caption: str | None
    fig_type: str | None
    page: int | None
    meta: dict[str, Any]


STRUCTURED_ABSTRACT_PREFIX_RE = re.compile(
    r"(?i)\b(objective|objectives|background|aim|aims|purpose|hypothesis|method|methods|design|results|conclusion|conclusions|participants|sample|analysis)\s*:"
)


INTRODUCTION_HEURISTIC_TOKENS: tuple[str, ...] = (
    "introduction",
    "background",
    "objective",
    "aim",
    "aims",
    "rationale",
    "hypoth",
    "motivation",
    "purpose",
)
METHODS_HEURISTIC_TOKENS: tuple[str, ...] = (
    "method",
    "methods",
    "material",
    "materials",
    "participant",
    "participants",
    "sample",
    "procedure",
    "protocol",
    "acquisition",
    "processing",
    "preprocess",
    "analysis",
    "network construction",
    "seed-based",
    "mdmr",
    "assessment",
    "connectivity maps",
    "scanner",
    "imag",
    "supplementary",
    "covariate",
)
RESULTS_HEURISTIC_TOKENS: tuple[str, ...] = (
    "result",
    "finding",
    "found",
    "identified",
    "identify",
    "revealed",
    "hyperconnectivity",
    "hypoconnectivity",
    "dysconnectivity",
    "association",
    "associated",
    "effect",
    "effects",
    "connectivity",
    "foci",
)
DISCUSSION_HEURISTIC_TOKENS: tuple[str, ...] = (
    "discussion",
    "interpret",
    "implication",
    "limitation",
    "limitations",
    "consider",
    "generalizab",
    "to date",
    "in our study",
    "in contrast",
    "consistent with",
)
CONCLUSION_HEURISTIC_TOKENS: tuple[str, ...] = (
    "conclusion",
    "concluding",
    "overall",
    "future research",
    "longitudinal",
    "summary",
    "in summary",
)


def _contains_token(text: str, token: str) -> bool:
    return token in text


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = text
    return any(_contains_token(lowered, token) for token in tokens)


def _infer_section_from_text(text: str, *, idx: int, total_chunks: int, prev_section: str | None = None) -> str:
    lowered = (text or "").lower()
    if not lowered:
        return "unknown"

    # Line-level section signals are usually strongest signals.
    match = re.match(r"\s*(objective|objectives|background|aim|aims|purpose|hypothesis|method|methods|design|results|discussion|conclusion|conclusions)\s*:", lowered)
    if match:
        token = match.group(1)
        if token in {"objective", "objectives", "background", "aim", "aims", "purpose", "hypothesis"}:
            return "introduction"
        if token in {"method", "methods", "design"}:
            return "methods"
        if token in {"results", "result"}:
            return "results"
        if token in {"discussion"}:
            return "discussion"
        if token in {"conclusion", "conclusions"}:
            return "conclusion"

    if _contains_any(lowered, CONCLUSION_HEURISTIC_TOKENS):
        return "conclusion"
    if _contains_any(lowered, DISCUSSION_HEURISTIC_TOKENS):
        return "discussion"
    if _contains_any(lowered, RESULTS_HEURISTIC_TOKENS):
        return "results"
    if _contains_any(lowered, METHODS_HEURISTIC_TOKENS):
        return "methods"
    if _contains_any(lowered, INTRODUCTION_HEURISTIC_TOKENS):
        return "introduction"

    # Positional fallback keeps a coherent narrative flow when headings are weak.
    if not total_chunks:
        return "unknown"
    pos = float(idx) / float(max(1, total_chunks - 1))
    if pos <= 0.18:
        return "introduction"
    if pos <= 0.40:
        return "methods"
    if pos <= 0.74:
        return "results"
    if pos <= 0.90:
        return "discussion"

    # Prefer explicit previous section over hard boundary at the document tail.
    if prev_section in {"introduction", "methods", "results", "discussion"}:
        return prev_section
    return "conclusion"


def _normalize_section_title(title: str) -> str:
    text = " ".join(str(title or "").split()).strip().lower()
    if not text:
        return "unknown"
    if _contains_any(text, CONCLUSION_HEURISTIC_TOKENS):
        return "conclusion"
    if _contains_any(text, DISCUSSION_HEURISTIC_TOKENS):
        return "discussion"
    if _contains_any(text, RESULTS_HEURISTIC_TOKENS):
        return "results"
    if _contains_any(text, METHODS_HEURISTIC_TOKENS):
        return "methods"
    if _contains_any(text, INTRODUCTION_HEURISTIC_TOKENS) or text == "introduction" or "abstract" in text:
        return "introduction"
    return "unknown"


def _split_structured_abstract(text: str) -> list[tuple[str, str]]:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return []
    matches = list(STRUCTURED_ABSTRACT_PREFIX_RE.finditer(clean))
    if not matches:
        return [("abstract", clean)]
    out: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        label = str(match.group(1) or "").strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
        statement = clean[start:end].strip(" -;,:")
        if statement:
            out.append((label or "abstract", statement))
    return out or [("abstract", clean)]


def parse_pdf_validated(
    session: Session,
    document_id: int,
    asset: Asset,
    path: Path,
    counts: dict[str, int],
) -> dict[str, int]:
    _validate_stack()
    tei_xml = _grobid_fulltext(path)
    meta = _extract_tei_metadata(tei_xml)
    if meta:
        chunk = Chunk(
            document_id=document_id,
            asset_id=asset.id,
            anchor="meta:tei",
            modality="meta",
            content=json.dumps(meta),
            meta=None,
        )
        session.add(chunk)
    for idx, text_block in enumerate(_tei_to_text_chunks(tei_xml)):
        chunk = Chunk(
            document_id=document_id,
            asset_id=asset.id,
            anchor=text_block["anchor"],
            modality="text",
            content=text_block["text"],
            meta=text_block.get("meta"),
        )
        session.add(chunk)
        counts["text"] += 1

    figures = _pdffigures2_extract(path, document_id)
    if not figures:
        figures = _fallback_page_images(path, document_id)
    for fig in figures:
        if fig.fig_type and fig.fig_type.lower() == "table":
            table_df = _table_from_image(fig.path)
            if table_df is None:
                continue
            _store_table(
                session,
                document_id,
                asset,
                table_df,
                f"table:{fig.path.stem}",
                extra_meta={
                    "caption": fig.caption,
                    "source": "pdffigures2",
                    "asset_kind": asset.kind,
                    "table_image_path": str(fig.path.resolve()),
                    "page": fig.page,
                    "figure_id": _normalize_figure_id(fig.caption or fig.path.stem),
                    "quality_flags": [],
                },
            )
            counts["table"] += 1
            continue

        ocr_text = _ocr_text_doctr(fig.path) if settings.figure_ocr_parse_enabled else ""
        refs = extract_refs_from_text(f"{fig.caption or ''} {ocr_text or ''}")
        meta_obj = {
            "path": str(fig.path.resolve()),
            "caption": fig.caption,
            "fig_type": fig.fig_type,
            "page": fig.page,
            "source": "pdffigures2",
            "extra": fig.meta,
            "asset_kind": asset.kind,
            "figure_refs": sorted(refs),
            "figure_id": _normalize_figure_id(fig.caption or fig.path.stem),
            "quality_flags": [],
            "ocr_source": "doctr" if ocr_text else None,
        }
        if ocr_text:
            meta_obj["ocr_text"] = ocr_text
        chunk = Chunk(
            document_id=document_id,
            asset_id=asset.id,
            anchor=f"figure:{fig.path.stem}",
            modality="figure",
            content="",
            meta=json.dumps(meta_obj),
        )
        session.add(chunk)
        counts["figure"] += 1

    session.commit()
    return counts


def _extract_tei_metadata(tei_xml: str) -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    def _text(node) -> str:
        if node is None:
            return ""
        return " ".join(node.itertext()).strip()

    try:
        root = ET.fromstring(tei_xml)
    except Exception:
        return {}
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    title = ""
    title_nodes = root.findall(".//tei:titleStmt/tei:title", ns)
    if title_nodes:
        title = _text(title_nodes[0])
    if not title:
        title_node = root.find(".//tei:title[@type='main']", ns)
        title = _text(title_node)

    authors = []
    metadata_source = "tei_unknown"
    author_nodes = root.findall(
        ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblStruct//tei:analytic//tei:author",
        ns,
    )
    if author_nodes:
        metadata_source = "tei_analytic"
    if not author_nodes:
        author_nodes = root.findall(".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:author", ns)
        if author_nodes:
            metadata_source = "tei_title_stmt"
    for author in author_nodes:
        pers = author.find(".//tei:persName", ns)
        if pers is not None:
            forenames = [n.text for n in pers.findall(".//tei:forename", ns) if n.text]
            surname = pers.find(".//tei:surname", ns)
            parts = []
            if forenames:
                parts.append(" ".join(forenames))
            if surname is not None and surname.text:
                parts.append(surname.text)
            name = " ".join(parts).strip()
            if name:
                authors.append(name)
                continue
        name = _text(author)
        if name:
            authors.append(name)
    deduped_authors: list[str] = []
    seen_authors: set[str] = set()
    for name in authors:
        normalized = " ".join(str(name or "").split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_authors:
            continue
        seen_authors.add(key)
        deduped_authors.append(normalized)
    authors_extracted_count = len(deduped_authors)
    authors = deduped_authors[:24]

    journal = ""
    journal_node = root.find(".//tei:sourceDesc//tei:monogr//tei:title[@level='j']", ns)
    if journal_node is None:
        journal_node = root.find(".//tei:sourceDesc//tei:monogr//tei:title", ns)
    journal = _text(journal_node)

    date = ""
    date_node = root.find(".//tei:sourceDesc//tei:monogr//tei:imprint//tei:date", ns)
    if date_node is None:
        date_node = root.find(".//tei:publicationStmt//tei:date", ns)
    if date_node is not None:
        date = date_node.get("when") or _text(date_node)

    meta = {
        "title": title,
        "authors": authors,
        "authors_extracted_count": authors_extracted_count,
        "authors_display_count": len(authors),
        "metadata_source": metadata_source,
        "journal": journal,
        "date": date,
    }
    return {k: v for k, v in meta.items() if v}


def _fallback_page_images(pdf_path: Path, document_id: int) -> list[FigureAsset]:
    try:
        import pypdfium2 as pdfium
    except Exception:
        return []
    fig_dir = artifacts_dir(document_id) / "figures_fallback"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf)
    max_pages = min(settings.figure_fallback_max_pages, total_pages)
    figures: list[FigureAsset] = []
    for i in range(max_pages):
        try:
            page = pdf.get_page(i)
            pil = page.render(scale=settings.figure_fallback_scale).to_pil()
            image_path = fig_dir / f"page_{i + 1}.png"
            pil.save(image_path, "PNG")
            figures.append(
                FigureAsset(
                    path=image_path,
                    caption=f"Page {i + 1} raster fallback",
                    fig_type="page",
                    page=i + 1,
                    meta={"source": "page_raster_fallback"},
                )
            )
        except Exception:
            continue
    return figures


def _validate_stack() -> None:
    _check_grobid()
    _require_pdffigures2()
    _require_tatr()
    _require_doctr()


def _check_grobid() -> None:
    url = settings.grobid_url.rstrip("/") + "/api/isalive"
    try:
        response = httpx.get(url, timeout=10)
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"GROBID is not reachable at {settings.grobid_url}. "
            "Start the Docker container and try again."
        ) from exc
    if response.status_code != 200:
        raise RuntimeError(
            f"GROBID health check failed ({response.status_code}). "
            "Ensure the service is running."
        )


def _require_pdffigures2() -> None:
    if settings.pdffigures2_cmd:
        if shutil.which(settings.pdffigures2_cmd) is None and not Path(
            settings.pdffigures2_cmd
        ).exists():
            raise RuntimeError(
                "PDFFigures2 command not found. Set PDFFIGURES2_CMD or PDFFIGURES2_JAR."
            )
        return
    if settings.pdffigures2_jar:
        if not Path(settings.pdffigures2_jar).exists():
            raise RuntimeError(
                f"PDFFigures2 jar not found at {settings.pdffigures2_jar}."
            )
        return
    raise RuntimeError(
        "PDFFigures2 is not configured. Set PDFFIGURES2_CMD or PDFFIGURES2_JAR."
    )


def _require_tatr() -> None:
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Table Transformer requires torch + transformers. "
            "Install backend requirements and retry."
        ) from exc


def _require_doctr() -> None:
    if not settings.doctr_enabled:
        raise RuntimeError("docTR is disabled (DOCTR_ENABLED=false).")
    try:
        from doctr.models import ocr_predictor  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "docTR is required for OCR in the validated pipeline. "
            "Install backend requirements and retry."
        ) from exc


def _grobid_fulltext(pdf_path: Path) -> str:
    url = settings.grobid_url.rstrip("/") + "/api/processFulltextDocument"
    fields: list[tuple[str, str]] = []
    if settings.grobid_consolidate_header:
        fields.append(("consolidateHeader", "1"))
    if settings.grobid_consolidate_citations:
        fields.append(("consolidateCitations", "1"))
    if settings.grobid_include_coordinates:
        fields.append(("teiCoordinates", "figure"))
        fields.append(("teiCoordinates", "ref"))
        fields.append(("teiCoordinates", "biblStruct"))

    with pdf_path.open("rb") as f:
        files: list[tuple[str, tuple[str | None, str | bytes, str | None]]] = [
            ("input", (pdf_path.name, f, "application/pdf"))
        ]
        for key, value in fields:
            files.append((key, (None, value, None)))
        response = httpx.post(
            url,
            files=files,
            timeout=settings.grobid_timeout_sec,
        )
    if response.status_code != 200:
        raise RuntimeError(f"GROBID error {response.status_code}: {response.text[:200]}")
    return response.text


def _tei_to_text_chunks(tei_xml: str) -> list[dict[str, Any]]:
    import xml.etree.ElementTree as ET

    def _clean(text: str) -> str:
        return " ".join(text.split()).strip()

    root = ET.fromstring(tei_xml)
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    chunks: list[dict[str, Any]] = []

    abstract_idx = 0
    for abstract in root.findall(".//tei:abstract", ns):
        text = _clean(" ".join(abstract.itertext()))
        if not text:
            continue
        for prefix, statement in _split_structured_abstract(text):
            section_norm = _normalize_section_title(prefix or "abstract")
            chunks.append(
                {
                    "anchor": "abstract" if abstract_idx == 0 else f"abstract:{abstract_idx}",
                    "text": statement,
                    "meta": json.dumps(
                        {
                            "section": "abstract",
                            "section_raw_title": prefix or "abstract",
                            "section_norm": section_norm if section_norm != "unknown" else "introduction",
                            "section_path": f"abstract/{abstract_idx}",
                            "paragraph_index": abstract_idx,
                            "source": "grobid",
                        }
                    ),
                }
            )
            abstract_idx += 1

    body = root.find(".//tei:text/tei:body", ns)
    if body is None:
        return chunks

    para_idx = 0
    body_sections: list[tuple[str, list[Any]]] = []
    for div in body.findall("./tei:div", ns):
        head = div.find("./tei:head", ns)
        section_title = _clean(" ".join(head.itertext())) if head is not None else ""
        paras = list(div.findall(".//tei:p", ns))
        body_sections.append((section_title, paras))

    total_paragraphs = max(1, sum(len(ps) for _, ps in body_sections))
    prev_section: str | None = None

    for section_title, paragraphs in body_sections:
        raw_title = (section_title or "body").strip()
        section_norm = _normalize_section_title(raw_title or "body")
        for p in paragraphs:
            text = _clean(" ".join(p.itertext()))
            if not text:
                continue

            normalized_section = section_norm
            if normalized_section == "unknown":
                normalized_section = _infer_section_from_text(
                    text,
                    idx=para_idx,
                    total_chunks=total_paragraphs,
                    prev_section=prev_section,
                )

            anchor = f"section:{raw_title or 'body'}:{para_idx}"
            chunks.append(
                {
                    "anchor": anchor,
                    "text": text,
                    "meta": json.dumps(
                        {
                            "section": raw_title or "body",
                            "section_raw_title": raw_title or "body",
                            "section_norm": normalized_section,
                            "section_path": f"{raw_title or 'body'}/{para_idx}",
                            "paragraph_index": para_idx,
                            "source": "grobid",
                        }
                    ),
                }
            )
            if normalized_section != "unknown":
                prev_section = normalized_section
            para_idx += 1

    if not chunks:
        # Fallback: grab all paragraph text in body
        for p in body.findall(".//tei:p", ns):
            text = _clean(" ".join(p.itertext()))
            if not text:
                continue
            chunks.append(
                {
                    "anchor": f"para:{para_idx}",
                    "text": text,
                    "meta": json.dumps(
                        {
                            "section": "body",
                            "section_raw_title": "body",
                            "section_norm": "unknown",
                            "section_path": f"body/{para_idx}",
                            "paragraph_index": para_idx,
                            "source": "grobid",
                        }
                    ),
                }
            )
            para_idx += 1

    return chunks


def _pdffigures2_extract(pdf_path: Path, document_id: int) -> list[FigureAsset]:
    if settings.pdffigures2_cmd:
        cmd = [settings.pdffigures2_cmd]
    elif settings.pdffigures2_jar:
        cmd = ["java"]
        if settings.pdffigures2_headless:
            cmd.append("-Djava.awt.headless=true")
        cmd += [
            "-cp",
            str(settings.pdffigures2_jar),
            "org.allenai.pdffigures2.FigureExtractorBatchCli",
        ]
    else:
        raise RuntimeError(
            "PDFFigures2 is not configured. Set PDFFIGURES2_CMD or PDFFIGURES2_JAR."
        )

    output_dir = artifacts_dir(document_id) / "pdffigures2"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_prefix = str(output_dir / "data")
    image_prefix = str(output_dir / "img")
    stats_file = str(output_dir / "stats.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_pdf = tmpdir_path / pdf_path.name
        shutil.copy2(pdf_path, tmp_pdf)
        run_cmd = [
            *cmd,
            str(tmpdir_path),
            "-s",
            stats_file,
            "-m",
            image_prefix,
            "-d",
            data_prefix,
        ]
        env = None
        if settings.pdffigures2_headless:
            env = dict(os.environ)
            current = env.get("JAVA_TOOL_OPTIONS", "")
            env["JAVA_TOOL_OPTIONS"] = (current + " -Djava.awt.headless=true").strip()
        try:
            subprocess.run(
                run_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=settings.pdffigures2_timeout_sec,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"PDFFigures2 failed: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc

    json_files = [
        p
        for p in output_dir.rglob("*.json")
        if p.name not in {"stats.json"}
    ]
    if not json_files:
        json_files = [
            p
            for p in output_dir.glob("data*.json")
            if p.name not in {"stats.json"}
        ]
    if not json_files:
        return []
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    data = json.loads(json_files[0].read_text())

    figures: list[FigureAsset] = []
    if not isinstance(data, list):
        return figures
    for item in data:
        if not isinstance(item, dict):
            continue
        caption = item.get("caption") or item.get("captionText")
        fig_type = (
            item.get("figType")
            or item.get("figureType")
            or item.get("type")
            or item.get("figure_type")
        )
        page = item.get("page")
        render_url = item.get("renderURL") or item.get("renderUrl")
        path = _resolve_render_path(render_url, output_dir)
        if path is None or not path.exists():
            continue
        figures.append(
            FigureAsset(
                path=path,
                caption=caption,
                fig_type=fig_type,
                page=page,
                meta=item,
            )
        )
    return figures


def _resolve_render_path(render_url: str | None, output_dir: Path) -> Path | None:
    if not render_url:
        return None
    parsed = urlparse(render_url)
    if parsed.scheme in {"http", "https"}:
        name = Path(parsed.path).name
        candidate = output_dir / name
        if candidate.exists():
            return candidate
        matches = list(output_dir.glob(f"*{name}"))
        return matches[0] if matches else None
    if render_url.startswith("file://"):
        candidate = Path(render_url.replace("file://", ""))
        return candidate if candidate.exists() else None
    candidate = Path(render_url)
    if candidate.exists():
        return candidate
    name = candidate.name
    matches = list(output_dir.glob(f"*{name}"))
    return matches[0] if matches else None


def _store_table(
    session: Session,
    document_id: int,
    asset: Asset,
    df: pd.DataFrame,
    anchor: str,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    content = df.to_json(orient="split")
    meta_obj: dict[str, Any] = {"rows": len(df), "cols": len(df.columns), "source": "validated_pipeline"}
    if extra_meta:
        meta_obj.update(extra_meta)
    chunk = Chunk(
        document_id=document_id,
        asset_id=asset.id,
        anchor=anchor,
        modality="table",
        content=content,
        meta=json.dumps(meta_obj),
    )
    session.add(chunk)


def _table_from_image(image_path: Path) -> pd.DataFrame | None:
    image = Image.open(image_path).convert("RGB")
    rows, cols = _tatr_rows_cols(image)
    words = _doctr_words(image_path)

    if not rows or not cols:
        lines = _doctr_lines(image_path)
        if not lines:
            return None
        return pd.DataFrame({"table": lines})

    rows.sort(key=lambda box: box[1])
    cols.sort(key=lambda box: box[0])

    grid: list[list[str]] = []
    for row_box in rows:
        row_cells: list[str] = []
        for col_box in cols:
            cell_box = _intersect(row_box, col_box)
            if cell_box is None:
                row_cells.append("")
                continue
            cell_words = _words_in_box(words, cell_box)
            row_cells.append(" ".join(cell_words).strip())
        grid.append(row_cells)

    return pd.DataFrame(grid)


def _intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> (
    tuple[float, float, float, float] | None
):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _words_in_box(words: Iterable[dict[str, Any]], box: tuple[float, float, float, float]) -> list[str]:
    x0, y0, x1, y1 = box
    collected: list[tuple[float, str]] = []
    for word in words:
        wx0, wy0, wx1, wy1 = word["bbox"]
        cx = (wx0 + wx1) / 2.0
        cy = (wy0 + wy1) / 2.0
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            collected.append((cx, word["text"]))
    collected.sort(key=lambda item: item[0])
    return [text for _cx, text in collected if text]


def _tatr_rows_cols(image: Image.Image) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    try:
        import torch
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    except Exception as exc:
        raise RuntimeError("transformers + torch are required for Table Transformer.") from exc

    processor, model, device = _tatr_model()
    encoding = processor(images=image, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=settings.tatr_threshold, target_sizes=target_sizes
    )[0]

    rows: list[tuple[float, float, float, float]] = []
    cols: list[tuple[float, float, float, float]] = []
    id2label = model.config.id2label
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label = id2label[int(label_id)]
        if label == "table row":
            rows.append(tuple(map(float, box.tolist())))
        elif label == "table column":
            cols.append(tuple(map(float, box.tolist())))

    return rows, cols


@lru_cache(maxsize=1)
def _tatr_model():
    try:
        import torch
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    except Exception as exc:
        raise RuntimeError("transformers + torch are required for Table Transformer.") from exc

    processor = AutoImageProcessor.from_pretrained(settings.tatr_struct_model)
    model = TableTransformerForObjectDetection.from_pretrained(settings.tatr_struct_model)
    device = _torch_device()
    model.to(device)
    model.eval()
    return processor, model, device


def _torch_device():
    import torch

    device_name = settings.torch_device.lower()
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "mps" and getattr(torch.backends, "mps", None):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _doctr_predictor():
    from app.services.analysis.ocr import _doctr_predictor as shared_predictor

    return shared_predictor()


def _doctr_lines(image_path: Path) -> list[str]:
    from app.services.analysis.ocr import ocr_image_lines

    return ocr_image_lines(image_path)


def _doctr_words(image_path: Path) -> list[dict[str, Any]]:
    return ocr_image_words(image_path)


def _ocr_text_doctr(image_path: Path) -> str:
    if not settings.doctr_enabled:
        return ""
    return ocr_image_text(image_path, max_chars=settings.doctr_max_chars)


def _normalize_figure_id(text: str) -> str | None:
    refs = extract_refs_from_text(text or "")
    if not refs:
        return None
    return sorted(refs)[0]
