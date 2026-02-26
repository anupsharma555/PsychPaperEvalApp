from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings


def ocr_image_lines(image_path: str | Path) -> list[str]:
    predictor = _doctr_predictor()
    if predictor is None:
        return []

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return []

    try:
        from doctr.io import DocumentFile
    except Exception:
        return []

    try:
        doc = DocumentFile.from_images(str(path))
        result = predictor(doc)
    except Exception:
        return []

    if not result.pages:
        return []

    lines: list[str] = []
    try:
        page = result.pages[0]
    except Exception:
        return []

    try:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words if word.value)
                if line_text:
                    lines.append(line_text)
    except Exception:
        return []

    return lines


def ocr_image_text(image_path: str | Path, max_chars: int | None = None) -> str:
    if max_chars is None:
        max_chars = settings.doctr_max_chars
    lines = ocr_image_lines(image_path)
    if not lines:
        return ""
    joined = " ".join(lines)
    try:
        limit = int(max_chars)
    except Exception:
        limit = settings.doctr_max_chars
    if limit > 0 and len(joined) > limit:
        joined = joined[:limit]
    return joined


def ocr_image_words(image_path: str | Path) -> list[dict[str, Any]]:
    predictor = _doctr_predictor()
    if predictor is None:
        return []

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return []

    try:
        from doctr.io import DocumentFile
    except Exception:
        return []

    try:
        doc = DocumentFile.from_images(str(path))
        result = predictor(doc)
    except Exception:
        return []

    if not result.pages:
        return []

    try:
        page = result.pages[0]
    except Exception:
        return []

    words: list[dict[str, Any]] = []
    try:
        width, height = page.dimensions
    except Exception:
        return []

    try:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if not word.value:
                        continue
                    geom = word.geometry
                    if not geom:
                        continue
                    try:
                        if len(geom) == 2 and len(geom[0]) == 2:
                            (x0, y0), (x1, y1) = geom
                        else:
                            points = []
                            for point in geom:
                                if isinstance(point, (list, tuple)) and len(point) >= 2:
                                    points.append((point[0], point[1]))
                            if not points:
                                continue
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                        words.append(
                            {
                                "text": word.value,
                                "bbox": (x0 * width, y0 * height, x1 * width, y1 * height),
                            }
                        )
                    except Exception:
                        continue
    except Exception:
        return []

    return words


@lru_cache(maxsize=1)
def _doctr_predictor():
    if not settings.doctr_enabled:
        return None
    try:
        from doctr.models import ocr_predictor
    except Exception:
        return None

    try:
        predictor = ocr_predictor(
            det_arch=settings.doctr_det_arch,
            reco_arch=settings.doctr_reco_arch,
            pretrained=True,
            assume_straight_pages=False,
        )
    except Exception:
        return None

    try:
        import torch

        device = _torch_device()
        predictor.det_model.to(device)
        predictor.reco_model.to(device)
    except Exception:
        pass
    return predictor


def _torch_device():
    try:
        import torch
    except Exception:
        return None

    device_name = settings.torch_device.lower()
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "mps" and getattr(torch.backends, "mps", None):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")
