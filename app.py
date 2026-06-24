from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image

from detector import detect_elements
from exporter import export_zip
from splitter import create_elements


OUTPUT_ROOT = Path("output")


def build_session_output_dir(output_root: str | Path = OUTPUT_ROOT) -> Path:
    session_dir = Path(output_root) / f"session-{uuid4().hex[:12]}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def elements_to_rows(elements: list[dict]) -> list[list]:
    return [
        [
            bool(element.get("selected", True)),
            element["file"],
            element["x"],
            element["y"],
            element["width"],
            element["height"],
        ]
        for element in elements
    ]


def rows_to_elements(rows, elements: list[dict]) -> list[dict]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()

    updated = []
    for row, element in zip(rows, elements):
        selected, filename, *_ = row
        copied = dict(element)
        copied["selected"] = bool(selected)
        copied["file"] = str(filename)
        updated.append(copied)
    return updated


def run_split(
    image: Image.Image,
    min_area: int = 500,
    merge_gap: int = 8,
    padding: int = 10,
    output_root: str | Path = OUTPUT_ROOT,
) -> tuple[list[str], list[list], list[dict]]:
    if image is None:
        return [], [], []

    session_dir = build_session_output_dir(output_root)
    boxes = detect_elements(
        image,
        min_area=int(min_area),
        merge_gap=int(merge_gap),
        padding=int(padding),
    )
    elements = create_elements(image, boxes, session_dir)
    gallery = [element["preview_path"] for element in elements]
    return gallery, elements_to_rows(elements), elements


def run_export(rows, elements: list[dict], output_root: str | Path = OUTPUT_ROOT) -> str | None:
    if not elements:
        return None

    session_dir = build_session_output_dir(output_root)
    updated_elements = rows_to_elements(rows, elements)
    return export_zip(updated_elements, session_dir)
