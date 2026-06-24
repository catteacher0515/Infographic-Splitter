from __future__ import annotations

from pathlib import Path

from PIL import Image


def _crop_image(image: Image.Image, box: dict) -> Image.Image:
    x = int(box["x"])
    y = int(box["y"])
    width = int(box["width"])
    height = int(box["height"])
    return image.crop((x, y, x + width, y + height))


def create_elements(image: Image.Image, boxes: list[dict], output_dir: str | Path) -> list[dict]:
    assets_dir = Path(output_dir) / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    elements = []
    for index, box in enumerate(boxes, start=1):
        filename = str(box.get("file") or f"element_{index:03d}.png")
        preview_path = assets_dir / filename
        _crop_image(image, box).save(preview_path)

        element = {
            "id": index,
            "file": filename,
            "x": int(box["x"]),
            "y": int(box["y"]),
            "width": int(box["width"]),
            "height": int(box["height"]),
            "selected": bool(box.get("selected", True)),
            "preview_path": str(preview_path),
        }
        for key in ("type", "source_candidate_ids", "reason"):
            if key in box:
                element[key] = box[key]
        elements.append(element)

    return elements
