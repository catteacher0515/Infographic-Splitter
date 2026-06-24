from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BOX_COLORS = [
    (255, 80, 80),
    (80, 160, 255),
    (80, 200, 120),
    (255, 180, 60),
    (180, 100, 255),
]


def _scale_image(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.convert("RGB"), 1.0

    scale = max_side / float(longest)
    resized = image.convert("RGB").resize(
        (int(width * scale), int(height * scale)),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def save_annotated_candidates(
    image: Image.Image,
    candidates: list[dict],
    output_path: str | Path,
    max_side: int = 1600,
) -> Path:
    annotated, scale = _scale_image(image, max_side)
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for index, candidate in enumerate(candidates):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1 = int(candidate["x"] * scale)
        y1 = int(candidate["y"] * scale)
        x2 = int((candidate["x"] + candidate["width"]) * scale)
        y2 = int((candidate["y"] + candidate["height"]) * scale)
        label = str(candidate["id"])

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label_box = draw.textbbox((x1, y1), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        draw.rectangle(
            (x1, y1, x1 + label_width + 8, y1 + label_height + 6),
            fill=color,
        )
        draw.text((x1 + 4, y1 + 3), label, fill=(0, 0, 0), font=font)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output)
    return output


def image_to_data_url(image_path: str | Path) -> str:
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"
