from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotator import image_to_data_url
from bbox_refiner import (
    bbox_from_trim,
    build_trim_messages,
    crop_with_bbox,
    parse_trim_response,
    validate_bbox,
)
from vision_client import QwenVisionClient


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_refined.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine one illustration crop bbox with Qwen vision.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or default_output_path(input_path)

    with Image.open(input_path) as image:
        image_width, image_height = image.size

    messages = build_trim_messages(
        image_to_data_url(input_path),
        image_width=image_width,
        image_height=image_height,
    )
    raw_response = QwenVisionClient().complete(messages)
    parsed = parse_trim_response(raw_response)
    proposed_bbox = bbox_from_trim(
        parsed["trim"],
        image_width=image_width,
        image_height=image_height,
    )
    bbox = validate_bbox(
        proposed_bbox,
        image_width=image_width,
        image_height=image_height,
    )
    crop_with_bbox(input_path, output_path, bbox)

    print("input:", input_path)
    print("output:", output_path)
    print("trim:", parsed["trim"])
    print("bbox:", bbox)
    print("reason:", parsed["reason"])
    print("raw_response:", raw_response)


if __name__ == "__main__":
    main()
