from pathlib import Path

from PIL import Image, ImageDraw

from app import build_session_output_dir, run_split


def test_build_session_output_dir_creates_unique_session(tmp_path: Path):
    first = build_session_output_dir(tmp_path)
    second = build_session_output_dir(tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_run_split_returns_gallery_rows_and_elements(tmp_path: Path):
    image = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 100, 80), outline="black", width=4)

    gallery, rows, elements = run_split(
        image,
        min_area=200,
        merge_gap=5,
        padding=0,
        output_root=tmp_path,
    )

    assert len(gallery) == 1
    assert len(rows) == 1
    assert len(elements) == 1
    assert rows[0][0] is True
    assert rows[0][1] == "element_001.png"
