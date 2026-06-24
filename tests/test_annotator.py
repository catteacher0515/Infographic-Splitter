from pathlib import Path

from PIL import Image

from annotator import image_to_data_url, save_annotated_candidates


def test_save_annotated_candidates_draws_boxes_and_ids(tmp_path: Path):
    image = Image.new("RGB", (240, 160), "white")
    candidates = [
        {"id": 1, "x": 20, "y": 20, "width": 80, "height": 50},
        {"id": 2, "x": 140, "y": 60, "width": 60, "height": 70},
    ]

    output_path = save_annotated_candidates(image, candidates, tmp_path / "annotated.png")

    assert output_path.exists()
    annotated = Image.open(output_path).convert("RGB")
    assert annotated.size == image.size
    assert annotated.getpixel((20, 20)) != (255, 255, 255)


def test_save_annotated_candidates_resizes_large_image(tmp_path: Path):
    image = Image.new("RGB", (4000, 2000), "white")
    candidates = [{"id": 1, "x": 100, "y": 100, "width": 300, "height": 200}]

    output_path = save_annotated_candidates(
        image,
        candidates,
        tmp_path / "annotated.png",
        max_side=1000,
    )

    annotated = Image.open(output_path)
    assert annotated.size == (1000, 500)


def test_image_to_data_url_returns_png_base64(tmp_path: Path):
    image = Image.new("RGB", (80, 40), "white")
    path = tmp_path / "image.png"
    image.save(path)

    data_url = image_to_data_url(path)

    assert data_url.startswith("data:image/png;base64,")
