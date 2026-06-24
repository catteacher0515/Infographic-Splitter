from pathlib import Path

from PIL import Image, ImageDraw

from splitter import create_elements


def test_create_elements_saves_crops_and_metadata(tmp_path: Path):
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 60, 70), outline="black", width=4)
    boxes = [{"x": 10, "y": 20, "width": 50, "height": 50, "area": 2500}]

    elements = create_elements(image, boxes, tmp_path)

    assert len(elements) == 1
    assert elements[0]["id"] == 1
    assert elements[0]["file"] == "element_001.png"
    assert elements[0]["selected"] is True
    assert elements[0]["x"] == 10
    assert elements[0]["y"] == 20
    assert elements[0]["width"] == 50
    assert elements[0]["height"] == 50
    assert Path(elements[0]["preview_path"]).exists()


def test_create_elements_uses_box_order_for_names(tmp_path: Path):
    image = Image.new("RGB", (300, 200), "white")
    boxes = [
        {"x": 100, "y": 20, "width": 40, "height": 30, "area": 1200},
        {"x": 20, "y": 20, "width": 40, "height": 30, "area": 1200},
    ]

    elements = create_elements(image, boxes, tmp_path)

    assert [element["file"] for element in elements] == [
        "element_001.png",
        "element_002.png",
    ]
    assert [element["x"] for element in elements] == [100, 20]


def test_create_elements_uses_custom_file_and_metadata(tmp_path: Path):
    image = Image.new("RGB", (200, 120), "white")
    boxes = [
        {
            "x": 10,
            "y": 20,
            "width": 80,
            "height": 50,
            "file": "loop_cycle.png",
            "type": "illustration",
            "source_candidate_ids": [1, 2],
            "reason": "same loop",
            "selected": False,
        }
    ]

    elements = create_elements(image, boxes, tmp_path)

    assert elements[0]["file"] == "loop_cycle.png"
    assert elements[0]["type"] == "illustration"
    assert elements[0]["source_candidate_ids"] == [1, 2]
    assert elements[0]["selected"] is False
