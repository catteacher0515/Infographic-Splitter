from PIL import Image, ImageDraw
import pytest

from detector import detect_elements, sort_boxes


def make_canvas(width=420, height=240):
    return Image.new("RGB", (width, height), "white")


def test_detects_separate_visual_blocks_top_to_bottom_left_to_right():
    image = make_canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 100, 70), outline="black", width=4)
    draw.rectangle((180, 20, 280, 70), outline="black", width=4)
    draw.rectangle((30, 150, 120, 210), outline="black", width=4)

    boxes = detect_elements(image, min_area=200, merge_gap=5, padding=0)

    assert len(boxes) == 3
    assert boxes[0]["x"] < boxes[1]["x"]
    assert boxes[0]["y"] < boxes[2]["y"]
    assert boxes[1]["y"] < boxes[2]["y"]


def test_filters_small_noise_by_min_area():
    image = make_canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 120, 90), outline="black", width=4)
    draw.ellipse((300, 30, 305, 35), fill="black")

    boxes = detect_elements(image, min_area=500, merge_gap=3, padding=0)

    assert len(boxes) == 1
    assert boxes[0]["width"] > 80


def test_merge_gap_combines_nearby_strokes_into_one_visual_block():
    image = make_canvas(width=240, height=140)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 70, 90), outline="black", width=4)
    draw.rectangle((96, 34, 142, 88), outline="black", width=4)

    boxes = detect_elements(image, min_area=200, merge_gap=18, padding=0)

    assert len(boxes) == 1
    assert boxes[0]["x"] <= 22
    assert boxes[0]["width"] >= 118


def test_merge_gap_keeps_distant_blocks_separate():
    image = make_canvas(width=320, height=140)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 70, 90), outline="black", width=4)
    draw.rectangle((220, 34, 270, 88), outline="black", width=4)

    boxes = detect_elements(image, min_area=200, merge_gap=18, padding=0)

    assert len(boxes) == 2
    assert boxes[0]["x"] < boxes[1]["x"]


def test_merge_gap_rejects_wide_horizontal_chain_merge():
    image = make_canvas(width=1000, height=260)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 80, 170, 180), outline="black", width=4)
    draw.rectangle((210, 84, 340, 176), outline="black", width=4)
    draw.rectangle((380, 82, 510, 178), outline="black", width=4)

    boxes = detect_elements(image, min_area=200, merge_gap=24, padding=0)

    assert len(boxes) == 3


def test_merge_gap_allows_wide_short_text_line_merge():
    image = make_canvas(width=420, height=140)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 50, 70, 90), outline="black", width=4)
    draw.rectangle((96, 52, 170, 88), outline="black", width=4)
    draw.rectangle((196, 52, 290, 88), outline="black", width=4)

    boxes = detect_elements(image, min_area=200, merge_gap=24, padding=0)

    assert len(boxes) == 1
    assert boxes[0]["width"] >= 255
    assert boxes[0]["height"] < 70


def test_small_text_stroke_can_merge_before_min_area_filter():
    image = make_canvas(width=260, height=120)
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 58, 54, 64), fill="black")
    draw.rectangle((82, 42, 150, 82), outline="black", width=4)

    boxes = detect_elements(image, min_area=500, merge_gap=16, padding=0)

    assert len(boxes) == 1
    assert boxes[0]["x"] <= 26
    assert boxes[0]["width"] >= 126


def test_small_text_stroke_can_merge_when_padding_overlaps_boxes():
    image = make_canvas(width=260, height=180)
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 58, 54, 64), fill="black")
    draw.rectangle((62, 42, 132, 82), outline="black", width=4)

    boxes = detect_elements(image, min_area=500, merge_gap=8, padding=10)

    assert len(boxes) == 1
    assert boxes[0]["x"] <= 16
    assert boxes[0]["width"] >= 125


def test_small_stroke_does_not_bridge_large_block_to_text():
    image = make_canvas(width=300, height=260)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 20, 180, 110), outline="black", width=4)
    draw.rectangle((98, 126, 140, 132), fill="black")
    draw.rectangle((98, 142, 140, 148), fill="black")
    draw.rectangle((72, 160, 170, 200), outline="black", width=4)

    boxes = detect_elements(image, min_area=500, merge_gap=8, padding=10)

    assert len(boxes) == 2
    assert boxes[0]["height"] < 120
    assert boxes[1]["height"] < 70


def test_padding_expands_box_and_clamps_to_image_bounds():
    image = make_canvas(width=120, height=100)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 40, 30), outline="black", width=4)

    boxes = detect_elements(image, min_area=100, merge_gap=3, padding=10)

    assert len(boxes) == 1
    assert boxes[0]["x"] == 0
    assert boxes[0]["y"] == 0
    assert boxes[0]["width"] >= 45
    assert boxes[0]["height"] >= 35


def test_sort_boxes_uses_row_then_column_order():
    boxes = [
        {"x": 200, "y": 20, "width": 30, "height": 20, "area": 600},
        {"x": 20, "y": 120, "width": 30, "height": 20, "area": 600},
        {"x": 20, "y": 25, "width": 30, "height": 20, "area": 600},
    ]

    sorted_boxes = sort_boxes(boxes)

    assert [(box["x"], box["y"]) for box in sorted_boxes] == [
        (20, 25),
        (200, 20),
        (20, 120),
    ]


def test_reference_overview_does_not_create_cross_column_visual_block():
    image_path = "/Users/huapingyu/Desktop/手绘视频/ AI 发展四个阶段/总览图.png"
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        pytest.skip("local reference image is not available")

    boxes = detect_elements(image, min_area=500, merge_gap=8, padding=10)

    assert max(box["width"] for box in boxes) < 650
