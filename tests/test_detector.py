from PIL import Image, ImageDraw

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
