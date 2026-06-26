from PIL import Image, ImageDraw

from bbox_refiner import (
    apply_remove_regions,
    bbox_from_trim,
    parse_refine_response,
    parse_remove_regions_response,
    parse_trim_response,
    validate_bbox,
)


def test_parse_refine_response_reads_bbox_from_json():
    response = '{"bbox":{"x":10,"y":20,"width":300,"height":200},"reason":"keep subject"}'

    result = parse_refine_response(response)

    assert result["bbox"] == {"x": 10, "y": 20, "width": 300, "height": 200}
    assert result["reason"] == "keep subject"


def test_validate_bbox_accepts_reasonable_crop():
    bbox = {"x": 10, "y": 20, "width": 400, "height": 300}

    validated = validate_bbox(bbox, image_width=453, image_height=352)

    assert validated == bbox


def test_validate_bbox_rejects_too_small_crop():
    bbox = {"x": 10, "y": 20, "width": 100, "height": 100}

    try:
        validate_bbox(bbox, image_width=453, image_height=352)
    except ValueError as error:
        assert "too small" in str(error)
    else:
        raise AssertionError("validate_bbox should reject too-small crops")


def test_validate_bbox_rejects_out_of_bounds_crop():
    bbox = {"x": 10, "y": 20, "width": 500, "height": 300}

    try:
        validate_bbox(bbox, image_width=453, image_height=352)
    except ValueError as error:
        assert "outside image bounds" in str(error)
    else:
        raise AssertionError("validate_bbox should reject out-of-bounds crops")


def test_parse_trim_response_reads_trim_from_json():
    response = '{"trim":{"left":40,"top":0,"right":20,"bottom":0},"reason":"remove arrows"}'

    result = parse_trim_response(response)

    assert result["trim"] == {"left": 40, "top": 0, "right": 20, "bottom": 0}
    assert result["reason"] == "remove arrows"


def test_bbox_from_trim_converts_margins_to_bbox():
    trim = {"left": 40, "top": 0, "right": 20, "bottom": 0}

    bbox = bbox_from_trim(trim, image_width=453, image_height=352)

    assert bbox == {"x": 40, "y": 0, "width": 393, "height": 352}


def test_parse_remove_regions_response_reads_remove_regions_from_json():
    response = (
        '{"remove_regions":[{"label":"left_arrow",'
        '"bbox":{"x":0,"y":35,"width":85,"height":60}}],'
        '"reason":"remove arrow"}'
    )

    result = parse_remove_regions_response(response)

    assert result["remove_regions"] == [
        {
            "label": "left_arrow",
            "bbox": {"x": 0, "y": 35, "width": 85, "height": 60},
        }
    ]
    assert result["reason"] == "remove arrow"


def test_apply_remove_regions_fills_removed_area_with_background():
    image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 25, 40, 55), fill="black")
    draw.rectangle((60, 20, 95, 60), fill="black")

    cleaned = apply_remove_regions(
        image,
        [
            {"label": "left_arrow", "bbox": {"x": 10, "y": 25, "width": 30, "height": 30}},
        ],
    )

    assert cleaned.getpixel((20, 40)) == (255, 255, 255)
    assert cleaned.getpixel((75, 40)) == (0, 0, 0)
