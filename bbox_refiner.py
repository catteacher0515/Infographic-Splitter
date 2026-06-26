from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from ai_grouper import extract_json_object


SYSTEM_PROMPT = """你是一个黑白手绘信息图插图裁剪框精修助手。

你只负责返回更紧的 bbox，不要解释过程，不要返回 Markdown。
坐标必须相对于输入图片本身，不是原始大图。"""


USER_PROMPT = """请识别图片中真正属于这个插图主体的区域，并返回更紧的 bbox。

主体应保留：
- 资料堆
- 小猫
- 电脑
- 周围文件
- 文件夹
- 必要的动作线或装饰线

应排除：
- 无关箭头
- 其他模块残留
- 过多空白
- 不属于这个插图场景的外部元素

请严格返回 JSON：
{
  "bbox": {
    "x": 0,
    "y": 0,
    "width": 100,
    "height": 100
  },
  "reason": "short reason"
}
"""


TRIM_PROMPT = """请返回为了去除无关边缘元素而需要从四边裁掉的像素边距。

主体必须保留：
- 资料堆
- 小猫
- 电脑
- 周围文件
- 文件夹
- 必要的动作线或装饰线

应裁掉：
- 左侧无关箭头
- 右侧边缘残留箭头
- 过多空白
- 不属于这个插图场景的外部元素

不要裁掉顶部文件或右侧文件夹。

请严格返回 JSON：
{
  "trim": {
    "left": 0,
    "top": 0,
    "right": 0,
    "bottom": 0
  },
  "reason": "short reason"
}
"""


SYSTEM_PROMPT_REMOVE = """你是一个黑白手绘知识信息图的冗余对象删除助手。

你只负责标出可以删除的冗余对象区域，不要改动主体，不要返回 Markdown。
坐标必须相对于输入图片本身。"""


USER_PROMPT_REMOVE = """请识别图片中可以删除的冗余对象，并返回它们的 bbox。

适合删除的对象：
- 左右边缘残留箭头
- 与主体无关的碎片
- 重复装饰线
- 远离主体的残留符号

不要删除：
- 主体本身
- 主体内部必要线条
- 关键说明文字
- 会影响主体识别的内容

请严格返回 JSON：
{
  "remove_regions": [
    {
      "label": "left_arrow",
      "bbox": {
        "x": 0,
        "y": 35,
        "width": 85,
        "height": 60
      }
    }
  ],
  "reason": "short reason"
}
"""


def build_refine_messages(
    image_data_url: str,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[dict]:
    prompt = USER_PROMPT
    if image_width is not None and image_height is not None:
        prompt = (
            f"输入图片尺寸：width={int(image_width)}, height={int(image_height)}。\n"
            f"bbox 必须满足：0 <= x < {int(image_width)}, 0 <= y < {int(image_height)}, "
            f"x + width <= {int(image_width)}, y + height <= {int(image_height)}。\n\n"
            f"{USER_PROMPT}"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_trim_messages(
    image_data_url: str,
    image_width: int,
    image_height: int,
) -> list[dict]:
    prompt = (
        f"输入图片尺寸：width={int(image_width)}, height={int(image_height)}。\n"
        "left/top/right/bottom 都是从对应图片边缘向内裁掉的像素数。\n"
        f"约束：0 <= left < {int(image_width)}, 0 <= right < {int(image_width)}, "
        f"0 <= top < {int(image_height)}, 0 <= bottom < {int(image_height)}。\n"
        "裁剪后必须保留主体完整。\n\n"
        f"{TRIM_PROMPT}"
    )
    return [
        {"role": "system", "content": "你是一个插图裁剪边距精修助手。只返回 JSON。"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_remove_messages(
    image_data_url: str,
    image_width: int,
    image_height: int,
) -> list[dict]:
    prompt = (
        f"输入图片尺寸：width={int(image_width)}, height={int(image_height)}。\n"
        "remove_regions 里的 bbox 坐标都必须相对于输入图片本身。\n"
        f"约束：0 <= x < {int(image_width)}, 0 <= y < {int(image_height)}, "
        f"x + width <= {int(image_width)}, y + height <= {int(image_height)}。\n"
        "如果没有可删除对象，可以返回空数组。\n\n"
        f"{USER_PROMPT_REMOVE}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_REMOVE},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def parse_refine_response(text: str) -> dict:
    parsed = extract_json_object(text)
    bbox = parsed.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError("Model response does not contain bbox")
    return {
        "bbox": {
            "x": int(bbox["x"]),
            "y": int(bbox["y"]),
            "width": int(bbox["width"]),
            "height": int(bbox["height"]),
        },
        "reason": str(parsed.get("reason", "")),
    }


def parse_trim_response(text: str) -> dict:
    parsed = extract_json_object(text)
    trim = parsed.get("trim")
    if not isinstance(trim, dict):
        raise ValueError("Model response does not contain trim")
    return {
        "trim": {
            "left": int(trim["left"]),
            "top": int(trim["top"]),
            "right": int(trim["right"]),
            "bottom": int(trim["bottom"]),
        },
        "reason": str(parsed.get("reason", "")),
    }


def parse_remove_regions_response(text: str) -> dict:
    parsed = extract_json_object(text)
    remove_regions = parsed.get("remove_regions")
    if not isinstance(remove_regions, list):
        raise ValueError("Model response does not contain remove_regions")

    cleaned_regions = []
    for region in remove_regions:
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, dict):
            continue
        cleaned_regions.append(
            {
                "label": str(region.get("label", "")).strip(),
                "bbox": {
                    "x": int(bbox["x"]),
                    "y": int(bbox["y"]),
                    "width": int(bbox["width"]),
                    "height": int(bbox["height"]),
                },
            }
        )

    return {
        "remove_regions": cleaned_regions,
        "reason": str(parsed.get("reason", "")),
    }


def bbox_from_trim(trim: dict, image_width: int, image_height: int) -> dict:
    left = int(trim["left"])
    top = int(trim["top"])
    right = int(trim["right"])
    bottom = int(trim["bottom"])
    return {
        "x": left,
        "y": top,
        "width": int(image_width) - left - right,
        "height": int(image_height) - top - bottom,
    }


def validate_bbox(
    bbox: dict,
    image_width: int,
    image_height: int,
    min_area_ratio: float = 0.40,
) -> dict:
    x = int(bbox["x"])
    y = int(bbox["y"])
    width = int(bbox["width"])
    height = int(bbox["height"])

    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("bbox has invalid dimensions")
    if x + width > image_width or y + height > image_height:
        raise ValueError("bbox is outside image bounds")

    image_area = image_width * image_height
    bbox_area = width * height
    if bbox_area < image_area * min_area_ratio:
        raise ValueError("bbox is too small")

    return {"x": x, "y": y, "width": width, "height": height}


def _estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    border_pixels: list[tuple[int, int, int]] = []

    for x in range(width):
        border_pixels.append(rgb_image.getpixel((x, 0)))
        if height > 1:
            border_pixels.append(rgb_image.getpixel((x, height - 1)))

    for y in range(1, max(1, height - 1)):
        border_pixels.append(rgb_image.getpixel((0, y)))
        if width > 1:
            border_pixels.append(rgb_image.getpixel((width - 1, y)))

    if not border_pixels:
        return (255, 255, 255)

    bright_pixels = [
        pixel
        for pixel in border_pixels
        if sum(pixel) / 3 >= 180
    ]
    palette = bright_pixels or border_pixels
    return Counter(palette).most_common(1)[0][0]


def apply_remove_regions(image: Image.Image, remove_regions: list[dict]) -> Image.Image:
    cleaned = image.convert("RGB").copy()
    draw = ImageDraw.Draw(cleaned)
    fill_color = _estimate_background_color(cleaned)
    width, height = cleaned.size

    for region in remove_regions:
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            valid_bbox = validate_bbox(
                bbox,
                image_width=width,
                image_height=height,
                min_area_ratio=0.0,
            )
        except Exception:
            continue

        x = int(valid_bbox["x"])
        y = int(valid_bbox["y"])
        box_width = int(valid_bbox["width"])
        box_height = int(valid_bbox["height"])
        draw.rectangle(
            (x, y, x + box_width - 1, y + box_height - 1),
            fill=fill_color,
        )

    return cleaned


def crop_with_bbox(input_path: str | Path, output_path: str | Path, bbox: dict) -> Path:
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        x = int(bbox["x"])
        y = int(bbox["y"])
        width = int(bbox["width"])
        height = int(bbox["height"])
        crop = image.crop((x, y, x + width, y + height))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output)
    return output
