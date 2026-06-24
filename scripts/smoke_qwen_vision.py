from __future__ import annotations

import base64
import io
import os

from openai import OpenAI
from PIL import Image, ImageDraw


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-flash"


def build_test_image_data_url() -> str:
    image = Image.new("RGB", (360, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 150, 120), outline="black", width=5)
    draw.ellipse((220, 40, 320, 140), outline="black", width=5)
    draw.text((52, 70), "BOX", fill="black")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL),
    )
    response = client.chat.completions.create(
        model=os.getenv("QWEN_MODEL", DEFAULT_MODEL),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请用一句中文说明图中有哪些黑色线稿元素。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": build_test_image_data_url()},
                    },
                ],
            }
        ],
        temperature=0,
    )

    print("model:", response.model)
    print("content:", response.choices[0].message.content)


if __name__ == "__main__":
    main()
