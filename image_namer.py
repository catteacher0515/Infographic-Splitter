from __future__ import annotations

from pathlib import Path

from ai_grouper import extract_json_object


SYSTEM_PROMPT = """你是一个图片内容命名助手。

你只根据图片内容生成简洁、可检索的中文文件名。
不要返回 Markdown，不要返回解释性正文，只返回 JSON。"""


USER_PROMPT = """请识别这张图片的主要内容，并生成一个简洁的中文文件名。

要求：
- 使用中文
- 需要时可用下划线连接多个词
- 文件名要具体，方便后续检索
- 不要带路径
- 可以带 .png 扩展名

请严格返回 JSON：
{
  "file": "prompt_card.png",
  "reason": "short reason"
}
"""


def build_naming_messages(image_data_url: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def parse_naming_response(text: str) -> dict:
    parsed = extract_json_object(text)
    raw_file = parsed.get("file") or parsed.get("filename") or parsed.get("name")
    if not raw_file:
        raise ValueError("Model response does not contain file")

    return {
        "file": str(raw_file).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def sanitize_copy_name(filename: str, original_path: str | Path) -> str:
    original_suffix = Path(original_path).suffix.lower() or ".png"
    raw_name = Path(str(filename).strip()).name
    stem = Path(raw_name).stem or "image"
    stem = stem.lower().replace("-", "_").replace(" ", "_")
    cleaned = []
    previous_underscore = False
    for char in stem:
        is_valid = char.isalnum() or char == "_"
        if is_valid:
            if char == "_":
                if previous_underscore:
                    continue
                previous_underscore = True
            else:
                previous_underscore = False
            cleaned.append(char)
        elif not previous_underscore:
            cleaned.append("_")
            previous_underscore = True

    final_stem = "".join(cleaned).strip("_") or "image"
    return f"{final_stem}{original_suffix}"


def dedupe_name(filename: str, used_names: set[str]) -> str:
    candidate = filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 2

    while candidate in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate)
    return candidate
