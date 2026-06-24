from __future__ import annotations

import json
import re


SYSTEM_PROMPT = """你是一个黑白手绘知识信息图的元素分组助手。

你不会直接裁剪图片。
你只根据图片和候选框编号，判断哪些候选框应该合并为一个可复用元素。

请严格返回 JSON，不要返回 Markdown，不要返回解释性正文。"""


USER_RULES = """请根据图片中的编号候选框，对这些候选框进行语义分组。

规则：
1. 保留中文说明文字为独立元素。
2. 语义上共同表达一个场景的图形应该合并，例如小猫 + 绳子 + 被拉的小猫。
3. 循环图应作为一个整体元素。
4. 序号 + 标题框可以合并为标题元素。
5. 等号、简单箭头等通用符号可以建议忽略。
6. 不要把整列内容合并成一个大元素。
7. 如果不确定，宁可少合并，不要过度合并。

请严格返回以下 JSON schema：
{
  "groups": [
    {
      "id": 1,
      "file": "element_name.png",
      "candidate_ids": [1, 2],
      "type": "illustration",
      "keep": true,
      "reason": "why these candidates belong together"
    }
  ],
  "ignored_candidate_ids": [3],
  "notes": "optional notes"
}

约束：
- candidate_ids 必须来自输入候选框 ID。
- 不允许创造不存在的候选框 ID。
- candidate_ids 不允许为空。
- file 使用英文小写、数字和下划线，并以 .png 结尾。
"""


def build_grouping_messages(
    candidates: list[dict],
    annotated_image_data_url: str,
) -> list[dict]:
    payload = {
        "task": "group infographic candidates",
        "rules": {
            "merge_semantic_scenes": True,
            "keep_chinese_labels_as_elements": True,
            "ignore_reusable_symbols": True,
            "do_not_merge_whole_columns": True,
        },
        "candidates": [
            {
                "id": int(candidate["id"]),
                "x": int(candidate["x"]),
                "y": int(candidate["y"]),
                "width": int(candidate["width"]),
                "height": int(candidate["height"]),
            }
            for candidate in candidates
        ],
    }
    text = f"{USER_RULES}\n\n候选框 JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": annotated_image_data_url}},
            ],
        },
    ]


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end >= start:
            stripped = stripped[start : end + 1]

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ValueError("Model response is not a valid JSON object") from error

    if not isinstance(value, dict):
        raise ValueError("Model response is not a valid JSON object")
    return value


def parse_grouping_response(text: str) -> dict:
    return extract_json_object(text)


ALLOWED_TYPES = {"title", "label", "illustration", "symbol", "arrow", "unknown"}


def _clean_group_file(filename: object, fallback_id: int) -> str:
    name = str(filename or "").strip().lower().replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^a-z0-9._]+", "_", name).strip("._")
    if not name:
        name = f"ai_group_{fallback_id:03d}"
    if not name.endswith(".png"):
        name = f"{name}.png"
    return name


def validate_grouping_response(response: dict, candidates: list[dict]) -> dict:
    valid_candidate_ids = {int(candidate["id"]) for candidate in candidates}
    used_candidate_ids: set[int] = set()
    groups = []

    for fallback_id, group in enumerate(response.get("groups", []), start=1):
        if not isinstance(group, dict):
            continue

        candidate_ids = []
        for raw_id in group.get("candidate_ids", []):
            try:
                candidate_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if candidate_id not in valid_candidate_ids:
                continue
            if candidate_id in used_candidate_ids:
                continue
            candidate_ids.append(candidate_id)

        if not candidate_ids:
            continue

        used_candidate_ids.update(candidate_ids)
        group_type = str(group.get("type", "unknown")).strip().lower()
        if group_type not in ALLOWED_TYPES:
            group_type = "unknown"

        groups.append(
            {
                "id": int(group.get("id") or len(groups) + 1),
                "file": _clean_group_file(group.get("file"), fallback_id),
                "candidate_ids": candidate_ids,
                "type": group_type,
                "keep": bool(group.get("keep", True)),
                "reason": str(group.get("reason", "")),
            }
        )

    ignored_candidate_ids = []
    for raw_id in response.get("ignored_candidate_ids", []):
        try:
            candidate_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if candidate_id in valid_candidate_ids and candidate_id not in used_candidate_ids:
            ignored_candidate_ids.append(candidate_id)

    return {
        "groups": groups,
        "ignored_candidate_ids": ignored_candidate_ids,
        "notes": str(response.get("notes", "")),
    }
