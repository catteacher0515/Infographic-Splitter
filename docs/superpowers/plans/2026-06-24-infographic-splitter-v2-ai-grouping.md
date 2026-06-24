# Infographic Splitter V2 AI Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add V2 AI-assisted semantic grouping using `qwen3-vl-flash` while preserving the existing OpenCV-only workflow.

**Architecture:** OpenCV remains responsible for candidate detection and pixel-accurate crop coordinates. V2 adds an annotated candidate preview, sends that preview plus candidate JSON to a Qwen vision model, validates the returned grouping JSON, merges candidate bounding boxes locally, and reuses the current crop/export pipeline.

**Tech Stack:** Python, Gradio, OpenCV, Pillow, OpenAI-compatible client, DashScope/Qwen `qwen3-vl-flash`, pytest.

---

## File Map

- `requirements.txt`: add `openai`.
- `scripts/smoke_qwen_vision.py`: manual API smoke test for `qwen3-vl-flash`; not part of default pytest.
- `annotator.py`: generate annotated candidate preview images and base64 data URLs.
- `vision_client.py`: read Qwen/DashScope environment variables and call OpenAI-compatible chat completions.
- `ai_grouper.py`: build prompts, parse/validate AI JSON, merge candidates into grouped elements.
- `splitter.py`: add support for custom grouped filenames and extra metadata while keeping existing behavior.
- `app.py`: add `AI 语义合并` workflow helper and Gradio UI button.
- `README.md`: document V2 setup, environment variables, and manual smoke test.
- `SPEC_V2.md`: already added; keep as the source spec.
- `tests/test_annotator.py`: tests for annotated preview/base64 generation.
- `tests/test_ai_grouper.py`: parser, validator, bbox merge, and grouped element tests.
- `tests/test_vision_client.py`: environment validation and mocked client tests.
- `tests/test_app_workflow.py`: AI workflow helper tests.

---

### Task 1: Qwen Vision Smoke Test

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/smoke_qwen_vision.py`

- [ ] **Step 1: Add `openai` dependency**

Update `requirements.txt` to:

```text
opencv-python
pillow
gradio
pytest
openai
```

- [ ] **Step 2: Create the manual smoke script**

Create `scripts/smoke_qwen_vision.py`:

```python
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
```

- [ ] **Step 3: Run the smoke test manually**

Run:

```bash
python scripts/smoke_qwen_vision.py
```

Expected when `DASHSCOPE_API_KEY` is set:

```text
model: qwen3-vl-flash
content: ...
```

Expected when `DASHSCOPE_API_KEY` is missing:

```text
DASHSCOPE_API_KEY is not set
```

- [ ] **Step 4: Run existing tests**

Run:

```bash
pytest -v
```

Expected: all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt scripts/smoke_qwen_vision.py
git commit -m "chore: add qwen vision smoke test"
```

---

### Task 2: Annotated Candidate Preview

**Files:**
- Create: `annotator.py`
- Create: `tests/test_annotator.py`

- [ ] **Step 1: Write failing tests for annotated preview**

Create `tests/test_annotator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_annotator.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'annotator'`.

- [ ] **Step 3: Implement `annotator.py`**

Create `annotator.py`:

```python
from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BOX_COLORS = [
    (255, 80, 80),
    (80, 160, 255),
    (80, 200, 120),
    (255, 180, 60),
    (180, 100, 255),
]


def _scale_image(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.convert("RGB"), 1.0

    scale = max_side / float(longest)
    resized = image.convert("RGB").resize(
        (int(width * scale), int(height * scale)),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def save_annotated_candidates(
    image: Image.Image,
    candidates: list[dict],
    output_path: str | Path,
    max_side: int = 1600,
) -> Path:
    annotated, scale = _scale_image(image, max_side)
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for index, candidate in enumerate(candidates):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1 = int(candidate["x"] * scale)
        y1 = int(candidate["y"] * scale)
        x2 = int((candidate["x"] + candidate["width"]) * scale)
        y2 = int((candidate["y"] + candidate["height"]) * scale)
        label = str(candidate["id"])

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label_box = draw.textbbox((x1, y1), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        draw.rectangle(
            (x1, y1, x1 + label_width + 8, y1 + label_height + 6),
            fill=color,
        )
        draw.text((x1 + 4, y1 + 3), label, fill=(0, 0, 0), font=font)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output)
    return output


def image_to_data_url(image_path: str | Path) -> str:
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_annotator.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Run full tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add annotator.py tests/test_annotator.py
git commit -m "feat: annotate candidate boxes for vision models"
```

---

### Task 3: AI Prompt And JSON Schema

**Files:**
- Create: `ai_grouper.py`
- Create: `tests/test_ai_grouper.py`

- [ ] **Step 1: Write failing tests for prompt and JSON extraction**

Create `tests/test_ai_grouper.py`:

```python
import pytest

from ai_grouper import (
    build_grouping_messages,
    extract_json_object,
    parse_grouping_response,
)


def sample_candidates():
    return [
        {"id": 1, "x": 10, "y": 20, "width": 100, "height": 60},
        {"id": 2, "x": 140, "y": 24, "width": 80, "height": 56},
    ]


def test_build_grouping_messages_includes_rules_and_candidates():
    messages = build_grouping_messages(
        candidates=sample_candidates(),
        annotated_image_data_url="data:image/png;base64,abc",
    )

    assert messages[0]["role"] == "system"
    assert "严格返回 JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "text"
    assert "candidate_ids" in user_content[0]["text"]
    assert '"id": 1' in user_content[0]["text"]
    assert user_content[1]["type"] == "image_url"


def test_extract_json_object_handles_markdown_wrapper():
    text = '```json\\n{"groups": [], "ignored_candidate_ids": []}\\n```'

    assert extract_json_object(text) == {"groups": [], "ignored_candidate_ids": []}


def test_parse_grouping_response_requires_json_object():
    with pytest.raises(ValueError, match="valid JSON object"):
        parse_grouping_response("not json")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_ai_grouper.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'ai_grouper'`.

- [ ] **Step 3: Implement prompt and parser foundation**

Create `ai_grouper.py`:

```python
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
    text = f"{USER_RULES}\\n\\n候选框 JSON：\\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
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
    fenced = re.search(r"```(?:json)?\\s*(\\{.*\\})\\s*```", stripped, re.DOTALL)
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
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_ai_grouper.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ai_grouper.py tests/test_ai_grouper.py
git commit -m "feat: build ai grouping prompt"
```

---

### Task 4: AI Grouping Parser And Validator

**Files:**
- Modify: `ai_grouper.py`
- Modify: `tests/test_ai_grouper.py`

- [ ] **Step 1: Add failing validation tests**

Append to `tests/test_ai_grouper.py`:

```python
from ai_grouper import validate_grouping_response


def test_validate_grouping_response_keeps_valid_groups():
    response = {
        "groups": [
            {
                "id": 1,
                "file": "loop_cycle.png",
                "candidate_ids": [1, 2],
                "type": "illustration",
                "keep": True,
                "reason": "same loop diagram",
            }
        ],
        "ignored_candidate_ids": [99],
    }

    validated = validate_grouping_response(response, sample_candidates())

    assert validated["groups"][0]["candidate_ids"] == [1, 2]
    assert validated["groups"][0]["file"] == "loop_cycle.png"
    assert validated["groups"][0]["type"] == "illustration"
    assert validated["ignored_candidate_ids"] == []


def test_validate_grouping_response_removes_duplicate_candidate_ids():
    response = {
        "groups": [
            {"id": 1, "file": "first.png", "candidate_ids": [1, 2], "keep": True},
            {"id": 2, "file": "second.png", "candidate_ids": [2], "keep": True},
        ]
    }

    validated = validate_grouping_response(response, sample_candidates())

    assert len(validated["groups"]) == 1
    assert validated["groups"][0]["candidate_ids"] == [1, 2]


def test_validate_grouping_response_skips_empty_or_invalid_groups():
    response = {
        "groups": [
            {"id": 1, "file": "bad.png", "candidate_ids": [999], "keep": True},
            {"id": 2, "file": "empty.png", "candidate_ids": [], "keep": True},
        ]
    }

    validated = validate_grouping_response(response, sample_candidates())

    assert validated["groups"] == []
```

- [ ] **Step 2: Run validation tests to verify failure**

Run:

```bash
pytest tests/test_ai_grouper.py::test_validate_grouping_response_keeps_valid_groups -v
```

Expected: fail with `ImportError` or missing function.

- [ ] **Step 3: Implement validator**

Add to `ai_grouper.py`:

```python
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
```

- [ ] **Step 4: Run AI grouper tests**

Run:

```bash
pytest tests/test_ai_grouper.py -v
```

Expected: all AI grouper tests pass.

- [ ] **Step 5: Commit**

```bash
git add ai_grouper.py tests/test_ai_grouper.py
git commit -m "feat: validate ai grouping responses"
```

---

### Task 5: Group BBox Merge And Cropping

**Files:**
- Modify: `splitter.py`
- Modify: `ai_grouper.py`
- Modify: `tests/test_ai_grouper.py`
- Modify: `tests/test_splitter.py`

- [ ] **Step 1: Add failing grouped crop tests**

Append to `tests/test_splitter.py`:

```python
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
```

Append to `tests/test_ai_grouper.py`:

```python
from ai_grouper import build_grouped_boxes


def test_build_grouped_boxes_merges_candidate_bounds():
    candidates = [
        {"id": 1, "x": 10, "y": 20, "width": 50, "height": 40},
        {"id": 2, "x": 70, "y": 30, "width": 30, "height": 60},
    ]
    grouping = {
        "groups": [
            {
                "file": "merged.png",
                "candidate_ids": [1, 2],
                "type": "illustration",
                "keep": True,
                "reason": "one scene",
            }
        ]
    }

    boxes = build_grouped_boxes(grouping, candidates)

    assert boxes == [
        {
            "x": 10,
            "y": 20,
            "width": 90,
            "height": 70,
            "file": "merged.png",
            "type": "illustration",
            "selected": True,
            "source_candidate_ids": [1, 2],
            "reason": "one scene",
        }
    ]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_splitter.py::test_create_elements_uses_custom_file_and_metadata tests/test_ai_grouper.py::test_build_grouped_boxes_merges_candidate_bounds -v
```

Expected: fail because metadata/custom grouped boxes are not supported yet.

- [ ] **Step 3: Update `splitter.py`**

Modify `create_elements` loop in `splitter.py` to preserve optional metadata:

```python
    for index, box in enumerate(boxes, start=1):
        filename = str(box.get("file") or f"element_{index:03d}.png")
        preview_path = assets_dir / filename
        _crop_image(image, box).save(preview_path)

        element = {
            "id": index,
            "file": filename,
            "x": int(box["x"]),
            "y": int(box["y"]),
            "width": int(box["width"]),
            "height": int(box["height"]),
            "selected": bool(box.get("selected", True)),
            "preview_path": str(preview_path),
        }
        for key in ("type", "source_candidate_ids", "reason"):
            if key in box:
                element[key] = box[key]
        elements.append(element)
```

- [ ] **Step 4: Implement `build_grouped_boxes`**

Add to `ai_grouper.py`:

```python
def build_grouped_boxes(grouping: dict, candidates: list[dict]) -> list[dict]:
    candidates_by_id = {int(candidate["id"]): candidate for candidate in candidates}
    boxes = []

    for group in grouping.get("groups", []):
        source_candidates = [
            candidates_by_id[candidate_id]
            for candidate_id in group["candidate_ids"]
            if candidate_id in candidates_by_id
        ]
        if not source_candidates:
            continue

        x1 = min(int(candidate["x"]) for candidate in source_candidates)
        y1 = min(int(candidate["y"]) for candidate in source_candidates)
        x2 = max(int(candidate["x"]) + int(candidate["width"]) for candidate in source_candidates)
        y2 = max(int(candidate["y"]) + int(candidate["height"]) for candidate in source_candidates)

        boxes.append(
            {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "file": group["file"],
                "type": group.get("type", "unknown"),
                "selected": bool(group.get("keep", True)),
                "source_candidate_ids": list(group["candidate_ids"]),
                "reason": str(group.get("reason", "")),
            }
        )

    return boxes
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_splitter.py tests/test_ai_grouper.py -v
```

Expected: all splitter and AI grouper tests pass.

- [ ] **Step 6: Run full tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add splitter.py ai_grouper.py tests/test_splitter.py tests/test_ai_grouper.py
git commit -m "feat: build grouped crops from ai response"
```

---

### Task 6: Vision Client

**Files:**
- Create: `vision_client.py`
- Create: `tests/test_vision_client.py`

- [ ] **Step 1: Write failing tests for configuration and mocked calls**

Create `tests/test_vision_client.py`:

```python
import pytest

from vision_client import QwenVisionClient, VisionConfig, load_vision_config


def test_load_vision_config_reads_defaults(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("QWEN_MODEL", raising=False)

    config = load_vision_config()

    assert config.api_key == "test-key"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.model == "qwen3-vl-flash"


def test_load_vision_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        load_vision_config()


def test_qwen_vision_client_returns_message_content():
    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "qwen3-vl-flash"
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Message", (), {"content": "{\\"groups\\": []}"})()},
                        )()
                    ]
                },
            )()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = QwenVisionClient(
        VisionConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="qwen3-vl-flash",
        ),
        client=FakeClient(),
    )

    content = client.complete([{"role": "user", "content": "hello"}])

    assert content == '{"groups": []}'
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_vision_client.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'vision_client'`.

- [ ] **Step 3: Implement `vision_client.py`**

Create `vision_client.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3-vl-flash"


@dataclass(frozen=True)
class VisionConfig:
    api_key: str
    base_url: str = DEFAULT_QWEN_BASE_URL
    model: str = DEFAULT_QWEN_MODEL


def load_vision_config() -> VisionConfig:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is not set")

    return VisionConfig(
        api_key=api_key,
        base_url=os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
        model=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
    )


class QwenVisionClient:
    def __init__(self, config: VisionConfig | None = None, client: Any | None = None):
        self.config = config or load_vision_config()
        self.client = client or OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def complete(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=0,
        )
        return str(response.choices[0].message.content)
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_vision_client.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Run full tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add vision_client.py tests/test_vision_client.py
git commit -m "feat: add qwen vision client"
```

---

### Task 7: Gradio AI Semantic Merge Workflow

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app_workflow.py`

- [ ] **Step 1: Add failing app workflow tests**

Append to `tests/test_app_workflow.py`:

```python
from app import elements_to_rows, run_ai_grouping


def test_elements_to_rows_includes_ai_metadata():
    rows = elements_to_rows(
        [
            {
                "selected": True,
                "file": "loop_cycle.png",
                "x": 10,
                "y": 20,
                "width": 90,
                "height": 70,
                "type": "illustration",
                "source_candidate_ids": [1, 2],
                "reason": "same loop",
            }
        ]
    )

    assert rows[0] == [
        True,
        "loop_cycle.png",
        10,
        20,
        90,
        70,
        "illustration",
        "1,2",
        "same loop",
    ]


def test_run_ai_grouping_returns_error_without_elements(tmp_path: Path):
    image = Image.new("RGB", (100, 100), "white")

    gallery, rows, elements, status = run_ai_grouping(
        image=image,
        elements=[],
        output_root=tmp_path,
    )

    assert gallery == []
    assert rows == []
    assert elements == []
    assert "先点击开始拆分" in status
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_app_workflow.py::test_elements_to_rows_includes_ai_metadata tests/test_app_workflow.py::test_run_ai_grouping_returns_error_without_elements -v
```

Expected: fail because AI metadata rows and `run_ai_grouping` are missing.

- [ ] **Step 3: Modify `elements_to_rows` and `rows_to_elements`**

Update `elements_to_rows` in `app.py`:

```python
def elements_to_rows(elements: list[dict]) -> list[list]:
    return [
        [
            bool(element.get("selected", True)),
            element["file"],
            element["x"],
            element["y"],
            element["width"],
            element["height"],
            element.get("type", ""),
            ",".join(str(item) for item in element.get("source_candidate_ids", [])),
            element.get("reason", ""),
        ]
        for element in elements
    ]
```

Update `rows_to_elements` in `app.py`:

```python
def rows_to_elements(rows, elements: list[dict]) -> list[dict]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()

    updated = []
    for row, element in zip(rows, elements):
        selected, filename, *_ = row
        copied = dict(element)
        copied["selected"] = bool(selected)
        copied["file"] = str(filename)
        updated.append(copied)
    return updated
```

- [ ] **Step 4: Implement `run_ai_grouping`**

Add imports to `app.py`:

```python
from ai_grouper import (
    build_grouped_boxes,
    build_grouping_messages,
    parse_grouping_response,
    validate_grouping_response,
)
from annotator import image_to_data_url, save_annotated_candidates
from vision_client import QwenVisionClient
```

Add function:

```python
def run_ai_grouping(
    image: Image.Image,
    elements: list[dict],
    output_root: str | Path = OUTPUT_ROOT,
    vision_client: QwenVisionClient | None = None,
) -> tuple[list[str], list[list], list[dict], str]:
    if image is None:
        return [], [], [], "请先上传图片"
    if not elements:
        return [], [], [], "请先点击开始拆分"

    session_dir = build_session_output_dir(output_root)
    annotated_path = save_annotated_candidates(
        image,
        elements,
        session_dir / "annotated_candidates.png",
    )
    messages = build_grouping_messages(
        candidates=elements,
        annotated_image_data_url=image_to_data_url(annotated_path),
    )

    try:
        client = vision_client or QwenVisionClient()
        raw_response = client.complete(messages)
        parsed = parse_grouping_response(raw_response)
        grouping = validate_grouping_response(parsed, elements)
        grouped_boxes = build_grouped_boxes(grouping, elements)
        grouped_elements = create_elements(image, grouped_boxes, session_dir)
    except Exception as error:
        return (
            [element["preview_path"] for element in elements],
            elements_to_rows(elements),
            elements,
            f"AI 语义合并失败：{error}",
        )

    return (
        [element["preview_path"] for element in grouped_elements],
        elements_to_rows(grouped_elements),
        grouped_elements,
        "AI 语义合并完成",
    )
```

- [ ] **Step 5: Update Gradio UI**

Modify `build_app()`:

1. Add state/status:

```python
        state_elements = gr.State([])
        ai_status = gr.Markdown("")
```

2. Add button beside split/export controls:

```python
                ai_button = gr.Button("AI 语义合并")
```

3. Update table headers and datatypes:

```python
                    headers=[
                        "selected",
                        "file",
                        "x",
                        "y",
                        "width",
                        "height",
                        "type",
                        "source_candidate_ids",
                        "reason",
                    ],
                    datatype=[
                        "bool",
                        "str",
                        "number",
                        "number",
                        "number",
                        "number",
                        "str",
                        "str",
                        "str",
                    ],
```

4. Wire click:

```python
        ai_button.click(
            fn=run_ai_grouping,
            inputs=[image_input, state_elements],
            outputs=[gallery, table, state_elements, ai_status],
        )
```

- [ ] **Step 6: Run app workflow tests**

Run:

```bash
pytest tests/test_app_workflow.py -v
```

Expected: app workflow tests pass.

- [ ] **Step 7: Run full tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 8: Manual local UI check**

Run:

```bash
python app.py
```

Open the Gradio URL, upload the sample infographic, click:

```text
开始拆分
AI 语义合并
导出 ZIP
```

Expected:

- OpenCV results appear after `开始拆分`.
- AI grouped results replace gallery after `AI 语义合并`.
- Status says `AI 语义合并完成` or a readable failure message.
- ZIP export still works.

- [ ] **Step 9: Commit**

```bash
git add app.py tests/test_app_workflow.py
git commit -m "feat: add ai semantic merge workflow"
```

---

### Task 8: Verification, README, SPEC Commit, Push

**Files:**
- Modify: `README.md`
- Add if not committed yet: `SPEC_V2.md`
- Add if not committed yet: `docs/superpowers/plans/2026-06-24-infographic-splitter-v2-ai-grouping.md`

- [ ] **Step 1: Update README**

Append to `README.md`:

```markdown
## V2 AI Semantic Grouping

V2 can optionally use Qwen vision models through Alibaba Cloud Model Studio / DashScope.

Set environment variables before launching the app:

```bash
export DASHSCOPE_API_KEY="your-api-key"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen3-vl-flash"
```

Run a manual smoke test:

```bash
python scripts/smoke_qwen_vision.py
```

Then start the app:

```bash
python app.py
```

Workflow:

```text
Upload image
-> Start splitting
-> AI semantic merge
-> Review grouped elements
-> Export ZIP
```

The AI step sends the annotated candidate image and candidate box JSON to the configured vision model. API keys are read only from environment variables and are not written to exports.
```

- [ ] **Step 2: Run full automated verification**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Run manual Qwen smoke test if API key is available**

Run:

```bash
python scripts/smoke_qwen_vision.py
```

Expected:

```text
model: qwen3-vl-flash
content: ...
```

If `DASHSCOPE_API_KEY` is not set, expected:

```text
DASHSCOPE_API_KEY is not set
```

This is not a failure for automated verification.

- [ ] **Step 4: Check git diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only V2-related files changed or added.

- [ ] **Step 5: Commit docs if needed**

If `SPEC_V2.md`, the plan, or `README.md` are uncommitted:

```bash
git add README.md SPEC_V2.md docs/superpowers/plans/2026-06-24-infographic-splitter-v2-ai-grouping.md
git commit -m "docs: add v2 ai grouping plan"
```

- [ ] **Step 6: Push**

Run:

```bash
git push
```

Expected:

```text
main -> main
```

- [ ] **Step 7: Final status check**

Run:

```bash
git status --short --branch
```

Expected:

```text
## main...origin/main
```

