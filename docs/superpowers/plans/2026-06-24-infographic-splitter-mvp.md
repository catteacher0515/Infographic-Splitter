# Infographic Splitter MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Gradio app that uploads a black-and-white infographic, detects visual elements with OpenCV, previews candidates, lets the user select and rename outputs, and exports a ZIP with cropped PNG assets and `manifest.json`.

**Architecture:** Keep image processing, crop metadata, ZIP export, and UI orchestration in separate small modules. Core behavior is tested without Gradio first; the UI calls the same pure functions used by tests. OpenCV detection produces candidate boxes, not semantic labels.

**Tech Stack:** Python, OpenCV (`opencv-python`), Pillow, Gradio, pytest, Python standard library (`json`, `zipfile`, `pathlib`, `tempfile`, `uuid`, `shutil`, `re`).

---

## File Structure

- Create: `requirements.txt`
  - Runtime dependencies for the MVP.
- Create: `detector.py`
  - Image preprocessing, morphology, connected component or contour detection, bounding-box filtering, padding, and sorting.
- Create: `splitter.py`
  - Crop generation, default naming, and element metadata creation.
- Create: `exporter.py`
  - Filename sanitation, selected-element filtering, ZIP writing, and manifest generation.
- Create: `app.py`
  - Gradio UI and workflow orchestration.
- Create: `tests/test_detector.py`
  - Synthetic image tests for detection, sorting, padding, and noise filtering.
- Create: `tests/test_splitter.py`
  - Crop generation and metadata tests.
- Create: `tests/test_exporter.py`
  - ZIP structure, manifest, rename, duplicate-name, and selected-only export tests.
- Create: `tests/test_app_workflow.py`
  - Thin workflow-level tests that verify UI helper functions connect splitting and export behavior.
- Create: `.gitignore`
  - Ignore generated output, caches, virtual environments, and local runtime files.
- Create directories at runtime or repo level:
  - `assets/`
  - `output/`

## Shared Data Contracts

Use plain dictionaries to keep the MVP simple.

`detector.py` returns boxes:

```python
{
    "x": 12,
    "y": 24,
    "width": 120,
    "height": 80,
    "area": 9600,
}
```

`splitter.py` returns elements:

```python
{
    "id": 1,
    "file": "element_001.png",
    "x": 12,
    "y": 24,
    "width": 120,
    "height": 80,
    "selected": True,
    "preview_path": "output/session-id/assets/element_001.png",
}
```

`exporter.py` writes manifest entries:

```python
{
    "id": 1,
    "file": "element_001.png",
    "x": 12,
    "y": 24,
    "width": 120,
    "height": 80,
}
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create directories: `assets/`, `output/`, `tests/`

- [ ] **Step 1: Create dependency file**

Create `requirements.txt`:

```text
opencv-python
pillow
gradio
pytest
```

- [ ] **Step 2: Create ignore rules**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
env/
output/
.DS_Store
```

- [ ] **Step 3: Create directories**

Run:

```bash
mkdir -p assets output tests
```

Expected: directories exist and are empty except for future generated files.

- [ ] **Step 4: Verify current repository state**

Run:

```bash
git status --short
```

Expected: `requirements.txt`, `.gitignore`, and `tests/` are untracked or staged depending on local workflow. `output/` should not appear after `.gitignore` is created.

- [ ] **Step 5: Commit scaffold**

```bash
git add requirements.txt .gitignore assets tests
git commit -m "chore: scaffold MVP project"
```

Expected: commit succeeds.

---

### Task 2: Detector Core

**Files:**
- Create: `detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write failing detector tests**

Create `tests/test_detector.py`:

```python
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
    assert [box["x"] for box in boxes] == sorted([box["x"] for box in boxes[:2]]) + [boxes[2]["x"]]
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

    assert [(box["x"], box["y"]) for box in sorted_boxes] == [(20, 25), (200, 20), (20, 120)]
```

- [ ] **Step 2: Run detector tests and verify failure**

Run:

```bash
pytest tests/test_detector.py -v
```

Expected: fails because `detector.py` does not exist or functions are missing.

- [ ] **Step 3: Implement detector**

Create `detector.py`:

```python
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def _to_cv_gray(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _threshold_and_invert(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _merge_strokes(binary: np.ndarray, merge_gap: int) -> np.ndarray:
    gap = max(1, int(merge_gap))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _clamp_box(x: int, y: int, w: int, h: int, image_width: int, image_height: int, padding: int) -> dict:
    pad = max(0, int(padding))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_width, x + w + pad)
    y2 = min(image_height, y + h + pad)
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return {
        "x": int(x1),
        "y": int(y1),
        "width": int(width),
        "height": int(height),
        "area": int(width * height),
    }


def sort_boxes(boxes: Iterable[dict]) -> list[dict]:
    boxes = list(boxes)
    if not boxes:
        return []

    median_height = float(np.median([box["height"] for box in boxes]))
    row_height = max(1.0, median_height * 0.75)

    def key(box: dict) -> tuple[int, int]:
        row = int(round(box["y"] / row_height))
        return row, int(box["x"])

    return sorted(boxes, key=key)


def detect_elements(
    image: Image.Image,
    min_area: int = 500,
    merge_gap: int = 8,
    padding: int = 10,
) -> list[dict]:
    gray = _to_cv_gray(image)
    binary = _threshold_and_invert(gray)
    merged = _merge_strokes(binary, merge_gap)
    image_height, image_width = gray.shape[:2]

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        raw_area = w * h
        if raw_area < int(min_area):
            continue
        boxes.append(_clamp_box(x, y, w, h, image_width, image_height, padding))

    return sort_boxes(boxes)
```

- [ ] **Step 4: Run detector tests and verify pass**

Run:

```bash
pytest tests/test_detector.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit detector**

```bash
git add detector.py tests/test_detector.py
git commit -m "feat: add OpenCV element detector"
```

Expected: commit succeeds.

---

### Task 3: Splitter Core

**Files:**
- Create: `splitter.py`
- Create: `tests/test_splitter.py`

- [ ] **Step 1: Write failing splitter tests**

Create `tests/test_splitter.py`:

```python
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


def test_create_elements_uses_sorted_box_order_for_names(tmp_path: Path):
    image = Image.new("RGB", (300, 200), "white")
    boxes = [
        {"x": 100, "y": 20, "width": 40, "height": 30, "area": 1200},
        {"x": 20, "y": 20, "width": 40, "height": 30, "area": 1200},
    ]

    elements = create_elements(image, boxes, tmp_path)

    assert [element["file"] for element in elements] == ["element_001.png", "element_002.png"]
    assert [element["x"] for element in elements] == [100, 20]
```

- [ ] **Step 2: Run splitter tests and verify failure**

Run:

```bash
pytest tests/test_splitter.py -v
```

Expected: fails because `splitter.py` does not exist or `create_elements` is missing.

- [ ] **Step 3: Implement splitter**

Create `splitter.py`:

```python
from __future__ import annotations

from pathlib import Path

from PIL import Image


def _crop_image(image: Image.Image, box: dict) -> Image.Image:
    x = int(box["x"])
    y = int(box["y"])
    width = int(box["width"])
    height = int(box["height"])
    return image.crop((x, y, x + width, y + height))


def create_elements(image: Image.Image, boxes: list[dict], output_dir: str | Path) -> list[dict]:
    assets_dir = Path(output_dir) / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    elements = []
    for index, box in enumerate(boxes, start=1):
        filename = f"element_{index:03d}.png"
        crop = _crop_image(image, box)
        preview_path = assets_dir / filename
        crop.save(preview_path)

        elements.append(
            {
                "id": index,
                "file": filename,
                "x": int(box["x"]),
                "y": int(box["y"]),
                "width": int(box["width"]),
                "height": int(box["height"]),
                "selected": True,
                "preview_path": str(preview_path),
            }
        )

    return elements
```

- [ ] **Step 4: Run splitter tests and verify pass**

Run:

```bash
pytest tests/test_splitter.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit splitter**

```bash
git add splitter.py tests/test_splitter.py
git commit -m "feat: add crop splitter"
```

Expected: commit succeeds.

---

### Task 4: Exporter Core

**Files:**
- Create: `exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write failing exporter tests**

Create `tests/test_exporter.py`:

```python
import json
import zipfile
from pathlib import Path

from PIL import Image

from exporter import export_zip, sanitize_filename


def make_preview(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "white").save(path)


def test_sanitize_filename_adds_png_and_removes_path_parts():
    assert sanitize_filename("../bad/name") == "name.png"
    assert sanitize_filename("element 1") == "element_1.png"
    assert sanitize_filename("ok.png") == "ok.png"


def test_export_zip_writes_selected_assets_and_manifest(tmp_path: Path):
    preview = tmp_path / "source" / "element_001.png"
    make_preview(preview)
    elements = [
        {
            "id": 1,
            "file": "renamed.png",
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40,
            "selected": True,
            "preview_path": str(preview),
        },
        {
            "id": 2,
            "file": "skip.png",
            "x": 0,
            "y": 0,
            "width": 10,
            "height": 10,
            "selected": False,
            "preview_path": str(preview),
        },
    ]

    zip_path = export_zip(elements, tmp_path / "export")

    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(archive.namelist())
        assert names == ["assets/renamed.png", "manifest.json"]
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest == {
        "elements": [
            {
                "id": 1,
                "file": "renamed.png",
                "x": 10,
                "y": 20,
                "width": 30,
                "height": 40,
            }
        ]
    }


def test_export_zip_deduplicates_filenames(tmp_path: Path):
    preview = tmp_path / "source" / "element_001.png"
    make_preview(preview)
    elements = [
        {
            "id": 1,
            "file": "same.png",
            "x": 0,
            "y": 0,
            "width": 20,
            "height": 20,
            "selected": True,
            "preview_path": str(preview),
        },
        {
            "id": 2,
            "file": "same.png",
            "x": 30,
            "y": 0,
            "width": 20,
            "height": 20,
            "selected": True,
            "preview_path": str(preview),
        },
    ]

    zip_path = export_zip(elements, tmp_path / "export")

    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith("assets/"))

    assert names == ["assets/same.png", "assets/same_2.png"]
```

- [ ] **Step 2: Run exporter tests and verify failure**

Run:

```bash
pytest tests/test_exporter.py -v
```

Expected: fails because `exporter.py` does not exist or functions are missing.

- [ ] **Step 3: Implement exporter**

Create `exporter.py`:

```python
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    name = Path(str(filename).strip()).name
    if not name:
        name = "element"
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not stem:
        stem = "element"
    if suffix != ".png":
        suffix = ".png"
    return f"{stem}{suffix}"


def _dedupe_filename(filename: str, used: set[str]) -> str:
    candidate = filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 2
    while candidate in used:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _manifest_entry(element: dict, filename: str) -> dict:
    return {
        "id": int(element["id"]),
        "file": filename,
        "x": int(element["x"]),
        "y": int(element["y"]),
        "width": int(element["width"]),
        "height": int(element["height"]),
    }


def export_zip(elements: list[dict], output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    export_assets_dir = output_dir / "assets"
    export_assets_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    manifest = {"elements": []}

    for element in elements:
        if not element.get("selected", True):
            continue

        filename = _dedupe_filename(sanitize_filename(element["file"]), used_names)
        source = Path(element["preview_path"])
        destination = export_assets_dir / filename
        shutil.copyfile(source, destination)
        manifest["elements"].append(_manifest_entry(element, filename))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = output_dir / "export.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, "manifest.json")
        for asset_path in sorted(export_assets_dir.glob("*.png")):
            archive.write(asset_path, f"assets/{asset_path.name}")

    return str(zip_path)
```

- [ ] **Step 4: Run exporter tests and verify pass**

Run:

```bash
pytest tests/test_exporter.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit exporter**

```bash
git add exporter.py tests/test_exporter.py
git commit -m "feat: add ZIP exporter"
```

Expected: commit succeeds.

---

### Task 5: Workflow Orchestration Helpers

**Files:**
- Create: `app.py`
- Create: `tests/test_app_workflow.py`

- [ ] **Step 1: Write failing workflow tests**

Create `tests/test_app_workflow.py`:

```python
from pathlib import Path

from PIL import Image, ImageDraw

from app import build_session_output_dir, run_split


def test_build_session_output_dir_creates_unique_session(tmp_path: Path):
    first = build_session_output_dir(tmp_path)
    second = build_session_output_dir(tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_run_split_returns_gallery_rows_and_elements(tmp_path: Path):
    image = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 100, 80), outline="black", width=4)

    gallery, rows, elements = run_split(image, min_area=200, merge_gap=5, padding=0, output_root=tmp_path)

    assert len(gallery) == 1
    assert len(rows) == 1
    assert len(elements) == 1
    assert rows[0][0] is True
    assert rows[0][1] == "element_001.png"
```

- [ ] **Step 2: Run workflow tests and verify failure**

Run:

```bash
pytest tests/test_app_workflow.py -v
```

Expected: fails because `app.py` does not exist or functions are missing.

- [ ] **Step 3: Implement UI helper functions in `app.py`**

Create the non-Gradio helper portion of `app.py`:

```python
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image

from detector import detect_elements
from exporter import export_zip
from splitter import create_elements


OUTPUT_ROOT = Path("output")


def build_session_output_dir(output_root: str | Path = OUTPUT_ROOT) -> Path:
    session_dir = Path(output_root) / f"session-{uuid4().hex[:12]}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def elements_to_rows(elements: list[dict]) -> list[list]:
    return [
        [
            bool(element.get("selected", True)),
            element["file"],
            element["x"],
            element["y"],
            element["width"],
            element["height"],
        ]
        for element in elements
    ]


def rows_to_elements(rows: list[list], elements: list[dict]) -> list[dict]:
    updated = []
    for row, element in zip(rows, elements):
        selected, filename, *_ = row
        copied = dict(element)
        copied["selected"] = bool(selected)
        copied["file"] = str(filename)
        updated.append(copied)
    return updated


def run_split(
    image: Image.Image,
    min_area: int = 500,
    merge_gap: int = 8,
    padding: int = 10,
    output_root: str | Path = OUTPUT_ROOT,
) -> tuple[list[str], list[list], list[dict]]:
    if image is None:
        return [], [], []

    session_dir = build_session_output_dir(output_root)
    boxes = detect_elements(image, min_area=min_area, merge_gap=merge_gap, padding=padding)
    elements = create_elements(image, boxes, session_dir)
    gallery = [element["preview_path"] for element in elements]
    return gallery, elements_to_rows(elements), elements


def run_export(rows: list[list], elements: list[dict], output_root: str | Path = OUTPUT_ROOT) -> str | None:
    if not elements:
        return None

    session_dir = build_session_output_dir(output_root)
    updated_elements = rows_to_elements(rows, elements)
    return export_zip(updated_elements, session_dir)
```

- [ ] **Step 4: Run workflow tests and verify pass**

Run:

```bash
pytest tests/test_app_workflow.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit workflow helpers**

```bash
git add app.py tests/test_app_workflow.py
git commit -m "feat: add app workflow helpers"
```

Expected: commit succeeds.

---

### Task 6: Gradio UI

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add Gradio UI to `app.py`**

Append this UI code below the helper functions in `app.py`:

```python
def build_app():
    import gradio as gr

    with gr.Blocks(title="Infographic Splitter") as demo:
        gr.Markdown("# Infographic Splitter")

        state_elements = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="上传图片")
                min_area = gr.Number(value=500, precision=0, label="min_area")
                merge_gap = gr.Slider(1, 40, value=8, step=1, label="merge_gap")
                padding = gr.Slider(0, 40, value=10, step=1, label="padding")
                split_button = gr.Button("开始拆分", variant="primary")
                export_button = gr.Button("导出 ZIP")
                zip_output = gr.File(label="下载 ZIP")

            with gr.Column(scale=2):
                gallery = gr.Gallery(label="拆分结果", columns=4, height=360)
                table = gr.Dataframe(
                    headers=["selected", "file", "x", "y", "width", "height"],
                    datatype=["bool", "str", "number", "number", "number", "number"],
                    interactive=True,
                    label="元素列表",
                )

        split_button.click(
            fn=run_split,
            inputs=[image_input, min_area, merge_gap, padding],
            outputs=[gallery, table, state_elements],
        )
        export_button.click(
            fn=run_export,
            inputs=[table, state_elements],
            outputs=[zip_output],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
```

- [ ] **Step 2: Run all tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Smoke test the app imports**

Run:

```bash
python -c "from app import build_app; app = build_app(); print(type(app).__name__)"
```

Expected: prints `Blocks`.

- [ ] **Step 4: Commit UI**

```bash
git add app.py
git commit -m "feat: add Gradio interface"
```

Expected: commit succeeds.

---

### Task 7: End-to-End Verification

**Files:**
- No required source changes unless verification finds a defect.

- [ ] **Step 1: Run complete test suite**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run synthetic export smoke script**

Run:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
from app import run_split, run_export

root = Path("output/smoke")
image = Image.new("RGB", (420, 240), "white")
draw = ImageDraw.Draw(image)
draw.rectangle((20, 20, 120, 80), outline="black", width=4)
draw.rectangle((220, 30, 340, 90), outline="black", width=4)

gallery, rows, elements = run_split(image, min_area=200, merge_gap=5, padding=10, output_root=root)
zip_path = run_export(rows, elements, output_root=root)
print(len(gallery), zip_path)
PY
```

Expected: prints `2` and a path ending in `export.zip`.

- [ ] **Step 3: Inspect generated ZIP**

Run:

```bash
python - <<'PY'
import json
import zipfile
from pathlib import Path

zips = sorted(Path("output/smoke").glob("session-*/export.zip"))
assert zips, "No smoke ZIP found"
with zipfile.ZipFile(zips[-1]) as archive:
    print(sorted(archive.namelist()))
    manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    print(len(manifest["elements"]))
PY
```

Expected: output includes `manifest.json`, at least two files under `assets/`, and manifest element count `2`.

- [ ] **Step 4: Start local app**

Run:

```bash
python app.py
```

Expected: Gradio starts and prints a local URL, usually `http://127.0.0.1:7860`.

- [ ] **Step 5: Manual happy-path check**

Use the browser URL from Step 4:

```text
Upload a supported black-and-white image
-> Click 开始拆分
-> Confirm candidate crops appear
-> Unselect one row
-> Rename one file
-> Click 导出 ZIP
-> Download ZIP
-> Confirm ZIP contains assets/ and manifest.json
```

Expected: happy path succeeds.

- [ ] **Step 6: Commit verification-only adjustments if needed**

If fixes were needed:

```bash
git add detector.py splitter.py exporter.py app.py tests
git commit -m "fix: address MVP smoke test issues"
```

If no fixes were needed, skip this commit.

---

### Task 8: Documentation and Final Push

**Files:**
- Create: `README.md`
- Modify: no source files unless documentation reveals a mismatch.

- [ ] **Step 1: Create README**

Create `README.md`:

```markdown
# Infographic Splitter

Local MVP for splitting black-and-white hand-drawn knowledge infographics into reusable visual elements.

## Scope

- Uses OpenCV, Pillow, and Gradio.
- No AI models, SAM, OCR, or large-model dependency.
- Targets white or light backgrounds with black line art and clear spacing.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open the local Gradio URL printed in the terminal.

## Workflow

```text
Upload image
-> Start splitting
-> Preview candidates
-> Select elements
-> Rename files
-> Export ZIP
```

## Output

```text
export.zip
|-- assets/
`-- manifest.json
```

See `SPEC.md` for the full MVP specification.
```

- [ ] **Step 2: Run final tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit README**

```bash
git add README.md
git commit -m "docs: add usage instructions"
```

Expected: commit succeeds.

- [ ] **Step 4: Push to GitHub**

Run:

```bash
git push
```

Expected: local commits are pushed to `origin/main`.

---

## Self-Review

Spec coverage:

- Upload formats and size limit are covered in UI and README tasks, but file-size validation should be added during implementation if Gradio does not enforce it directly.
- OpenCV-only detection is covered by `detector.py`.
- Candidate preview, selection, rename, and ZIP export are covered by `app.py` and `exporter.py`.
- Sorting, padding, manifest structure, selected-only export, and filename cleanup are covered by tests.
- Future phases are intentionally not implemented.

Known implementation risks:

- `merge_gap` defaults may need tuning against the real reference image.
- Gradio `Dataframe` may pass rows as pandas objects depending on installed Gradio version; `rows_to_elements` may need a small adapter if tests or manual smoke testing reveal that.
- File-size validation is not deeply tested in this plan because Gradio upload handling can vary. Add a small helper and test if strict enforcement becomes necessary.

Placeholder scan:

- No `TBD` or open-ended implementation steps remain.
- Each code task includes concrete files, test commands, implementation code, and commit points.

Type consistency:

- Box dictionaries use `x`, `y`, `width`, `height`, `area`.
- Element dictionaries use `id`, `file`, `x`, `y`, `width`, `height`, `selected`, `preview_path`.
- Manifest entries omit `selected` and `preview_path` as required by `SPEC.md`.
