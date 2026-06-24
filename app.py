from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image

from ai_grouper import (
    build_grouped_boxes,
    build_grouping_messages,
    parse_grouping_response,
    validate_grouping_response,
)
from annotator import image_to_data_url, save_annotated_candidates
from detector import detect_elements
from exporter import export_zip
from splitter import create_elements
from vision_client import QwenVisionClient


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
            element.get("type", ""),
            ",".join(str(item) for item in element.get("source_candidate_ids", [])),
            element.get("reason", ""),
        ]
        for element in elements
    ]


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
    boxes = detect_elements(
        image,
        min_area=int(min_area),
        merge_gap=int(merge_gap),
        padding=int(padding),
    )
    elements = create_elements(image, boxes, session_dir)
    gallery = [element["preview_path"] for element in elements]
    return gallery, elements_to_rows(elements), elements


def run_export(rows, elements: list[dict], output_root: str | Path = OUTPUT_ROOT) -> str | None:
    if not elements:
        return None

    session_dir = build_session_output_dir(output_root)
    updated_elements = rows_to_elements(rows, elements)
    return export_zip(updated_elements, session_dir)


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


def build_app():
    import gradio as gr

    with gr.Blocks(title="Infographic Splitter") as demo:
        gr.Markdown("# Infographic Splitter")

        state_elements = gr.State([])
        ai_status = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="上传图片")
                min_area = gr.Number(value=500, precision=0, label="min_area")
                merge_gap = gr.Slider(1, 40, value=8, step=1, label="merge_gap")
                padding = gr.Slider(0, 40, value=10, step=1, label="padding")
                split_button = gr.Button("开始拆分", variant="primary")
                ai_button = gr.Button("AI 语义合并")
                export_button = gr.Button("导出 ZIP")
                zip_output = gr.File(label="下载 ZIP")

            with gr.Column(scale=2):
                gallery = gr.Gallery(label="拆分结果", columns=4, height=360)
                table = gr.Dataframe(
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
                    interactive=True,
                    label="元素列表",
                )

        split_button.click(
            fn=run_split,
            inputs=[image_input, min_area, merge_gap, padding],
            outputs=[gallery, table, state_elements],
        )
        ai_button.click(
            fn=run_ai_grouping,
            inputs=[image_input, state_elements],
            outputs=[gallery, table, state_elements, ai_status],
        )
        export_button.click(
            fn=run_export,
            inputs=[table, state_elements],
            outputs=[zip_output],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
