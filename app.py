from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

from PIL import Image

from annotator import image_to_data_url
from image_namer import (
    dedupe_name,
    build_naming_messages,
    parse_naming_response,
    sanitize_copy_name,
)
from vision_client import QwenVisionClient


OUTPUT_ROOT = Path("output")


def build_session_output_dir(output_root: str | Path = OUTPUT_ROOT) -> Path:
    session_dir = Path(output_root) / f"session-{uuid4().hex[:12]}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def rows_to_downloads(rows) -> list[str]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    return [str(row[4]) for row in rows if len(row) > 4 and row[2] == "renamed"]


def _normalize_files(files) -> list[str]:
    if not files:
        return []
    if hasattr(files, "tolist"):
        files = files.tolist()
    normalized = []
    for item in files:
        if isinstance(item, dict) and "name" in item:
            normalized.append(str(item["name"]))
        elif isinstance(item, dict) and "path" in item:
            normalized.append(str(item["path"]))
        else:
            normalized.append(str(item))
    return normalized


def _copy_with_renamed_file(source_path: Path, destination_dir: Path, filename: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / filename
    shutil.copy2(source_path, destination_path)
    return destination_path


def _zip_renamed_files(file_paths: list[str], output_dir: Path) -> str | None:
    if not file_paths:
        return None

    zip_path = output_dir / "renamed_images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in file_paths:
            archive.write(file_path, arcname=Path(file_path).name)
    return str(zip_path)


def _rename_one(
    file_path: Path,
    output_dir: Path,
    vision_client: QwenVisionClient,
) -> dict:
    try:
        with Image.open(file_path) as image:
            image.verify()
        messages = build_naming_messages(image_to_data_url(file_path))
        raw_response = vision_client.complete(messages)
        parsed = parse_naming_response(raw_response)
        proposed_name = sanitize_copy_name(parsed["file"], file_path)
        return {
            "source_file": file_path.name,
            "file": proposed_name,
            "status": "renamed",
            "output_path": "",
            "reason": parsed["reason"],
        }
    except Exception as error:
        return {
            "source_file": file_path.name,
            "file": file_path.name,
            "status": "failed",
            "output_path": "",
            "reason": f"失败：{error}",
        }


def rename_images(
    files,
    output_root: str | Path = OUTPUT_ROOT,
    vision_client: QwenVisionClient | None = None,
) -> tuple[list[str], str | None, list[list], str]:
    file_paths = [Path(path) for path in _normalize_files(files)]
    if not file_paths:
        return [], None, [], "请先上传图片"

    session_dir = build_session_output_dir(output_root)
    renamed_dir = session_dir / "renamed"
    client = vision_client or QwenVisionClient()

    rows = []
    downloads = []
    used_names: set[str] = set()

    for file_path in file_paths:
        result = _rename_one(file_path, renamed_dir, client)
        if result["status"] == "renamed":
            result["file"] = dedupe_name(result["file"], used_names)
            result["output_path"] = str(
                _copy_with_renamed_file(file_path, renamed_dir, result["file"])
            )
            downloads.append(result["output_path"])
        rows.append(
            [
                result["source_file"],
                result["file"],
                result["status"],
                result["reason"],
                result["output_path"],
            ]
        )

    renamed_count = sum(1 for row in rows if row[2] == "renamed")
    if renamed_count == len(rows):
        status = f"AI 重命名完成：{renamed_count} 张图片"
    else:
        status = f"AI 重命名完成：{renamed_count} / {len(rows)} 张图片"
    zip_path = _zip_renamed_files(downloads, session_dir)
    return downloads, zip_path, rows, status


def build_app():
    import gradio as gr

    with gr.Blocks(title="Image Renamer") as demo:
        gr.Markdown("# Image Renamer")

        with gr.Row():
            with gr.Column(scale=1):
                files_input = gr.Files(
                    label="上传图片",
                    file_count="multiple",
                    type="filepath",
                )
                rename_button = gr.Button("AI 重命名", variant="primary")
                status = gr.Markdown("上传图片后，点击 AI 重命名。")

            with gr.Column(scale=2):
                table = gr.Dataframe(
                    headers=[
                        "原文件名",
                        "新文件名",
                        "状态",
                        "原因",
                        "输出路径",
                    ],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    label="结果",
                )
                downloads = gr.Files(label="下载副本", interactive=False)
                zip_output = gr.File(label="下载 zip")

        rename_button.click(
            fn=rename_images,
            inputs=[files_input],
            outputs=[downloads, zip_output, table, status],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
