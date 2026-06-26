from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from uuid import uuid4

from PIL import Image

from image_namer import dedupe_name

try:
    from rembg import new_session as rembg_new_session
    from rembg import remove as rembg_remove
except ImportError:  # pragma: no cover - handled at runtime
    rembg_new_session = None
    rembg_remove = None


OUTPUT_ROOT = Path("output")
DEFAULT_REMBG_MODEL = "u2net"


def build_session_output_dir(output_root: str | Path = OUTPUT_ROOT) -> Path:
    session_dir = Path(output_root) / f"session-{uuid4().hex[:12]}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


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


def build_transparent_copy_name(source_path: str | Path) -> str:
    stem = Path(source_path).stem.strip() or "image"
    cleaned = []
    previous_underscore = False
    for char in stem:
        if char.isalnum():
            cleaned.append(char)
            previous_underscore = False
            continue
        if char in {" ", "-", "_"} and not previous_underscore:
            cleaned.append("_")
            previous_underscore = True
            continue
        if not previous_underscore:
            cleaned.append("_")
            previous_underscore = True

    final_stem = "".join(cleaned).strip("_") or "image"
    return f"{final_stem}.png"


def _zip_files(file_paths: list[str], output_dir: Path) -> str | None:
    if not file_paths:
        return None

    zip_path = output_dir / "transparent_images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in file_paths:
            archive.write(file_path, arcname=Path(file_path).name)
    return str(zip_path)


def _load_image_from_result(result) -> Image.Image:
    if isinstance(result, Image.Image):
        return result.convert("RGBA")
    if isinstance(result, (bytes, bytearray)):
        with Image.open(io.BytesIO(result)) as image:
            return image.convert("RGBA").copy()
    raise TypeError("rembg output must be an image or bytes")


def _default_session_factory():
    if rembg_new_session is None:
        raise RuntimeError(
            "rembg is not installed. Install it with `pip install rembg`."
        )
    model_name = os.getenv("REMBG_MODEL", DEFAULT_REMBG_MODEL)
    return rembg_new_session(model_name)


def _default_remover(data: bytes, session=None):
    if rembg_remove is None:
        raise RuntimeError(
            "rembg is not installed. Install it with `pip install rembg`."
        )
    if session is None:
        return rembg_remove(data)
    return rembg_remove(data, session=session)


def _remove_background_one(
    file_path: Path,
    output_dir: Path,
    remover,
    session,
    used_names: set[str],
) -> dict:
    try:
        with Image.open(file_path) as image:
            image.verify()

        result = remover(file_path.read_bytes(), session=session)
        output_image = _load_image_from_result(result)
        output_name = dedupe_name(build_transparent_copy_name(file_path), used_names)
        output_path = output_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_image.save(output_path, format="PNG")
        return {
            "source_file": file_path.name,
            "file": output_name,
            "status": "transparent",
            "output_path": str(output_path),
            "reason": "background removed",
        }
    except Exception as error:
        return {
            "source_file": file_path.name,
            "file": file_path.name,
            "status": "failed",
            "output_path": "",
            "reason": f"失败：{error}",
        }


def remove_backgrounds(
    files,
    output_root: str | Path = OUTPUT_ROOT,
    remover=None,
    session_factory=None,
) -> tuple[list[str], str | None, list[list], str]:
    file_paths = [Path(path) for path in _normalize_files(files)]
    if not file_paths:
        return [], None, [], "请先上传图片"

    session_dir = build_session_output_dir(output_root)
    transparent_dir = session_dir / "transparent"
    session = session_factory() if session_factory else _default_session_factory()
    remover = remover or _default_remover

    rows = []
    downloads = []
    used_names: set[str] = set()

    for file_path in file_paths:
        result = _remove_background_one(
            file_path,
            transparent_dir,
            remover,
            session,
            used_names=used_names,
        )
        if result["status"] == "transparent":
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

    transparent_count = sum(1 for row in rows if row[2] == "transparent")
    if transparent_count == len(rows):
        status = f"透明背景完成：{transparent_count} 张图片"
    else:
        status = f"透明背景完成：{transparent_count} / {len(rows)} 张图片"
    zip_path = _zip_files(downloads, session_dir)
    return downloads, zip_path, rows, status
