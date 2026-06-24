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
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    zip_path = output_dir / "export.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, "manifest.json")
        for asset_path in sorted(export_assets_dir.glob("*.png")):
            archive.write(asset_path, f"assets/{asset_path.name}")

    return str(zip_path)
