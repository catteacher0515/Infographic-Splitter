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
