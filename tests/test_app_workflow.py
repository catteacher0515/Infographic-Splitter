from pathlib import Path
import zipfile

from PIL import Image

from app import (
    build_session_output_dir,
    rename_images,
    rows_to_downloads,
)


def test_build_session_output_dir_creates_unique_session(tmp_path: Path):
    first = build_session_output_dir(tmp_path)
    second = build_session_output_dir(tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_rows_to_downloads_returns_generated_files():
    rows = [
        ["source_a.png", "prompt_card.png", "renamed", "ok", "/tmp/prompt_card.png"],
        ["source_b.png", "context_scene.png", "renamed", "ok", "/tmp/context_scene.png"],
    ]

    downloads = rows_to_downloads(rows)

    assert downloads == ["/tmp/prompt_card.png", "/tmp/context_scene.png"]


def test_rename_images_returns_error_without_files(tmp_path: Path):
    downloads, zip_path, rows, status = rename_images(
        files=[],
        output_root=tmp_path,
    )

    assert downloads == []
    assert zip_path is None
    assert rows == []
    assert "请先上传图片" in status


def test_rename_images_creates_renamed_copies_with_deduped_names(tmp_path: Path):
    class FakeVisionClient:
        def complete(self, messages):
            return '{"file":"提示卡片.png","reason":"单张提示卡片"}'

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (80, 40), "white").save(first)
    Image.new("RGB", (80, 40), "white").save(second)

    downloads, zip_path, rows, status = rename_images(
        files=[str(first), str(second)],
        output_root=tmp_path,
        vision_client=FakeVisionClient(),
    )

    assert status == "AI 重命名完成：2 张图片"
    assert [row[1] for row in rows] == ["提示卡片.png", "提示卡片_2.png"]
    assert [Path(path).name for path in downloads] == ["提示卡片.png", "提示卡片_2.png"]
    assert all(Path(path).exists() for path in downloads)
    assert zip_path is not None
    assert Path(zip_path).exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["提示卡片.png", "提示卡片_2.png"]
    assert rows[0][2] == "renamed"


def test_rename_images_marks_failed_file_when_ai_response_is_invalid(tmp_path: Path):
    class FakeVisionClient:
        def complete(self, messages):
            return "not json"

    image_path = tmp_path / "single.png"
    Image.new("RGB", (80, 40), "white").save(image_path)

    downloads, zip_path, rows, status = rename_images(
        files=[str(image_path)],
        output_root=tmp_path,
        vision_client=FakeVisionClient(),
    )

    assert downloads == []
    assert zip_path is None
    assert rows[0][0] == "single.png"
    assert rows[0][2] == "failed"
    assert "失败" in rows[0][3]
    assert status == "AI 重命名完成：0 / 1 张图片"
