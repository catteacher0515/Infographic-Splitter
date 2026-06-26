from io import BytesIO
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw

from background_remover import build_transparent_copy_name, remove_backgrounds


def test_build_transparent_copy_name_uses_png_extension():
    assert build_transparent_copy_name("photo.jpg") == "photo.png"
    assert build_transparent_copy_name("子图.webp") == "子图.png"


def test_remove_backgrounds_creates_transparent_copies_and_zip(tmp_path: Path):
    class FakeRemover:
        def __call__(self, data, session=None):
            with Image.open(BytesIO(data)) as image:
                image = image.convert("RGBA")
                output = Image.new("RGBA", image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(output)
                draw.rectangle((10, 10, 30, 30), fill=(0, 0, 0, 255))
                buffer = BytesIO()
                output.save(buffer, format="PNG")
                return buffer.getvalue()

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (40, 40), "white").save(first)
    Image.new("RGB", (40, 40), "white").save(second)

    downloads, zip_path, rows, status = remove_backgrounds(
        files=[str(first), str(second)],
        output_root=tmp_path,
        remover=FakeRemover(),
        session_factory=lambda: object(),
    )

    assert status == "透明背景完成：2 张图片"
    assert zip_path is not None
    assert Path(zip_path).exists()
    assert [Path(path).name for path in downloads] == ["first.png", "second.png"]
    assert rows[0][2] == "transparent"
    assert Image.open(downloads[0]).getpixel((5, 5))[3] == 0
    assert Image.open(downloads[0]).getpixel((15, 15))[3] == 255

    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["first.png", "second.png"]


def test_remove_backgrounds_reports_failed_files(tmp_path: Path):
    class FakeRemover:
        def __call__(self, data, session=None):
            raise RuntimeError("boom")

    image_path = tmp_path / "single.png"
    Image.new("RGB", (20, 20), "white").save(image_path)

    downloads, zip_path, rows, status = remove_backgrounds(
        files=[str(image_path)],
        output_root=tmp_path,
        remover=FakeRemover(),
        session_factory=lambda: object(),
    )

    assert downloads == []
    assert zip_path is None
    assert rows[0][0] == "single.png"
    assert rows[0][2] == "failed"
    assert "失败" in rows[0][3]
    assert status == "透明背景完成：0 / 1 张图片"
