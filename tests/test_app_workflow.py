from pathlib import Path

from PIL import Image, ImageDraw

from app import build_session_output_dir, elements_to_rows, run_ai_grouping, run_split


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

    gallery, rows, elements = run_split(
        image,
        min_area=200,
        merge_gap=5,
        padding=0,
        output_root=tmp_path,
    )

    assert len(gallery) == 1
    assert len(rows) == 1
    assert len(elements) == 1
    assert rows[0][0] is True
    assert rows[0][1] == "element_001.png"


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


def test_run_ai_grouping_uses_grouped_elements_on_success(tmp_path: Path):
    class FakeVisionClient:
        def complete(self, messages):
            return (
                '{"groups":[{"id":1,"file":"ai_group.png",'
                '"candidate_ids":[1],"type":"illustration","keep":true,'
                '"reason":"single grouped element"}],"ignored_candidate_ids":[]}'
            )

    image = Image.new("RGB", (140, 100), "white")
    elements = [
        {
            "id": 1,
            "file": "element_001.png",
            "x": 10,
            "y": 20,
            "width": 60,
            "height": 40,
            "selected": True,
            "preview_path": str(tmp_path / "old.png"),
        }
    ]

    gallery, rows, grouped, status = run_ai_grouping(
        image=image,
        elements=elements,
        output_root=tmp_path,
        vision_client=FakeVisionClient(),
    )

    assert status == "AI 语义合并完成：1 个元素"
    assert rows[0][1] == "ai_group.png"
    assert grouped[0]["file"] == "ai_group.png"
    assert grouped[0]["type"] == "illustration"
    assert grouped[0]["source_candidate_ids"] == [1]
    assert len(gallery) == 1
