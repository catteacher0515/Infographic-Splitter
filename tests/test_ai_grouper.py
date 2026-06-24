import pytest

from ai_grouper import (
    build_grouped_boxes,
    build_grouping_messages,
    extract_json_object,
    parse_grouping_response,
    validate_grouping_response,
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
    text = '```json\n{"groups": [], "ignored_candidate_ids": []}\n```'

    assert extract_json_object(text) == {"groups": [], "ignored_candidate_ids": []}


def test_parse_grouping_response_requires_json_object():
    with pytest.raises(ValueError, match="valid JSON object"):
        parse_grouping_response("not json")


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
