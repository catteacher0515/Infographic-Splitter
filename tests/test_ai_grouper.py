import pytest

from ai_grouper import (
    build_grouping_messages,
    extract_json_object,
    parse_grouping_response,
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
