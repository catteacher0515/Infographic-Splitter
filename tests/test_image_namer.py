from image_namer import (
    build_naming_messages,
    dedupe_name,
    parse_naming_response,
    sanitize_copy_name,
)


def test_build_naming_messages_includes_system_and_image():
    messages = build_naming_messages("data:image/png;base64,abc")

    assert messages[0]["role"] == "system"
    assert "图片内容命名助手" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][1]["type"] == "image_url"


def test_parse_naming_response_reads_file_name_from_json():
    response = '{"file":"prompt card.png","reason":"single prompt"}'

    parsed = parse_naming_response(response)

    assert parsed["file"] == "prompt card.png"
    assert parsed["reason"] == "single prompt"


def test_sanitize_copy_name_keeps_original_extension():
    assert sanitize_copy_name("一句话 Prompt", "source.jpg") == "一句话_prompt.jpg"
    assert sanitize_copy_name("../坏/名字", "source.png") == "名字.png"


def test_dedupe_name_adds_counter_suffix():
    used = set()

    first = dedupe_name("prompt_card.png", used)
    second = dedupe_name("prompt_card.png", used)

    assert first == "prompt_card.png"
    assert second == "prompt_card_2.png"
