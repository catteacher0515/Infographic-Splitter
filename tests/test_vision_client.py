import pytest

from vision_client import QwenVisionClient, VisionConfig, load_vision_config


def test_load_vision_config_reads_defaults(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("QWEN_MODEL", raising=False)

    config = load_vision_config()

    assert config.api_key == "test-key"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.model == "qwen3-vl-flash"


def test_load_vision_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        load_vision_config()


def test_qwen_vision_client_returns_message_content():
    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "qwen3-vl-flash"
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": '{"groups": []}'},
                                )()
                            },
                        )()
                    ]
                },
            )()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = QwenVisionClient(
        VisionConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="qwen3-vl-flash",
        ),
        client=FakeClient(),
    )

    content = client.complete([{"role": "user", "content": "hello"}])

    assert content == '{"groups": []}'
