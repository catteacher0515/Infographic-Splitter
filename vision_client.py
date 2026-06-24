from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3-vl-flash"


@dataclass(frozen=True)
class VisionConfig:
    api_key: str
    base_url: str = DEFAULT_QWEN_BASE_URL
    model: str = DEFAULT_QWEN_MODEL


def load_vision_config() -> VisionConfig:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is not set")

    return VisionConfig(
        api_key=api_key,
        base_url=os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
        model=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
    )


class QwenVisionClient:
    def __init__(self, config: VisionConfig | None = None, client: Any | None = None):
        self.config = config or load_vision_config()
        self.client = client or OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def complete(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=0,
        )
        return str(response.choices[0].message.content)
