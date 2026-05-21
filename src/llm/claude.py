import os
import anthropic
from .base import BaseLLM, LLMResponse


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        kwargs = {"model": self._model, "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        return LLMResponse(text=msg.content[0].text, model=self._model)