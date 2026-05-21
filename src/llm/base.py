from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        ...

    def generate_json(self, prompt: str, system: str = "") -> str:
        """Returns raw JSON string. Strips markdown fences."""
        full = prompt + "\n\nRespond ONLY with valid JSON without any preamble or markdown fences."
        resp = self.generate(full, system)
        text = resp.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        return text.strip()