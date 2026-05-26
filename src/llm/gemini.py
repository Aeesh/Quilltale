import os
import logging
import time
from google import genai
from dotenv import load_dotenv
from .base import BaseLLM, LLMResponse

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    def __init__(self, model: str = "gemini-3.1-flash-lite"):
        self._client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )
        self._model_name = model

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        final_prompt = f"{system}\n\n{prompt}" if system else prompt
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=final_prompt,
                )
                return LLMResponse(text=response.text, model=self._model_name)
            except Exception as e:
                if attempt == 2:
                    raise

                time.sleep(2 ** attempt)