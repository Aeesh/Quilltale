"""
Generate scene images via the HuggingFace Inference API.

Images are only generated when the game scene changes.
"""

import os
import logging
import requests
from PIL import Image
from huggingface_hub import InferenceClient
from io import BytesIO

logger = logging.getLogger(__name__)

# Shared visual style appended to every image prompt for visual consistency
STYLE_PREFIX = (
    "Dark fantasy illustration, oil painting style, dramatic lighting, "
    "detailed environment, atmospheric fog, muted earth tones with accent colours. "
)

MODEL = "black-forest-labs/FLUX.1-schnell"

_client = InferenceClient(token=os.environ.get("HF_TOKEN"))


def generate_scene_image(prompt: str) -> Image.Image | None:
    """
    Generate a scene image from a text prompt.
    Returns PIL Image or None on failure.
    """
    if not prompt:
        return None

    full_prompt = STYLE_PREFIX + prompt

    try:
        image = _client.text_to_image(
            full_prompt,
            model=MODEL,
        )
        return image
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        logger.info("Falling back to Pollinations for image generation.")
        try:
            # fallback to Pollinations
            url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(full_prompt)}"
            response = requests.get(url, timeout=60)
            image = Image.open(BytesIO(response.content))
            return image
        except Exception:
            return None