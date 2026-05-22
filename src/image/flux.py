"""
Generate scene images using FLUX.1-schnell.

Images are only generated when the game scene changes.
Handles both ZeroGPU (HF Spaces) and CPU (local development) paths.
"""

import logging
import torch
from diffusers import FluxPipeline
from PIL import Image

logger = logging.getLogger(__name__)

# Shared visual style appended to every image prompt for visual consistency
STYLE_PREFIX = (
    "Dark fantasy illustration, oil painting style, dramatic lighting, "
    "detailed environment, atmospheric fog, muted earth tones with accent colours. "
)

flux_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)

try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False

if HAS_ZEROGPU:
    flux_pipe.to("cuda")
else:
    flux_pipe.enable_model_cpu_offload()


def generate_scene_image(prompt: str) -> Image.Image | None:
    """
    Generate a scene image from a text prompt.
    Returns PIL Image or None on failure.
    """
    if not prompt:
        return None

    full_prompt = STYLE_PREFIX + prompt

    try:
        return _generate(full_prompt)
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return None


def _run_pipe(prompt: str) -> Image.Image:
    """Core generation logic shared between both paths."""
    result = flux_pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        width=768,
        height=512,
    )
    return result.images[0]


if HAS_ZEROGPU:
    @spaces.GPU
    def _generate(prompt: str) -> Image.Image:
        return _run_pipe(prompt)
else:
    def _generate(prompt: str) -> Image.Image:
        return _run_pipe(prompt)