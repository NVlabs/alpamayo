"""
Smoke tests for the VLM component of Alpamayo.
"""

import pytest
from pathlib import Path

from mlx_port.load_alpamayo_vlm import load_alpamayo_vlm_weights
from mlx_port.vlm_loader import generate_alpamayo_coc, MIN_PIXELS, MAX_PIXELS

TEST_IMAGE = Path("reports/stage1_test_image_front_wide.png")


@pytest.mark.skipif(not TEST_IMAGE.exists(), reason="Test image not present")
def test_vlm_load_and_generate():
    """VLM should load fine-tuned weights and produce non-empty CoC text."""
    model, processor = load_alpamayo_vlm_weights()

    images = [TEST_IMAGE]
    output = generate_alpamayo_coc(
        model,
        processor,
        images=images,
        max_tokens=32,
        temperature=0.0,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # Basic smoke assertions
    assert output is not None
    assert isinstance(output.text, str)
    assert len(output.text) > 0
    assert "speed bump" in output.text.lower() or "lane" in output.text.lower()
