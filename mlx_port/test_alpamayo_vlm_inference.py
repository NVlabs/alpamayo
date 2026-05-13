"""
Inference test using the fine-tuned Alpamayo VLM weights (Qwen3-VL-8B backbone).

This script:
1. Loads the base Qwen3-VL-8B structure
2. Loads the fine-tuned VLM weights from Alpamayo-R1-10B using key remapping
3. Runs Alpamayo-style Chain-of-Cognition generation on the test image
"""

from pathlib import Path
from mlx_port.load_alpamayo_vlm import load_alpamayo_vlm_weights
from mlx_port.vlm_loader import generate_alpamayo_coc, MIN_PIXELS, MAX_PIXELS


TEST_IMAGE = Path("reports/stage1_test_image_front_wide.png")


def main():
    if not TEST_IMAGE.exists():
        print(f"Test image not found at {TEST_IMAGE}")
        return

    print("=== Loading Alpamayo fine-tuned VLM weights ===")
    model, processor = load_alpamayo_vlm_weights()

    print("\n=== Running inference with Alpamayo VLM ===")
    images = [TEST_IMAGE]

    output = generate_alpamayo_coc(
        model,
        processor,
        images=images,
        max_tokens=128,
        temperature=0.0,  # deterministic for testing
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("\n--- Alpamayo VLM Output ---")
    print(output)
    print("\n[Done] Alpamayo VLM inference test completed.")


if __name__ == "__main__":
    main()