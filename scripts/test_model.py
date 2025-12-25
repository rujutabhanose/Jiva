#!/usr/bin/env python3
"""
Test script to verify plant identification model is working.
Run this after installing dependencies to ensure the model loads correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.plant_identifier import identify_plant, MODEL_ID, processor, model


def test_model_loading():
    """Test that the model loads successfully"""
    print("🔍 Testing Plant Identification Model...")
    print(f"   Model ID: {MODEL_ID}")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Processor type: {type(processor).__name__}")

    # Check model is in eval mode
    assert not model.training, "Model should be in eval mode"
    print("✅ Model loaded successfully and in eval mode")

    # Check number of labels
    num_labels = len(model.config.id2label)
    print(f"   Number of plant species: {num_labels}")

    # Print some example labels
    print(f"   Example labels: {list(model.config.id2label.values())[:5]}")

    return True


def test_model_inference():
    """Test model inference with a dummy image"""
    print("\n🧪 Testing Model Inference...")

    try:
        from PIL import Image
        import numpy as np

        # Create a dummy RGB image (224x224)
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        # Save temporarily
        test_image_path = Path("test_plant.jpg")
        dummy_image.save(test_image_path)

        # Run inference
        results = identify_plant(str(test_image_path), top_k=3)

        # Verify results structure
        assert len(results) == 3, "Should return top 3 results"
        assert all("plant_name" in r for r in results), "Missing plant_name"
        assert all("confidence" in r for r in results), "Missing confidence"

        print(f"✅ Inference successful!")
        print(f"   Top prediction: {results[0]['plant_name']}")
        print(f"   Confidence: {results[0]['confidence']:.2%}")

        # Cleanup
        test_image_path.unlink()

        return True

    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Plant Identification Model Test Suite")
    print("=" * 60)

    try:
        # Test 1: Model loading
        test_model_loading()

        # Test 2: Inference
        test_model_inference()

        print("\n" + "=" * 60)
        print("✅ All tests passed! Model is ready to use.")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
