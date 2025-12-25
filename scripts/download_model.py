#!/usr/bin/env python3
"""
Pre-download the plant identification model.
Run this script to download the model before starting the server.
This prevents the first request from being slow.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_model():
    """Download the plant identification model from HuggingFace"""
    print("📦 Downloading Plant Identification Model...")
    print("   This may take a few minutes depending on your internet speed.\n")

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        MODEL_ID = "juppy44/plant-identification-2m-vit-b"

        print(f"🔽 Downloading processor from {MODEL_ID}...")
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        print("✅ Processor downloaded")

        print(f"\n🔽 Downloading model from {MODEL_ID}...")
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        print("✅ Model downloaded")

        # Get model info
        num_labels = len(model.config.id2label)
        print(f"\n📊 Model Information:")
        print(f"   • Total plant species: {num_labels}")
        print(f"   • Model architecture: {model.config.model_type}")
        print(f"   • Hidden size: {model.config.hidden_size}")

        # Print some example species
        print(f"\n🌿 Example plant species this model can identify:")
        for i, label in enumerate(list(model.config.id2label.values())[:10]):
            print(f"   {i+1}. {label}")
        print("   ... and many more!")

        print("\n" + "=" * 60)
        print("✅ Model downloaded successfully!")
        print("   The model is cached and ready for use.")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(download_model())
