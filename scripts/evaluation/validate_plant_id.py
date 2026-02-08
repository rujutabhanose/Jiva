#!/usr/bin/env python3
"""
Validate Plant Identification - Enhanced Ensemble Version

Tests both the original single model and the new ensemble approach.
Compares results and shows confidence improvements.
"""

import sys
from pathlib import Path
import time

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
import torch

# Config
UPLOADS_DIR = Path("./uploads")
TOP_K = 5

print("=" * 70)
print("PLANT IDENTIFICATION - VALIDATION SUITE")
print("=" * 70)


def test_single_model():
    """Test original single ViT model"""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    MODEL_ID = "juppy44/plant-identification-2m-vit-b"

    print("\n[TEST 1] Single Model (Original)")
    print("-" * 50)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    print(f"  Model: {MODEL_ID}")
    print(f"  Species coverage: {len(model.config.id2label)}")

    return processor, model


def test_ensemble_model():
    """Test new ensemble approach"""
    try:
        from scripts.plant_id.plant_identifier_ensemble import EnsemblePlantIdentifier
        from scripts.plant_id.plant_id_postprocessor import PlantIdentificationPostProcessor
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent / "plant_id"))
        from plant_identifier_ensemble import EnsemblePlantIdentifier
        from plant_id_postprocessor import PlantIdentificationPostProcessor

    print("\n[TEST 2] Ensemble Model (Enhanced)")
    print("-" * 50)

    identifier = EnsemblePlantIdentifier()
    postprocessor = PlantIdentificationPostProcessor()

    print(f"  Models loaded: {len(identifier.models)}")
    print(f"  Confidence threshold: {postprocessor.confidence_threshold}%")

    return identifier, postprocessor


def run_comparison(test_images, single_model, ensemble_model):
    """Compare single vs ensemble results"""
    processor, model = single_model
    identifier, postprocessor = ensemble_model

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    results_summary = []

    for img_path in test_images[:5]:  # Limit to 5 for speed
        print(f"\n{'='*70}")
        print(f"Image: {img_path.name}")
        print("=" * 70)

        # --- Single Model ---
        print("\n  [Single Model]")
        start = time.time()
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            probs = outputs.logits.softmax(dim=-1)[0]
            topk = torch.topk(probs, k=TOP_K)

            single_top = model.config.id2label[topk.indices[0].item()]
            single_conf = float(topk.values[0]) * 100
            single_time = time.time() - start

            print(f"    Top: {single_top}")
            print(f"    Confidence: {single_conf:.1f}%")
            print(f"    Time: {single_time:.2f}s")

        except Exception as e:
            print(f"    Error: {e}")
            single_top, single_conf, single_time = "Error", 0, 0

        # --- Ensemble Model ---
        print("\n  [Ensemble Model]")
        start = time.time()
        try:
            raw_result = identifier.identify_single(str(img_path), top_k=TOP_K, debug=False)
            final_result = postprocessor.process_result(raw_result)
            ensemble_time = time.time() - start

            if final_result.get('primary'):
                ensemble_top = final_result['primary'].get('common_name') or final_result['primary'].get('scientific_name')
                ensemble_conf = final_result['primary'].get('confidence', 0)
                conf_level = final_result.get('confidence_level', 'unknown')
            else:
                ensemble_top = "Low confidence"
                ensemble_conf = 0
                conf_level = "very_low"

            print(f"    Top: {ensemble_top}")
            print(f"    Confidence: {ensemble_conf:.1f}%")
            print(f"    Level: {conf_level}")
            print(f"    Time: {ensemble_time:.2f}s")

            if final_result.get('recommendation'):
                print(f"    Care: {final_result['recommendation'][:60]}...")

        except Exception as e:
            print(f"    Error: {e}")
            ensemble_top, ensemble_conf, ensemble_time, conf_level = "Error", 0, 0, "error"

        # --- Summary ---
        results_summary.append({
            'image': img_path.name,
            'single': {'name': single_top, 'conf': single_conf, 'time': single_time},
            'ensemble': {'name': ensemble_top, 'conf': ensemble_conf, 'time': ensemble_time, 'level': conf_level}
        })

    return results_summary


def print_summary(results):
    """Print comparison summary"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Image':<25} {'Single Model':<25} {'Ensemble Model':<25}")
    print("-" * 75)

    for r in results:
        single = f"{r['single']['conf']:.0f}%"
        ensemble = f"{r['ensemble']['conf']:.0f}% ({r['ensemble']['level']})"
        print(f"{r['image'][:24]:<25} {single:<25} {ensemble:<25}")

    # Averages
    avg_single = sum(r['single']['conf'] for r in results) / len(results) if results else 0
    avg_ensemble = sum(r['ensemble']['conf'] for r in results) / len(results) if results else 0
    avg_single_time = sum(r['single']['time'] for r in results) / len(results) if results else 0
    avg_ensemble_time = sum(r['ensemble']['time'] for r in results) / len(results) if results else 0

    print("-" * 75)
    print(f"{'Average Confidence:':<25} {avg_single:.1f}%{'':<19} {avg_ensemble:.1f}%")
    print(f"{'Average Time:':<25} {avg_single_time:.2f}s{'':<19} {avg_ensemble_time:.2f}s")


def main():
    # Find test images
    print("\n[SETUP] Finding test images...")
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    test_images = [
        f for f in UPLOADS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"  Found {len(test_images)} images in {UPLOADS_DIR}")

    if not test_images:
        print("\n  No images found! Add images to ./uploads/ directory.")
        sys.exit(1)

    # Load models
    single_model = test_single_model()
    ensemble_model = test_ensemble_model()

    # Run comparison
    results = run_comparison(test_images, single_model, ensemble_model)

    # Print summary
    print_summary(results)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
