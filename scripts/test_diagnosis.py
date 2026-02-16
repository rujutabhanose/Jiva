#!/usr/bin/env python3
"""
Test script for the hybrid diagnosis engine.
Tests disease detection (TFLite + HuggingFace) and nutrient deficiency (CoLeaf) on any image.

Usage:
    python scripts/test_diagnosis.py <image_path>
    python scripts/test_diagnosis.py <image_path> --full    # Run full pipeline including plant ID
    python scripts/test_diagnosis.py <image_path> --disease  # Disease detection only
    python scripts/test_diagnosis.py <image_path> --nutrient  # Nutrient detection only
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )


def test_disease_detection(image_path: str, top_k: int = 5):
    """Run disease detection only (TFLite + HuggingFace hybrid)."""
    from app.services.diagnosis_engine import _run_disease_model, _disease_model_format

    print("\n" + "=" * 60)
    print("  DISEASE DETECTION")
    print("=" * 60)
    print(f"  Image:  {image_path}")
    print(f"  Model:  {_disease_model_format or 'none loaded'}")
    print("-" * 60)

    result = _run_disease_model(image_path, top_k=top_k)

    source = result.get("source", "unknown")
    confidence = result.get("confidence", 0)
    diagnoses = result.get("diagnoses", [])

    print(f"  Source:     {source}")
    print(f"  Confidence: {confidence * 100:.2f}%")
    print(f"  Results:    {len(diagnoses)} diagnoses")
    print("-" * 60)

    if not diagnoses:
        print("  No diseases detected.")
    else:
        for i, d in enumerate(diagnoses, 1):
            conf = d.get("confidence", 0) * 100
            label = d.get("label", "unknown")
            category = d.get("category", "unknown")
            name = d.get("info").name if d.get("info") else label
            src = d.get("source", "unknown")

            marker = " <-- TOP" if i == 1 else ""
            print(f"  [{i}] {name}")
            print(f"      Label:      {label}")
            print(f"      Confidence: {conf:.2f}%")
            print(f"      Category:   {category}")
            print(f"      Source:     {src}")
            print(f"      {marker}")
            if i < len(diagnoses):
                print()

    print("=" * 60)
    return result


def test_nutrient_detection(image_path: str):
    """Run nutrient deficiency detection only (CoLeaf)."""
    from app.services.coleaf_engine import run_coleaf

    print("\n" + "=" * 60)
    print("  NUTRIENT DEFICIENCY DETECTION (CoLeaf)")
    print("=" * 60)
    print(f"  Image: {image_path}")
    print("-" * 60)

    result = run_coleaf(image_path)
    diagnoses = result.get("diagnoses", [])
    confidence = result.get("confidence", 0)

    print(f"  Confidence: {confidence * 100:.2f}%")
    print(f"  Results:    {len(diagnoses)} diagnoses")
    print("-" * 60)

    if not diagnoses:
        print("  No nutrient deficiencies detected.")
    else:
        for i, d in enumerate(diagnoses, 1):
            conf = d.get("confidence", 0) * 100
            label = d.get("label", "unknown")
            category = d.get("category", "unknown")
            print(f"  [{i}] {label} ({conf:.2f}%) - {category}")

    print("=" * 60)
    return result


def test_full_pipeline(image_path: str):
    """Run the full diagnosis pipeline (disease + nutrient + plant ID)."""
    from app.services.diagnosis_engine import diagnose_image

    print("\n" + "=" * 60)
    print("  FULL DIAGNOSIS PIPELINE")
    print("=" * 60)
    print(f"  Image: {image_path}")
    print("-" * 60)

    result = diagnose_image(image_path)

    success = result.get("success", False)
    health_score = result.get("plant_health_score")
    plant_name = result.get("plant_name")
    primary = result.get("primary_diagnosis")
    all_diag = result.get("all_diagnoses", [])
    recommendations = result.get("recommendations", [])
    sources = result.get("ai_sources_used", [])

    print(f"  Success:      {success}")
    print(f"  Plant Name:   {plant_name or 'Not identified'}")
    print(f"  Health Score: {health_score}/100")
    print(f"  AI Sources:   {', '.join(sources) if sources else 'none'}")
    print("-" * 60)

    if primary:
        print(f"\n  PRIMARY DIAGNOSIS:")
        print(f"    Name:       {primary.get('name', 'unknown')}")
        print(f"    Confidence: {primary.get('confidence', 0) * 100:.2f}%")
        print(f"    Category:   {primary.get('category', 'unknown')}")
        print(f"    Severity:   {primary.get('severity', 'unknown')}")
        print(f"    Source:     {primary.get('source', 'unknown')}")

        symptoms = primary.get("symptoms", [])
        if symptoms:
            print(f"\n    Symptoms:")
            for s in symptoms[:3]:
                print(f"      - {s}")

        causes = primary.get("causes", [])
        if causes:
            print(f"\n    Causes:")
            for c in causes[:3]:
                print(f"      - {c}")

        treatment = primary.get("treatment", [])
        if treatment:
            print(f"\n    Treatment:")
            for t in treatment[:3]:
                print(f"      - {t}")

    if len(all_diag) > 1:
        print(f"\n  OTHER DIAGNOSES:")
        for d in all_diag[1:]:
            print(f"    - {d.get('name', 'unknown')} ({d.get('confidence', 0) * 100:.2f}%) [{d.get('category')}]")

    if recommendations:
        print(f"\n  RECOMMENDATIONS:")
        for r in recommendations:
            print(f"    {r}")

    print("\n" + "=" * 60)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test the Jiva diagnosis engine on any plant image."
    )
    parser.add_argument(
        "image",
        help="Path to the plant image to diagnose"
    )
    parser.add_argument(
        "--disease", action="store_true",
        help="Run disease detection only"
    )
    parser.add_argument(
        "--nutrient", action="store_true",
        help="Run nutrient deficiency detection only"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full pipeline (disease + nutrient + plant ID)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top predictions to return (default: 5)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose/debug logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Validate image exists
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1

    if not image_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
        print(f"Warning: Unexpected image format: {image_path.suffix}")

    print(f"\nJiva Diagnosis Engine - Test Runner")
    print(f"Image: {image_path}")

    # Default: run both disease and nutrient if no flag specified
    run_disease = args.disease or (not args.disease and not args.nutrient and not args.full)
    run_nutrient = args.nutrient or (not args.disease and not args.nutrient and not args.full)
    run_full = args.full

    if run_full:
        test_full_pipeline(str(image_path))
    else:
        if run_disease:
            test_disease_detection(str(image_path), top_k=args.top_k)
        if run_nutrient:
            test_nutrient_detection(str(image_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
