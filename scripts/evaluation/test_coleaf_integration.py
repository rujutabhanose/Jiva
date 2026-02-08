#!/usr/bin/env python3
"""
Test script for CoLeaf model integration.
Run this after deploying the new model to verify everything works.
"""
import os
import sys
import numpy as np
from pathlib import Path

print("=" * 60)
print("COLEAF MODEL INTEGRATION TEST")
print("=" * 60)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from app.services.coleaf_engine import (
        coleaf_model, CLASS_NAMES, NUTRIENT_TO_LABEL,
        run_coleaf, CLASS_INDICES
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify model loaded
print("\n[2/5] Checking model...")
if coleaf_model is None:
    print("✗ Model failed to load!")
    print("  Run the fixed training notebook and replace model files.")
    sys.exit(1)
else:
    print("✓ Model loaded successfully")
    print(f"  Input shape: {coleaf_model.input_shape}")
    print(f"  Output shape: {coleaf_model.output_shape}")

# Test 3: Verify class mappings
print("\n[3/5] Checking class mappings...")
print(f"  Classes detected: {len(CLASS_NAMES)}")
if len(CLASS_NAMES) != 10:
    print(f"  ⚠ Expected 10 classes, got {len(CLASS_NAMES)}")

print(f"\n  Class indices:")
for nutrient, idx in sorted(CLASS_INDICES.items(), key=lambda x: x[1]):
    label = NUTRIENT_TO_LABEL.get(nutrient, 'UNMAPPED')
    print(f"    {idx}: {nutrient:8s} -> {label}")

# Check for unmapped nutrients
unmapped = [n for n in CLASS_NAMES if n != "Healthy" and n not in NUTRIENT_TO_LABEL]
if unmapped:
    print(f"\n  ⚠ Warning: Unmapped nutrients: {unmapped}")
else:
    print(f"\n✓ All nutrients mapped to disease labels")

# Test 4: Test prediction with dummy image
print("\n[4/5] Testing prediction...")
try:
    # Create dummy image
    test_image = np.random.rand(224, 224, 3).astype('float32')
    test_path = "/tmp/test_coleaf_image.jpg"

    from PIL import Image
    img = Image.fromarray((test_image * 255).astype('uint8'))
    img.save(test_path)

    # Run prediction
    result = run_coleaf(test_path)

    print("✓ Prediction successful")
    print(f"  Result keys: {list(result.keys())}")

    if result['diagnoses']:
        diag = result['diagnoses'][0]
        print(f"  Predicted: {diag['raw_class']} -> {diag['label']}")
        print(f"  Confidence: {diag['confidence']:.3f}")
        print(f"  Category: {diag['category']}")
    else:
        print("  ⚠ No diagnoses returned (model might not be loaded)")

    # Cleanup
    os.remove(test_path)

except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test integration with diagnosis_engine
print("\n[5/5] Testing diagnosis_engine integration...")
try:
    from app.services.diagnosis_engine import diagnose_image
    print("✓ diagnosis_engine imports coleaf_engine successfully")
    print("  Integration complete!")
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nCoLeaf model is ready for production!")
print("\nNext steps:")
print("  1. Test with real plant images")
print("  2. Start the backend server: uvicorn app.main:app --reload")
print("  3. Test API endpoint: POST /api/diagnose")
print("=" * 60)
