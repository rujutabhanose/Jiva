#!/usr/bin/env python3
"""
Test the Plant Identification API endpoint.
This script tests the /api/v1/identify/ endpoint with a sample image.
"""

import requests
import sys
from pathlib import Path
from PIL import Image
import io
import numpy as np


def create_test_image():
    """Create a test plant-like image"""
    # Create a simple green image that looks vaguely like vegetation
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)

    # Add some green patches (simulating leaves)
    for i in range(10):
        x = np.random.randint(0, 200)
        y = np.random.randint(0, 200)
        size = np.random.randint(20, 40)

        # Green color with some variation
        green = np.random.randint(80, 200)
        img_array[y:y+size, x:x+size, 1] = green  # Green channel
        img_array[y:y+size, x:x+size, 0] = green // 3  # Red channel
        img_array[y:y+size, x:x+size, 2] = green // 2  # Blue channel

    img = Image.fromarray(img_array)
    return img


def test_identify_endpoint(api_url="http://localhost:8000", image_path=None):
    """
    Test the /api/v1/identify/ endpoint

    Args:
        api_url: Base URL of the API
        image_path: Path to plant image (if None, creates a test image)
    """
    print("=" * 60)
    print("Testing Plant Identification API")
    print("=" * 60)

    # Check if server is running
    print(f"\n1. Checking if server is running at {api_url}...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\n💡 Make sure the server is running:")
        print("   cd backend && uvicorn app.main:app --reload")
        return False

    # Prepare image
    print("\n2. Preparing test image...")
    if image_path and Path(image_path).exists():
        print(f"   Using provided image: {image_path}")
        files = {"file": open(image_path, "rb")}
    else:
        print("   Creating synthetic test image...")
        img = create_test_image()
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        files = {"file": ("test_plant.jpg", img_bytes, "image/jpeg")}

    # Test the identify endpoint
    print("\n3. Calling /api/v1/identify/ endpoint...")
    try:
        response = requests.post(
            f"{api_url}/api/v1/identify/",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            print("✅ Request successful!")
            data = response.json()

            print("\n4. Results:")
            print(f"   Success: {data.get('success')}")
            print(f"\n   Top {len(data.get('results', []))} Predictions:")

            for i, result in enumerate(data.get('results', []), 1):
                plant_name = result.get('plant_name', 'Unknown')
                confidence = result.get('confidence', 0)
                confidence_pct = result.get('confidence_percent', confidence * 100)

                print(f"\n   {i}. {plant_name}")
                print(f"      Confidence: {confidence_pct:.2f}%")

            print("\n" + "=" * 60)
            print("✅ API test completed successfully!")
            print("=" * 60)
            return True

        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timed out after 30 seconds")
        print("   This might mean the model is loading for the first time.")
        print("   Try running the request again.")
        return False
    except Exception as e:
        print(f"❌ Error during request: {e}")
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Plant Identification API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--image",
        help="Path to plant image (optional, will create synthetic image if not provided)"
    )

    args = parser.parse_args()

    success = test_identify_endpoint(args.url, args.image)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


'''/Users/rujutabhanose/Downloads/img_9170.webp'''