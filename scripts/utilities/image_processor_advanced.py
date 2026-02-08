#!/usr/bin/env python3
"""
Advanced image preprocessing to improve plant identification accuracy
- Auto-enhance contrast and brightness
- Remove background noise
- Optimize for ViT model
"""

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

class PlantImagePreprocessor:
    """Optimize plant images for ViT identification"""
    
    def __init__(self):
        self.target_size = (224, 224)
    
    def enhance_image(self, image_path, debug=False):
        """Enhanced preprocessing pipeline"""
        
        # 1. Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # 2. Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Auto-detect and remove background (optional)
        # Use HSV color space to isolate plant (green channel)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Green color range (plant foliage)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Dilate mask to connect small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # Apply mask to isolate plant (but keep some context)
        if np.sum(mask) > 0:  # If plant detected
            # Blend: 70% masked, 30% original (keep context)
            image_masked = cv2.bitwise_and(image, image, mask=mask)
            image_context = image.copy()
            alpha = 0.7
            image = cv2.addWeighted(image_masked, alpha, image_context, 1 - alpha, 0)
        
        # 4. Enhance contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        lab = cv2.merge([l_channel, a_channel, b_channel])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 5. Adaptive brightness normalization
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        # Target brightness is 127 (middle)
        if mean_brightness < 80:  # Too dark
            image = cv2.convertScaleAbs(image, alpha=1.4, beta=20)
        elif mean_brightness > 180:  # Too bright
            image = cv2.convertScaleAbs(image, alpha=0.9, beta=-20)
        
        # 6. Sharpen to enhance leaf details
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]]) / 1.5
        image = cv2.filter2D(image, -1, kernel_sharpen)
        
        # 7. Resize to target
        image = cv2.resize(image, self.target_size)
        
        # 8. Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def batch_enhance(self, image_paths, debug=False):
        """Enhance multiple images"""
        images = []
        for path in image_paths:
            try:
                img = self.enhance_image(path, debug=debug)
                images.append(img)
            except Exception as e:
                print(f"⚠️  Error processing {path}: {e}")
        
        return np.array(images)

def test_preprocessing():
    """Test the preprocessor"""
    preprocessor = PlantImagePreprocessor()
    
    # Test on first available image
    from pathlib import Path
    test_images = list(Path("./uploads").glob("*"))
    
    if test_images:
        img = preprocessor.enhance_image(str(test_images))
        print(f"✓ Preprocessed image shape: {img.shape}")
        print(f"  Min value: {img.min():.3f}, Max value: {img.max():.3f}")
    else:
        print("No test images found")

if __name__ == "__main__":
    test_preprocessing()