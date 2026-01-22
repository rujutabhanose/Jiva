#!/usr/bin/env python3
"""
Model Evaluation Script for Plant Disease Detection
Supports both single-task and multi-task models
"""

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Optional imports
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class ModelEvaluator:
    """Evaluate trained plant disease detection models"""

    def __init__(self, model_path, class_mapping_path=None, input_size=(224, 224)):
        """
        Initialize evaluator

        Args:
            model_path: Path to saved .h5 or .keras model
            class_mapping_path: Path to class mapping JSON (optional)
            input_size: Input image size (width, height)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.model = None
        self.class_mapping = None
        self.is_multi_task = False

        # Load model
        self._load_model()

        # Load class mapping if provided
        if class_mapping_path:
            with open(class_mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
                print(f"✓ Loaded class mapping with {self.class_mapping['num_classes']} classes")

    def _load_model(self):
        """Load the trained model"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)

            # Check if multi-task model
            if len(self.model.outputs) > 1:
                self.is_multi_task = True
                print(f"✓ Loaded multi-task model with {len(self.model.outputs)} outputs")
            else:
                print("✓ Loaded single-task model")

            self.model.summary()

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def preprocess_image(self, image_path):
        """Preprocess single image for prediction"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')

            # Resize
            img = img.resize(self.input_size)

            # Convert to array and normalize
            img_array = np.array(img) / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_label_from_filename(self, filename):
        """
        Extract expected label from filename
        Example: "PotatoEarlyBlight1.JPG" -> "Potato___Early_blight"
        """
        # Remove extension and number
        name = Path(filename).stem

        # Remove trailing numbers
        import re
        name = re.sub(r'\d+$', '', name)

        # Try to match to class mapping
        if self.class_mapping:
            # Try exact match first
            for class_name in self.class_mapping['classes']:
                simplified_class = class_name.replace('___', '').replace('_', '').replace(',', '').replace(' ', '').lower()
                simplified_file = name.replace('_', '').replace('-', '').lower()

                if simplified_file in simplified_class or simplified_class in simplified_file:
                    return class_name

        return name

    def evaluate_on_directory(self, test_dir):
        """
        Evaluate model on all images in a directory

        Args:
            test_dir: Directory containing test images (flat structure with labeled filenames)
        """
        test_dir = Path(test_dir)

        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_dir.glob(f'*{ext}'))

        if not image_files:
            print(f"❌ No images found in {test_dir}")
            return

        print(f"\n📊 Evaluating on {len(image_files)} images...")

        # Storage for results
        predictions = []
        true_labels = []
        filenames = []

        # Process each image
        for img_path in sorted(image_files):
            # Preprocess
            img_array = self.preprocess_image(img_path)
            if img_array is None:
                continue

            # Predict
            pred = self.model.predict(img_array, verbose=0)

            # Handle multi-task vs single-task
            if self.is_multi_task:
                # Use disease classification head (first output)
                pred = pred[0]

            pred_class = np.argmax(pred[0])
            confidence = float(np.max(pred[0]))

            # Extract true label from filename
            true_label = self.extract_label_from_filename(img_path.name)

            # Store results
            predictions.append((pred_class, confidence))
            true_labels.append(true_label)
            filenames.append(img_path.name)

            # Print individual result
            if self.class_mapping and str(pred_class) in self.class_mapping['index_to_name']:
                pred_label = self.class_mapping['index_to_name'][str(pred_class)]
            else:
                pred_label = f"Class_{pred_class}"

            match = "✓" if pred_label == true_label else "✗"
            print(f"{match} {img_path.name}: {pred_label} ({confidence:.2%})")

        # Calculate metrics
        self._print_metrics(predictions, true_labels, filenames)

        return predictions, true_labels, filenames

    def _print_metrics(self, predictions, true_labels, filenames):
        """Calculate and print evaluation metrics"""

        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)

        pred_classes = [p[0] for p in predictions]
        confidences = [p[1] for p in predictions]

        # Map true labels to indices
        true_indices = []
        if self.class_mapping:
            for label in true_labels:
                if label in self.class_mapping['name_to_index']:
                    true_indices.append(int(self.class_mapping['name_to_index'][label]))
                else:
                    true_indices.append(-1)  # Unknown

        # Calculate accuracy
        if true_indices and -1 not in true_indices:
            correct = sum(p == t for p, t in zip(pred_classes, true_indices))
            accuracy = correct / len(pred_classes)
            print(f"\n✓ Overall Accuracy: {accuracy:.2%} ({correct}/{len(pred_classes)})")
        else:
            print("\n⚠️  Cannot calculate accuracy (unknown true labels)")

        # Average confidence
        avg_confidence = np.mean(confidences)
        print(f"✓ Average Confidence: {avg_confidence:.2%}")

        # Per-class breakdown
        if self.class_mapping and true_indices and -1 not in true_indices:
            print("\n" + "-"*60)
            print("PER-CLASS PERFORMANCE")
            print("-"*60)

            unique_classes = sorted(set(true_indices))
            for class_idx in unique_classes:
                class_name = self.class_mapping['index_to_name'].get(str(class_idx), f"Class_{class_idx}")

                # Find all instances of this class
                class_preds = [p for p, t in zip(pred_classes, true_indices) if t == class_idx]
                class_correct = sum(p == class_idx for p in class_preds)

                if class_preds:
                    class_acc = class_correct / len(class_preds)
                    print(f"{class_name}: {class_acc:.2%} ({class_correct}/{len(class_preds)})")

        # Top-5 most confident predictions
        print("\n" + "-"*60)
        print("TOP 5 MOST CONFIDENT PREDICTIONS")
        print("-"*60)

        sorted_by_conf = sorted(zip(confidences, filenames, pred_classes), reverse=True)
        for conf, fname, pred_class in sorted_by_conf[:5]:
            if self.class_mapping and str(pred_class) in self.class_mapping['index_to_name']:
                pred_label = self.class_mapping['index_to_name'][str(pred_class)]
            else:
                pred_label = f"Class_{pred_class}"
            print(f"{fname}: {pred_label} ({conf:.2%})")

        # Bottom-5 least confident predictions
        print("\n" + "-"*60)
        print("TOP 5 LEAST CONFIDENT PREDICTIONS")
        print("-"*60)

        for conf, fname, pred_class in sorted_by_conf[-5:]:
            if self.class_mapping and str(pred_class) in self.class_mapping['index_to_name']:
                pred_label = self.class_mapping['index_to_name'][str(pred_class)]
            else:
                pred_label = f"Class_{pred_class}"
            print(f"{fname}: {pred_label} ({conf:.2%})")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained plant disease detection model"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model file (.h5 or .keras)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        required=True,
        help='Directory containing test images'
    )
    parser.add_argument(
        '--class-mapping',
        type=str,
        default='class_mapping.json',
        help='Path to class mapping JSON file'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Input image size (width height)'
    )

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("\nAvailable models in ./models/:")
        models_dir = Path("./models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.h5"):
                print(f"  - {model_file.name}")
            for model_file in models_dir.glob("*.keras"):
                print(f"  - {model_file.name}")
        return

    if not Path(args.test_dir).exists():
        print(f"❌ Test directory not found: {args.test_dir}")
        return

    # Check class mapping
    class_mapping_path = args.class_mapping if Path(args.class_mapping).exists() else None
    if not class_mapping_path:
        print("⚠️  Class mapping file not found, using indices only")

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        class_mapping_path=class_mapping_path,
        input_size=tuple(args.input_size)
    )

    # Run evaluation
    print("\n" + "="*60)
    print(f"EVALUATION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    evaluator.evaluate_on_directory(args.test_dir)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
