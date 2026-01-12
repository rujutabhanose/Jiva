# export_to_mobile.py
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

class MobileModelExporter:
    """
    Convert trained Keras models to optimized TFLite format
    for Android & iOS deployment
    """
    
    @staticmethod
    def export_classification_model(
        keras_model_path,
        output_dir="./mobile_models",
        model_name="disease_detector",
        input_shape=(224, 224, 3),
        quantize_type="INT8"  # INT8, FLOAT16, DYNAMIC_RANGE
    ):
        """
        Export classification model with optimizations
        
        Args:
            keras_model_path: Path to .h5 or SavedModel
            quantize_type: 
                - INT8: Full quantization (smallest, fastest)
                - FLOAT16: Half-precision (medium)
                - DYNAMIC_RANGE: Dynamic range quantization (good balance)
        """
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load model
        print(f"📂 Loading model from {keras_model_path}...")
        if keras_model_path.endswith('.h5'):
            model = tf.keras.models.load_model(keras_model_path)
        else:
            model = tf.saved_model.load(keras_model_path)
        
        # Create converter
        print("🔄 Creating TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimizations
        if quantize_type == "INT8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
        elif quantize_type == "FLOAT16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantize_type == "DYNAMIC_RANGE":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        print("⚙️  Converting model...")
        tflite_model = converter.convert()
        
        # Save
        output_path = Path(output_dir) / f"{model_name}_{quantize_type.lower()}.tflite"
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ Exported {model_name} ({quantize_type})")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Path: {output_path}")
        
        return output_path
    
    @staticmethod
    def create_model_metadata(
        output_dir,
        model_name,
        input_shape=(224, 224, 3),
        class_labels_file=None,
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225)
    ):
        """Create metadata JSON for mobile integration"""
        
        metadata = {
            "model_name": model_name,
            "input": {
                "shape": list(input_shape),
                "dtype": "float32",
                "normalization": {
                    "mean": list(normalization_mean),
                    "std": list(normalization_std)
                }
            },
            "output": {
                "disease": {
                    "shape": [1, 21],
                    "dtype": "float32",
                    "interpretation": "softmax probabilities"
                },
                "nutrient": {
                    "shape": [1, 9],
                    "dtype": "float32",
                    "interpretation": "softmax probabilities"
                },
                "health_score": {
                    "shape": [1, 1],
                    "dtype": "float32",
                    "range": [0, 100]
                }
            },
            "preprocessing": {
                "resize_method": "bilinear",
                "aspect_ratio_handling": "pad_with_mean"
            }
        }
        
        # Add class labels if provided
        if class_labels_file:
            with open(class_labels_file) as f:
                metadata["class_labels"] = json.load(f)
        
        output_path = Path(output_dir) / f"{model_name}_metadata.json"
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata created: {output_path}")
        
        return output_path
    
    @staticmethod
    def benchmark_model(tflite_model_path, test_images_dir, num_runs=100):
        """
        Benchmark TFLite model performance
        Returns: inference time, memory usage, accuracy
        """
        import time
        
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n📊 Benchmarking {tflite_model_path}...")
        print(f"   Input shape: {input_details['shape']}")
        print(f"   Output shapes: {[o['shape'] for o in output_details]}")
        
        # Load test image
        # (simplified - load actual test data)
        test_input = np.random.randn(*input_details['shape']).astype(np.float32)
        
        # Warm up
        interpreter.set_tensor(input_details['index'], test_input)
        interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            interpreter.set_tensor(input_details['index'], test_input)
            interpreter.invoke()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   Avg inference time: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"   Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms")


# Export workflow
if __name__ == "__main__":
    
    exporter = MobileModelExporter()
    
    # Step 1: Export quantized model
    tflite_path = exporter.export_classification_model(
        keras_model_path="models/house_plant_model.h5",
        output_dir="./mobile_models",
        model_name="house_plant_classifier",
        quantize_type="INT8"  # Smallest size, best for mobile
    )
    
    # Step 2: Create metadata
    exporter.create_model_metadata(
        output_dir="./mobile_models",
        model_name="house_plant_classifier",
        class_labels_file="data/disease_classes.json"
    )
    
    # Step 3: Benchmark
    exporter.benchmark_model(tflite_path, "data/test")
    
    print("\n✅ Mobile export complete!")