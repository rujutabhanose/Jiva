"""
Load trained weights from the production model into the fixed architecture.
"""
import tensorflow as tf
import h5py
import numpy as np
from pathlib import Path

print("=" * 60)
print("LOADING TRAINED WEIGHTS INTO FIXED MODEL")
print("=" * 60)

MODELS_DIR = Path("models")
FIXED_MODEL = MODELS_DIR / "coleaf_production_v2_fixed.keras"
TRAINED_WEIGHTS = MODELS_DIR / "coleaf_production_v2.keras"  # Has architecture issue but contains trained weights

print("\nStep 1: Load fixed architecture...")
model = tf.keras.models.load_model(str(FIXED_MODEL))
print(f"✓ Fixed model loaded")
print(f"  Layers: {len(model.layers)}")

print("\nStep 2: Extract weights from trained model file...")
try:
    # Open the H5 file directly to extract weights
    with h5py.File(str(TRAINED_WEIGHTS), 'r') as f:
        # Check if model_weights group exists
        if 'model_weights' in f:
            print("  Found model_weights group")
            weights_group = f['model_weights']
            
            # List all weight groups
            def print_structure(group, indent=0):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        print("  " * indent + f"Group: {key}")
                        print_structure(item, indent + 1)
                    else:
                        print("  " * indent + f"Dataset: {key} - shape {item.shape}")
            
            print("\n  Weight structure:")
            print_structure(weights_group)
            
            # Try to load weights by matching layer names
            print("\nStep 3: Loading weights by layer matching...")
            loaded_count = 0
            
            for layer in model.layers:
                layer_name = layer.name
                print(f"  Processing layer: {layer_name}")
                
                try:
                    # Try to find matching weights in the file
                    if layer_name in weights_group:
                        layer_weights_group = weights_group[layer_name]
                        
                        # Get weight arrays
                        weight_arrays = []
                        for weight_name in layer_weights_group.keys():
                            weight_data = layer_weights_group[weight_name][:]
                            weight_arrays.append(weight_data)
                            print(f"    Loaded {weight_name}: {weight_data.shape}")
                        
                        # Set weights to layer
                        if weight_arrays:
                            layer.set_weights(weight_arrays)
                            loaded_count += 1
                            print(f"    ✓ Weights set for {layer_name}")
                except Exception as e:
                    print(f"    ⚠ Could not load weights for {layer_name}: {e}")
            
            print(f"\n✓ Loaded weights for {loaded_count} layers")
            
        else:
            print("  ⚠ No model_weights group found in file")
            print(f"  Available keys: {list(f.keys())}")
            
except Exception as e:
    print(f"  ✗ Error reading weights file: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 4: Saving model with trained weights...")
FINAL_MODEL = MODELS_DIR / "coleaf_production_v2_working.keras"
model.save(str(FINAL_MODEL))
print(f"✓ Saved: {FINAL_MODEL}")

print("\nStep 5: Testing final model...")
test_model = tf.keras.models.load_model(str(FINAL_MODEL))
print("✓ Model loads successfully!")

test_img = np.random.rand(1, 224, 224, 3).astype('float32')
pred = test_model.predict(test_img, verbose=0)
print(f"✓ Prediction works! Shape: {pred.shape}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
