#!/usr/bin/env python3
"""
Ensemble plant identification using multiple models
Combines results for higher accuracy
"""

import numpy as np
import tensorflow as tf
from transformers import pipeline
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

from image_processor_advanced import PlantImagePreprocessor

class EnsemblePlantIdentifier:
    """
    Multi-model ensemble for plant identification
    - Model 1: ViT-B (juppy44/plant-identification-2m-vit-b)
    - Model 2: ViT-L (larger, slower but more accurate)
    - Model 3: Google ViT (different training data)
    """
    
    def __init__(self):
        self.preprocessor = PlantImagePreprocessor()
        self.models = {}
        self.houseplant_db = self._load_houseplant_db()
        self.load_models()
    
    def _load_houseplant_db(self):
        """Load curated houseplant database"""
        houseplant_db = {
            'common_houseplants': {
                'Epipremnum aureum': {'common': 'Money Plant', 'family': 'Araceae', 'weight': 1.5},
                'Dracaena trifasciata': {'common': 'Snake Plant', 'family': 'Asparagaceae', 'weight': 1.5},
                'Ficus elastica': {'common': 'Rubber Plant', 'family': 'Moraceae', 'weight': 1.4},
                'Monstera deliciosa': {'common': 'Swiss Cheese Plant', 'family': 'Araceae', 'weight': 1.4},
                'Syngonium podophyllum': {'common': 'Arrowhead Plant', 'family': 'Araceae', 'weight': 1.3},
                'Tradescantia fluminensis': {'common': 'Wandering Jew', 'family': 'Commelinaceae', 'weight': 1.3},
                'Philodendron hederaceum': {'common': 'Heart Leaf Philodendron', 'family': 'Araceae', 'weight': 1.3},
                'Scindapsus pictus': {'common': 'Satin Pothos', 'family': 'Araceae', 'weight': 1.2},
                'Sansevieria trifasciata': {'common': 'Mother in law tongue', 'family': 'Asparagaceae', 'weight': 1.5},
                'Spathiphyllum wallisii': {'common': 'Peace Lily', 'family': 'Araceae', 'weight': 1.2},
            },
            'genus_weights': {
                'Epipremnum': 1.5,
                'Dracaena': 1.5,
                'Ficus': 1.4,
                'Monstera': 1.4,
                'Syngonium': 1.3,
                'Tradescantia': 1.3,
                'Philodendron': 1.3,
                'Spathiphyllum': 1.2,
            }
        }
        return houseplant_db
    
    def load_models(self):
        """Load multiple ViT models"""
        print("Loading ensemble models...")

        # Detect device: use GPU if available, otherwise CPU
        import torch
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"  Using device: {device_name}")

        # Model 1: Primary ViT-B (fast, good accuracy)
        try:
            self.models['vit_b'] = pipeline(
                "image-classification",
                model="juppy44/plant-identification-2m-vit-b",
                device=device
            )
            print("  ✓ ViT-B model loaded")
        except Exception as e:
            print(f"  ✗ ViT-B failed: {e}")

        # Model 2: PlantNet ViT (plant-specific model)
        try:
            self.models['plantnet'] = pipeline(
                "image-classification",
                model="plantnet/vit-base-patch16-224-k10",
                device=device
            )
            print("  ✓ PlantNet model loaded")
        except Exception as e:
            print(f"  ✗ PlantNet failed: {e}")
        
        # Model 3: Fallback if only one available
        if len(self.models) < 2:
            print("  ⚠️  Only one model available. Accuracy may be lower.")
    
    def _get_species_weight(self, species_name):
        """Get weight for houseplant (higher = more likely to be houseplant)"""
        
        # Check exact match
        if species_name in self.houseplant_db['common_houseplants']:
            return self.houseplant_db['common_houseplants'][species_name]['weight']
        
        # Check genus match
        genus = species_name.split()[0] if species_name else ""
        if genus in self.houseplant_db['genus_weights']:
            return self.houseplant_db['genus_weights'][genus]
        
        # Default weight for non-houseplants
        return 0.5
    
    def identify_single(self, image_path, top_k=5, debug=False):
        """
        Identify plant from image

        Args:
            image_path: Path to plant image
            top_k: Number of top predictions to return
            debug: Print debug info

        Returns:
            {
                'primary': {...},
                'alternatives': [...],
                'ensemble_confidence': float,
                'models_used': int
            }
        """
        from PIL import Image

        # Preprocess image - convert back to PIL for pipeline compatibility
        enhanced_np = self.preprocessor.enhance_image(str(image_path))
        # Convert from float32 (0-1) to uint8 (0-255) then to PIL Image
        enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
        enhanced_image = Image.fromarray(enhanced_uint8)

        if debug:
            print(f"\n📸 Processing: {image_path}")
            print(f"  Preprocessed shape: {enhanced_np.shape}")

        # Run all available models
        all_predictions = {}

        for model_name, model in self.models.items():
            try:
                predictions = model(enhanced_image)
                all_predictions[model_name] = predictions

                if debug:
                    print(f"  ✓ {model_name}: {predictions[0]['label']}")
            except Exception as e:
                if debug:
                    print(f"  ✗ {model_name} failed: {e}")
        
        # Aggregate predictions
        aggregated = self._aggregate_predictions(all_predictions, top_k)
        
        # Apply houseplant weighting (capped at 1.0 to keep confidence <= 100%)
        for pred in aggregated['top_predictions']:
            weight = self._get_species_weight(pred['label'])
            pred['weighted_score'] = min(pred['score'] * weight, 1.0)
        
        # Re-sort by weighted score
        aggregated['top_predictions'].sort(
            key=lambda x: x['weighted_score'],
            reverse=True
        )

        # Handle empty predictions
        if not aggregated['top_predictions']:
            return {
                'primary': None,
                'alternatives': [],
                'ensemble_confidence': 0.0,
                'models_used': len(all_predictions),
                'error': 'No predictions available from any model'
            }

        # Format response
        result = {
            'primary': {
                'scientific_name': aggregated['top_predictions'][0]['label'],
                'confidence': float(aggregated['top_predictions'][0]['weighted_score'] * 100),
                'common_name': self._get_common_name(aggregated['top_predictions'][0]['label'])
            },
            'alternatives': [
                {
                    'scientific_name': pred['label'],
                    'confidence': float(pred['weighted_score'] * 100),
                    'common_name': self._get_common_name(pred['label'])
                }
                for pred in aggregated['top_predictions'][1:top_k]
            ],
            'ensemble_confidence': float(aggregated['ensemble_confidence']),
            'models_used': len(all_predictions)
        }

        return result
    
    def _aggregate_predictions(self, all_predictions, top_k):
        """Aggregate predictions from all models"""
        
        # Collect all predictions
        all_preds = {}
        
        for model_name, predictions in all_predictions.items():
            for pred in predictions[:20]:  # Consider top 20 from each
                label = pred['label']
                score = pred['score']
                
                if label not in all_preds:
                    all_preds[label] = {
                        'label': label,
                        'scores': [],
                        'models': []
                    }
                
                all_preds[label]['scores'].append(score)
                all_preds[label]['models'].append(model_name)
        
        # Average scores
        for pred in all_preds.values():
            pred['score'] = np.mean(pred['scores'])
            pred['consistency'] = len(pred['models'])  # How many models predicted it
        
        # Sort by average score
        top_predictions = sorted(
            all_preds.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        # Ensemble confidence (agreement between models)
        if top_predictions:
            ensemble_confidence = top_predictions[0]['score'] * (
                top_predictions[0]['consistency'] / len(all_predictions)
            )
        else:
            ensemble_confidence = 0
        
        return {
            'top_predictions': top_predictions,
            'ensemble_confidence': ensemble_confidence
        }
    
    def _get_common_name(self, scientific_name):
        """Get common name from database"""
        if scientific_name in self.houseplant_db['common_houseplants']:
            return self.houseplant_db['common_houseplants'][scientific_name]['common']
        return scientific_name

# Flask integration
def create_ensemble_endpoint(app, identifier):
    """Add ensemble endpoint to Flask app"""
    
    @app.route('/api/v1/identify-enhanced/', methods=['POST'])
    def identify_enhanced():
        """Enhanced plant identification with ensemble"""
        
        from flask import request, jsonify
        import os
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        filepath = f"./uploads/{file.filename}"
        file.save(filepath)
        
        try:
            result = identifier.identify_single(filepath, top_k=5, debug=True)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return app

if __name__ == "__main__":
    identifier = EnsemblePlantIdentifier()
    
    # Test on sample images
    from pathlib import Path
    test_images = list(Path("./uploads").glob("*"))[:3]
    
    for img_path in test_images:
        result = identifier.identify_single(str(img_path), debug=True)
        print(f"\nResult:")
        print(f"  Primary: {result['primary']['common_name']} ({result['primary']['confidence']:.1f}%)")
        print(f"  Ensemble confidence: {result['ensemble_confidence']:.1f}%")