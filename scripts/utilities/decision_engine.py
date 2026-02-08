"""
Production-ready decision engine with rules
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict

class DiagnosticDecisionEngine:
    """Orchestrates TFLite + Ensemble + Rules."""
    
    def __init__(self, kb_path: Path, calibrators_path: Path, weights_path: Path):
        with open(kb_path) as f:
            self.kb = json.load(f)
        with open(calibrators_path, 'rb') as f:
            self.calibrators = pickle.load(f)
        with open(weights_path) as f:
            self.weights = json.load(f)
    
    def diagnose(self, plant: str, tflite_probs: np.ndarray,
                 ensemble_probs: np.ndarray, class_names: List[str],
                 climate: str = None) -> Dict:
        """
        Main diagnosis logic.
        
        Returns: {
            "disease": str,
            "confidence": float (0-1),
            "action": "ACCEPT" | "ASK_QA" | "DEFER",
            "treatment": {...},
            "reasoning": str
        }
        """
        
        # Step 1: Calibrate
        cal_tflite = self._calibrate(tflite_probs, 'tflite')
        cal_ensemble = self._calibrate(ensemble_probs, 'ensemble')
        
        # Step 2: Weighted average
        w_tflite = self.weights.get('tflite', 0.3)
        w_ensemble = self.weights.get('ensemble', 0.7)
        
        scores = w_tflite * cal_tflite + w_ensemble * cal_ensemble
        
        # Step 3: Get top 3
        top_indices = np.argsort(scores)[-3:][::-1]
        top_diseases = [class_names[i] for i in top_indices]
        top_scores = scores[top_indices].copy()
        
        # Step 4: Apply rules
        for i, disease in enumerate(top_diseases):
            top_scores[i] = self._apply_rules(plant, disease, top_scores[i], climate)
        
        # Step 5: Re-rank
        ranked = sorted(zip(top_diseases, top_scores), key=lambda x: x[1], reverse=True)
        top_disease, top_conf = ranked[0]
        
        # Step 6: Decision
        if top_conf >= 0.85:
            action = "ACCEPT"
            level = "HIGH"
        elif top_conf >= 0.70:
            action = "ACCEPT"
            level = "MEDIUM"
        elif top_conf >= 0.60:
            action = "ASK_QA"
            level = "LOW"
        else:
            action = "DEFER"
            level = "VERY_LOW"
        
        # Step 7: Get treatment
        treatment = self.kb.get('treatments', {}).get(top_disease, {})
        
        return {
            "disease": top_disease,
            "confidence": float(top_conf),
            "confidence_level": level,
            "action": action,
            "treatment": treatment,
            "top_3": [{"disease": d, "score": float(s)} for d, s in ranked],
            "reasoning": f"{plant} → {top_disease} ({top_conf*100:.0f}%)"
        }
    
    def _calibrate(self, probs, model_type):
        """Apply isotonic regression."""
        cal = np.zeros_like(probs)
        for i in range(len(probs)):
            key = f'class_{i}'
            if key in self.calibrators:
                cal[i] = self.calibrators[key].predict([probs[i]])[0]
            else:
                cal[i] = probs[i]
        return cal
    
    def _apply_rules(self, plant, disease, conf, climate=None):
        """Apply plausibility rules."""
        plant_data = self.kb.get('plants', {}).get(plant, {})
        
        # Rule: Disease not common on this plant
        if disease not in plant_data.get('common_diseases', []):
            conf *= 0.3
        
        # Rule: Climate incompatible
        if climate and plant_data.get('climate'):
            if climate not in plant_data['climate']:
                conf *= 0.8
        
        return np.clip(conf, 0, 1)