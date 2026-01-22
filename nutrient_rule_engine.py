"""
Rule-Based Nutrient Detection System
- Works for ANY plant leaf
- Detects visual symptoms (colors, patterns, shapes)
- Independent of model training
- Used to boost neural network confidence
"""

import cv2
import numpy as np
from pathlib import Path
import json

# ============================================================================
# SYMPTOM DETECTION ENGINE
# ============================================================================

class NutrientRuleEngine:
    """
    Rule-based system for detecting nutrient deficiencies
    Works for any plant leaf by detecting color/shape/pattern changes
    """
    
    def __init__(self):
        self.symptom_rules = self._initialize_rules()
    
    def _initialize_rules(self):
        """
        Define symptom detection rules
        Each rule: visual indicator -> likely deficiency
        """
        return {
            'nitrogen': {
                'indicators': ['yellowing', 'pale_leaves', 'light_green'],
                'confidence_boost': 1.4,
                'description': 'Yellowing of older leaves, light green color throughout'
            },
            'phosphorus': {
                'indicators': ['purpling', 'dark_red', 'dark_green'],
                'confidence_boost': 1.3,
                'description': 'Purple/reddish discoloration, dark veins'
            },
            'potassium': {
                'indicators': ['brown_edges', 'leaf_margin_burn', 'bronze_spots'],
                'confidence_boost': 1.35,
                'description': 'Brown/burned leaf edges, scorch marks'
            },
            'magnesium': {
                'indicators': ['interveinal_yellowing', 'green_veins', 'yellow_areas'],
                'confidence_boost': 1.32,
                'description': 'Yellow between green veins, veins stay green'
            },
            'calcium': {
                'indicators': ['distorted_growth', 'necrotic_spots', 'leaf_curl'],
                'confidence_boost': 1.28,
                'description': 'Curled new leaves, distorted growth, brown spots'
            },
            'iron': {
                'indicators': ['interveinal_yellowing_severe', 'white_areas', 'light_leaves'],
                'confidence_boost': 1.35,
                'description': 'Severe yellowing between veins on young leaves'
            },
            'manganese': {
                'indicators': ['brown_spots', 'grey_spots', 'checkered_pattern'],
                'confidence_boost': 1.30,
                'description': 'Brown/grey spots, tan/grey necrotic areas'
            },
            'boron': {
                'indicators': ['leaf_crinkle', 'distorted_leaf', 'rosette_growth'],
                'confidence_boost': 1.25,
                'description': 'Crinkled, distorted leaves, unusual rosette growth'
            },
            'healthy': {
                'indicators': ['uniform_green', 'no_spots', 'normal_texture'],
                'confidence_boost': 1.0,
                'description': 'Normal green color, no spots or discoloration'
            }
        }
    
    def detect_yellowing(self, hsv_image):
        """
        Detect yellowing/pale green color
        Yellow in HSV: H=20-40, S=50-255, V=50-255
        """
        lower = np.array([20, 50, 50])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv_image, lower, upper)
        yellow_ratio = cv2.countNonZero(mask) / mask.size
        return yellow_ratio
    
    def detect_purpling(self, hsv_image):
        """
        Detect purple/reddish discoloration
        Purple in HSV: H=270-330 or H=0-20, high saturation
        """
        # Red range (around 0)
        lower1 = np.array([0, 100, 50])
        upper1 = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        
        # Purple range
        lower2 = np.array([270, 100, 50])
        upper2 = np.array([330, 255, 255])
        # Convert to HSV range (0-179 for OpenCV)
        lower2 = np.array([135, 100, 50])
        upper2 = np.array([165, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower2, upper2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        purple_ratio = cv2.countNonZero(mask) / mask.size
        return purple_ratio
    
    def detect_brown_edges(self, hsv_image):
        """
        Detect brown discoloration/burned edges
        Brown in HSV: H=10-30, S=50-255, V=50-150
        """
        lower = np.array([10, 50, 50])
        upper = np.array([30, 255, 150])
        mask = cv2.inRange(hsv_image, lower, upper)
        brown_ratio = cv2.countNonZero(mask) / mask.size
        return brown_ratio
    
    def detect_green_veins(self, image_rgb):
        """
        Detect interveinal yellowing (green veins on yellow background)
        """
        # Convert to LAB color space (better for this)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Green has low a value (greenness)
        # Detect veins by morphological operations
        _, veins = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
        
        vein_ratio = cv2.countNonZero(veins) / veins.size
        return vein_ratio
    
    def detect_brown_spots(self, image_rgb):
        """
        Detect brown/grey spots on leaf
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Brown spots
        lower = np.array([10, 30, 50])
        upper = np.array([30, 255, 200])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations to find distinct spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        spots = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        spot_ratio = cv2.countNonZero(spots) / spots.size
        return spot_ratio
    
    def detect_texture_distortion(self, image_rgb):
        """
        Detect crinkled/distorted leaf texture
        Uses Laplacian variance (measure of edge sharpness)
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        return texture_variance
    
    def detect_health(self, image_rgb):
        """
        Detect healthy leaf: uniform green, good texture
        """
        # Measure color uniformity
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Healthy leaves have:
        # - H around 60-90 (green range)
        # - S moderate (not washed out)
        # - V good (bright)
        
        green_hue = np.mean((h >= 40) & (h <= 100))
        saturation = np.mean(s) / 255
        brightness = np.mean(v) / 255
        
        health_score = (green_hue + saturation + brightness) / 3
        return health_score
    
    def predict(self, image_path_or_array, verbose=False):
        """
        Main prediction function
        
        Args:
            image_path_or_array: file path (str) or numpy array (image)
            verbose: print diagnostic information
        
        Returns:
            dict with deficiency predictions and confidence
        """
        # Load image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array
        
        if image is None:
            return {'error': 'Could not load image'}
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect all symptoms
        symptoms = {
            'yellowing': self.detect_yellowing(hsv),
            'purpling': self.detect_purpling(hsv),
            'brown_edges': self.detect_brown_edges(hsv),
            'green_veins': self.detect_green_veins(image),
            'brown_spots': self.detect_brown_spots(image),
            'texture_distortion': self.detect_texture_distortion(image),
            'health': self.detect_health(image),
        }
        
        if verbose:
            print("\n[SYMPTOM ANALYSIS]")
            for symptom, value in symptoms.items():
                print(f"  {symptom:20s}: {value:.3f}")
        
        # Score each deficiency
        scores = {}
        
        # Nitrogen: yellowing + pale
        scores['N'] = (symptoms['yellowing'] * 0.7 + 
                      (1 - symptoms['health']) * 0.3) * 100
        
        # Phosphorus: purpling
        scores['P'] = symptoms['purpling'] * 100
        
        # Potassium: brown edges
        scores['K'] = (symptoms['brown_edges'] * 0.8 + 
                      symptoms['brown_spots'] * 0.2) * 100
        
        # Magnesium: interveinal yellowing
        scores['Mg'] = (symptoms['yellowing'] * 0.6 + 
                       symptoms['green_veins'] * 0.4) * 100
        
        # Calcium: distorted growth
        scores['Ca'] = (symptoms['texture_distortion'] * 0.6 + 
                       symptoms['brown_spots'] * 0.4) * 100
        
        # Iron: severe yellowing on new leaves
        scores['Fe'] = symptoms['yellowing'] * 0.8 * 100
        
        # Manganese: brown spots
        scores['Mn'] = symptoms['brown_spots'] * 100
        
        # Boron: leaf crinkle/distortion
        scores['B'] = symptoms['texture_distortion'] * 100
        
        # Healthy: no symptoms
        scores['Healthy'] = symptoms['health'] * 100
        
        if verbose:
            print("\n[DEFICIENCY SCORES]")
            for deficiency, score in sorted(scores.items(), key=lambda x: x, reverse=True):
                print(f"  {deficiency:10s}: {score:6.1f}")
        
        return {
            'symptoms': symptoms,
            'scores': scores,
            'top_deficiency': max(scores, key=scores.get),
            'top_score': max(scores.values()),
        }

# ============================================================================
# INTEGRATION WITH MODEL
# ============================================================================

def boost_model_confidence(model_output, rule_engine_output):
    """
    Combine neural network output with rule-based system
    
    Args:
        model_output: dict with 'class' and 'confidence' from NN
        rule_engine_output: dict from rule engine
    
    Returns:
        boosted confidence
    """
    model_class = model_output['class']
    model_confidence = model_output['confidence']
    rule_score = rule_engine_output['scores'].get(model_class, 0)
    
    # If rule engine agrees, boost confidence
    if rule_score > 20:
        boost_factor = 1.0 + (rule_score / 100.0 * 0.3)  # Up to 30% boost
        boosted_confidence = min(model_confidence * boost_factor, 0.99)
        return boosted_confidence, f"Rule engine confirmed: {rule_score:.0f}/100"
    
    # If rule engine disagrees, reduce confidence
    else:
        return model_confidence * 0.8, "Rule engine suggests otherwise"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    engine = NutrientRuleEngine()
    
    # Test on image
    test_image_path = "./uploads/test_leaf.jpg"
    
    if Path(test_image_path).exists():
        result = engine.predict(test_image_path, verbose=True)
        print("\n[FINAL PREDICTION]")
        print(f"  Top deficiency: {result['top_deficiency']}")
        print(f"  Confidence: {result['top_score']:.1f}")
    else:
        print(f"Test image not found: {test_image_path}")