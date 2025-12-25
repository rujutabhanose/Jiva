# Plant Disease Diagnosis Model Integration

## Overview

The Jiva backend uses a MobileNetV2-based plant disease classification model with a custom agronomical mapping layer to provide accurate plant health diagnoses.

## Model Details

- **Model**: [linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification)
- **Architecture**: MobileNetV2
- **Framework**: TensorFlow/Keras via HuggingFace Transformers
- **Input**: 224x224 RGB images
- **Output**: Classification scores for various plant diseases and deficiencies

## Architecture

```
┌─────────────────────┐
│  Input Image        │
│  (Plant Photo)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  MobileNetV2        │
│  Classification     │
│  Model              │
└──────────┬──────────┘
           │
           │ Raw Labels + Confidence
           ▼
┌─────────────────────┐
│  Mapping Layer      │
│  (disease_mapping)  │
│                     │
│  - Label            │
│    Normalization    │
│  - Category         │
│    Assignment       │
│  - Agronomy Data    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Structured         │
│  Diagnosis          │
│                     │
│  - Name             │
│  - Symptoms         │
│  - Causes           │
│  - Treatment        │
│  - Severity         │
└─────────────────────┘
```

## Supported Categories

### Nutritional Deficiencies

The system maps model outputs to these nutrient deficiencies:

| Nutrient | Symbol | Key Symptoms |
|----------|--------|--------------|
| Nitrogen | N | Yellowing of older leaves, stunted growth |
| Phosphorus | P | Purple/reddish discoloration, dark leaves |
| Potassium | K | Brown leaf margins, weak stems |
| Iron | Fe | Interveinal chlorosis on young leaves |
| Magnesium | Mg | Interveinal chlorosis on older leaves |
| Calcium | Ca | Tip burn, blossom end rot |

### Disease Types

| Category | Examples | Key Characteristics |
|----------|----------|---------------------|
| Fungal | Powdery Mildew, Early Blight, Late Blight, Leaf Spot | White coating, spots, rapid spread |
| Bacterial | Bacterial Spot, Bacterial Wilt, Bacterial Blight | Water-soaked lesions, wilting, ooze |
| Viral | (Future expansion) | Mottling, distortion |

## Mapping Logic

The mapping layer (`disease_mapping.py`) provides:

1. **Label Normalization**: Converts model output labels to standard format
   ```python
   "N_deficiency" → "nitrogen_deficiency"
   "powdery" → "powdery_mildew"
   ```

2. **Structured Information**: Each condition maps to:
   - **Category**: nutrient_deficiency, fungal, bacterial, viral
   - **Subcategory**: Specific nutrient (N, P, K, etc.) or disease name
   - **Symptoms**: List of observable signs
   - **Causes**: Root causes and contributing factors
   - **Treatment**: Step-by-step remediation actions

3. **Severity Assessment**: Based on confidence scores:
   - **Mild**: <50% confidence
   - **Moderate**: 50-80% confidence
   - **Severe**: >80% confidence

## Diagnosis Flow

### 1. Image Upload
```python
POST /api/v1/diagnose/
Content-Type: multipart/form-data
file: <plant_image.jpg>
```

### 2. Model Inference
```python
predictions = model(image, top_k=5)
# Returns: [
#   {"label": "nitrogen_deficiency", "score": 0.85},
#   {"label": "powdery_mildew", "score": 0.12},
#   ...
# ]
```

### 3. Mapping to Agronomy Data
```python
for prediction in predictions:
    diagnosis_info = get_diagnosis_info(prediction['label'])
    # Returns structured DiagnosisInfo with symptoms, causes, treatment
```

### 4. Response Generation
```json
{
  "success": true,
  "plant_health_score": 72.5,
  "summary": {
    "nutrient_deficiencies": [
      {"nutrient": "N", "name": "Nitrogen Deficiency", "confidence": 85.0}
    ],
    "diseases": [],
    "total_issues": 1
  },
  "diagnoses": [{
    "name": "Nitrogen Deficiency",
    "category": "nutrient_deficiency",
    "subcategory": "N",
    "confidence": 85.0,
    "severity": "severe",
    "symptoms": [
      "Yellowing of older leaves (chlorosis) starting from leaf tips",
      "Stunted growth and smaller leaves",
      ...
    ],
    "causes": [
      "Poor soil fertility or depleted nitrogen levels",
      ...
    ],
    "treatment": [
      "Apply nitrogen-rich fertilizers (urea, ammonium nitrate, or blood meal)",
      "Use organic compost or well-rotted manure",
      ...
    ]
  }],
  "recommendations": [
    "PRIMARY ACTION: Apply nitrogen-rich fertilizers",
    "Apply balanced fertilizer to address N deficiencies",
    ...
  ]
}
```

## Confidence Threshold

Default: **0.3 (30%)**

- Lower threshold = More sensitive, may include false positives
- Higher threshold = More specific, may miss subtle issues
- Adjustable per request via `confidence_threshold` parameter

## Health Score Calculation

```python
health_score = 100 - Σ(confidence × severity_mult × category_mult × 15)
```

Where:
- **Severity Multipliers**: mild=0.5, moderate=1.0, severe=1.5
- **Category Multipliers**:
  - Nutrient deficiency: 1.0
  - Fungal: 1.3
  - Bacterial: 1.5
  - Viral: 1.7

## Adding New Conditions

To add a new disease or deficiency mapping:

1. Open `backend/app/services/disease_mapping.py`
2. Add entry to `DISEASE_MAPPINGS` dictionary:

```python
"your_condition_label": DiagnosisInfo(
    category="fungal",  # or nutrient_deficiency, bacterial, etc.
    subcategory="your_disease_name",
    name="Display Name",
    symptoms=[
        "Symptom 1",
        "Symptom 2",
        ...
    ],
    causes=[
        "Cause 1",
        "Cause 2",
        ...
    ],
    treatment=[
        "Treatment step 1",
        "Treatment step 2",
        ...
    ],
    severity_indicators={
        "mild": "Description",
        "moderate": "Description",
        "severe": "Description"
    }
),
```

3. Add label aliases if needed in `LABEL_ALIASES`

## API Endpoints

### Diagnose Plant
```
POST /api/v1/diagnose/
```
Upload image and get diagnosis

### Get Categories
```
GET /api/v1/diagnose/categories
```
List all supported categories

### Health Check
```
GET /api/v1/diagnose/health
```
Check if model is loaded and ready

## Model Loading

The model loads automatically on server startup. First-time loading will download the model from HuggingFace (~14MB).

**Loading time**: 5-10 seconds (first run), <1 second (subsequent)

## Performance

- **Inference time**: ~200-500ms per image (CPU)
- **Memory usage**: ~500MB (model in memory)
- **GPU acceleration**: Supported (automatic if CUDA available)

## Troubleshooting

### Model Not Loading
```
Error: Model not loaded. Please check model initialization.
```
**Solution**: Check internet connection for first-time download, verify `transformers` and `torch` are installed

### Low Confidence Scores
```
message: "No confident predictions found"
```
**Solution**:
- Ensure image shows clear symptoms
- Try better lighting and focus
- Lower confidence_threshold parameter

### Unmapped Labels
```
warning: "Unmapped label detected: unknown_disease"
```
**Solution**: Add mapping for this label in `disease_mapping.py`

## References

- Model source: [HuggingFace Model Card](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification)
- Plant pathology: Standard agronomy textbooks and extension service guidelines
- Nutrient deficiency symptoms: Agricultural university extension publications
