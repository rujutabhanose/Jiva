# Testing the Plant Diagnosis System

## Quick Start

### 1. Start the Backend Server

```bash
cd backend
python -m uvicorn app.main:app --reload
```

The model will load automatically on startup. Wait for:
```
INFO:     Plant disease model loaded successfully
INFO:     Application startup complete.
```

### 2. Check Model Health

```bash
curl http://localhost:8000/api/v1/diagnose/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "Diagnosis engine ready",
  "ready": true,
  "model": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
}
```

### 3. Get Available Categories

```bash
curl http://localhost:8000/api/v1/diagnose/categories
```

Response:
```json
{
  "nutrient_deficiencies": ["N", "P", "K", "Fe", "Mg", "Ca"],
  "disease_types": ["fungal", "bacterial", "viral"]
}
```

## Testing Diagnosis

### Using cURL

```bash
curl -X POST http://localhost:8000/api/v1/diagnose/ \
  -F "file=@/path/to/plant_image.jpg" \
  -F "confidence_threshold=0.3" \
  -F "top_k=5"
```

### Using Python Requests

```python
import requests

# Test diagnosis
with open('plant_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/diagnose/',
        files={'file': f},
        data={
            'confidence_threshold': 0.3,
            'top_k': 5
        }
    )

result = response.json()
print(f"Health Score: {result['plant_health_score']}")
print(f"Issues Found: {result['summary']['total_issues']}")

for diagnosis in result['diagnoses']:
    print(f"\n{diagnosis['name']} ({diagnosis['confidence']}%)")
    print(f"  Category: {diagnosis['category']}")
    print(f"  Severity: {diagnosis['severity']}")
    print(f"  Top Treatment: {diagnosis['treatment'][0]}")
```

### Using HTTPie

```bash
http -f POST http://localhost:8000/api/v1/diagnose/ \
  file@plant_image.jpg \
  confidence_threshold=0.3 \
  top_k=5
```

## Example Responses

### Nitrogen Deficiency Detection

**Input**: Image showing yellowing leaves

**Output**:
```json
{
  "success": true,
  "plant_health_score": 72.5,
  "summary": {
    "nutrient_deficiencies": [
      {
        "nutrient": "N",
        "name": "Nitrogen Deficiency",
        "confidence": 85.23
      }
    ],
    "diseases": [],
    "total_issues": 1
  },
  "diagnoses": [
    {
      "raw_label": "nitrogen_deficiency",
      "normalized_label": "nitrogen_deficiency",
      "confidence": 85.23,
      "category": "nutrient_deficiency",
      "subcategory": "N",
      "name": "Nitrogen Deficiency",
      "symptoms": [
        "Yellowing of older leaves (chlorosis) starting from leaf tips",
        "Stunted growth and smaller leaves",
        "Pale green to yellow color throughout plant",
        "Older leaves drop prematurely",
        "Reduced flowering and fruiting"
      ],
      "causes": [
        "Poor soil fertility or depleted nitrogen levels",
        "Excessive rainfall leaching nitrogen from soil",
        "Sandy soils with low organic matter",
        "Over-cropping without replenishing nutrients",
        "Cold soil temperatures limiting nitrogen availability"
      ],
      "treatment": [
        "Apply nitrogen-rich fertilizers (urea, ammonium nitrate, or blood meal)",
        "Use organic compost or well-rotted manure",
        "Apply foliar spray with diluted liquid nitrogen fertilizer",
        "Plant nitrogen-fixing cover crops (legumes)",
        "Mulch to reduce nitrogen leaching",
        "Split fertilizer applications for better absorption"
      ],
      "severity": "severe"
    }
  ],
  "recommendations": [
    "PRIMARY ACTION: Apply nitrogen-rich fertilizers (urea, ammonium nitrate, or blood meal)",
    "Apply balanced fertilizer to address N deficiencies",
    "Monitor plant health daily and document changes",
    "Ensure proper watering schedule - avoid overwatering",
    "Maintain good air circulation around plants",
    "Remove and destroy any dead or severely infected plant material"
  ]
}
```

### Powdery Mildew Detection

**Input**: Image showing white powdery coating on leaves

**Output**:
```json
{
  "success": true,
  "plant_health_score": 68.7,
  "summary": {
    "nutrient_deficiencies": [],
    "diseases": [
      {
        "type": "fungal",
        "name": "Powdery Mildew",
        "confidence": 91.5
      }
    ],
    "total_issues": 1
  },
  "diagnoses": [
    {
      "raw_label": "powdery_mildew",
      "normalized_label": "powdery_mildew",
      "confidence": 91.5,
      "category": "fungal",
      "subcategory": "powdery_mildew",
      "name": "Powdery Mildew",
      "symptoms": [
        "White powdery coating on leaves, stems, and flowers",
        "Yellowing and curling of infected leaves",
        "Stunted growth and distorted leaves",
        "Premature leaf drop",
        "Reduced photosynthesis",
        "Flower and fruit damage in severe cases"
      ],
      "causes": [
        "High humidity with dry leaf surfaces",
        "Poor air circulation around plants",
        "Dense plant canopy trapping moisture",
        "Moderate temperatures (60-80°F)",
        "Shaded conditions",
        "Stressed plants more susceptible"
      ],
      "treatment": [
        "Remove and destroy infected plant parts immediately",
        "Improve air circulation by spacing plants properly",
        "Apply sulfur-based fungicide weekly",
        "Use neem oil spray as organic treatment",
        "Apply potassium bicarbonate solution",
        "Spray with baking soda solution (1 tbsp per gallon water)",
        "Water at base of plants, avoid wetting foliage",
        "Prune dense growth to increase airflow",
        "Apply milk spray (1:9 milk to water ratio)"
      ],
      "severity": "severe"
    }
  ],
  "recommendations": [
    "PRIMARY ACTION: Remove and destroy infected plant parts immediately",
    "⚠️ Apply fungicide treatment and improve air circulation to control fungal infections",
    "Monitor plant health daily and document changes",
    "Ensure proper watering schedule - avoid overwatering",
    "Maintain good air circulation around plants",
    "Remove and destroy any dead or severely infected plant material"
  ]
}
```

### Multiple Issues Detected

**Input**: Image with both nutrient deficiency and disease

**Output**:
```json
{
  "success": true,
  "plant_health_score": 45.2,
  "summary": {
    "nutrient_deficiencies": [
      {
        "nutrient": "K",
        "name": "Potassium Deficiency",
        "confidence": 67.8
      }
    ],
    "diseases": [
      {
        "type": "fungal",
        "name": "Early Blight",
        "confidence": 73.4
      }
    ],
    "total_issues": 2
  },
  "diagnoses": [...],
  "recommendations": [
    "PRIMARY ACTION: Remove and destroy infected leaves and debris",
    "⚠️ Apply fungicide treatment and improve air circulation to control fungal infections",
    "Apply balanced fertilizer to address K deficiencies",
    "Monitor plant health daily and document changes",
    "Ensure proper watering schedule - avoid overwatering",
    "Maintain good air circulation around plants"
  ]
}
```

## Testing with Postman

1. Create new POST request to `http://localhost:8000/api/v1/diagnose/`
2. Set Body type to `form-data`
3. Add key `file` with type `File` and select your plant image
4. Optional: Add keys for `confidence_threshold` and `top_k`
5. Send request

## Test Images

For testing, you can use:

1. **Sample images** from plant disease datasets
2. **Your own photos** of affected plants
3. **Images from the web** (ensure they show clear symptoms)

Good test images should have:
- Clear focus on affected areas
- Good lighting (natural daylight preferred)
- Close-up view of symptoms
- Minimal background clutter

## Interpreting Results

### Health Score
- **90-100**: Healthy or minor issues
- **70-89**: Moderate concerns
- **50-69**: Significant problems
- **< 50**: Critical condition

### Confidence Levels
- **> 80%**: High confidence, act immediately
- **50-80%**: Moderate confidence, monitor and treat
- **< 50%**: Low confidence, consider other factors

### Severity Levels
- **Mild**: Early stage, preventive measures sufficient
- **Moderate**: Established issue, treatment needed
- **Severe**: Advanced stage, aggressive treatment required

## Troubleshooting

### Error: Model not loaded
**Solution**: Wait 5-10 seconds after server start for model to load

### Error: No confident predictions
**Solution**:
- Check image quality and clarity
- Ensure symptoms are visible
- Try lowering `confidence_threshold`

### Unexpected results
**Solution**:
- Verify image shows clear disease symptoms
- Check if condition is in supported mappings
- Review raw_predictions in response for model outputs

## Performance Benchmarks

Test on a typical server setup:
- **CPU**: 200-500ms per image
- **GPU**: 50-100ms per image
- **Memory**: ~500MB model footprint

## Integration Testing

Test the full flow from mobile app:

1. Mobile app captures image
2. Uploads to `/api/v1/diagnose/`
3. Receives structured diagnosis
4. Displays symptoms, causes, treatments
5. Shows health score and recommendations

## Monitoring

Track these metrics in production:
- Average inference time
- Confidence score distribution
- Most common diagnosed conditions
- Health score distribution
- Error rates
