# Intelligent Fallback System for Unknown Labels

## Overview

The diagnosis system includes an intelligent fallback mechanism to handle cases where the model produces labels that aren't explicitly mapped in our database. This ensures users always receive actionable guidance, even for uncertain or unmapped conditions.

## When Fallbacks Are Used

Fallbacks activate when:
1. Model predicts a label not in our mapping database
2. Label exists but confidence is extremely low (< 30%)
3. Symptoms are visible but can't be confidently identified

## Fallback Categories

### 1. General Plant Stress (Environmental)
**Used when:**
- Confidence < 25% (very uncertain)
- Label contains environmental keywords: "stress", "damage", "burn", "scorch", "water", "drought", "heat", "cold"
- Label contains unmapped disease keywords
- Last resort when no other pattern matches

**Provides:**
- Broad environmental assessment
- General troubleshooting steps
- Guidance on basic plant care
- Recommendation to consult an expert

**Example Response:**
```json
{
  "name": "General Plant Stress (Uncertain Diagnosis)",
  "category": "environmental",
  "subcategory": "general_stress",
  "confidence": 18.5,
  "severity": "mild",
  "treatment": [
    "⚠️ Note: This is a general diagnosis based on limited information.",
    "For accurate treatment, consult a professional agronomist or plant pathologist.",
    "Review and optimize growing conditions (light, water, temperature)",
    "Check soil moisture levels - adjust watering schedule",
    "Inspect plant thoroughly for pests or disease signs",
    ...
  ]
}
```

### 2. General Nutrient Deficiency
**Used when:**
- Confidence 25-100% but label not mapped
- Label contains nutrient keywords: "deficiency", "chlorosis", "yellowing", "nutrient", "npk", "fertilizer"
- Default for moderate confidence unmapped labels (most plant issues are nutrition-related)

**Provides:**
- Nutrient-focused assessment
- Soil testing recommendations
- Balanced fertilization guidance
- pH and soil health tips

**Example Response:**
```json
{
  "name": "Potential Nutrient Deficiency (Uncertain Diagnosis)",
  "category": "nutrient_deficiency",
  "subcategory": "general",
  "confidence": 42.3,
  "severity": "mild",
  "treatment": [
    "⚠️ Note: This is a general diagnosis based on limited information.",
    "For accurate treatment, consult a professional agronomist or plant pathologist.",
    "Conduct soil test to identify specific nutrient deficiencies",
    "Apply balanced NPK fertilizer (10-10-10 or similar)",
    "Add compost or well-rotted organic matter to improve soil health",
    ...
  ]
}
```

## Decision Logic

```python
def _determine_fallback(label: str, confidence: float) -> str:
    """Intelligent fallback selection"""

    # Very low confidence -> environmental stress
    if confidence < 0.25:
        return "general_plant_stress"

    # Nutrient-related keywords
    if any(["deficiency", "yellowing", "chlorosis", ...] in label):
        return "general_nutrient_deficiency"

    # Disease keywords (but unmapped)
    if any(["disease", "fungal", "bacterial", ...] in label):
        return "general_plant_stress"

    # Environmental keywords
    if any(["stress", "burn", "drought", ...] in label):
        return "general_plant_stress"

    # Default: assume nutrition issue (most common)
    if confidence >= 0.25:
        return "general_nutrient_deficiency"

    # Last resort
    return "general_plant_stress"
```

## Confidence Level Behavior

| Confidence | Fallback Behavior | User Guidance |
|------------|-------------------|---------------|
| < 25% | General Plant Stress | "Very uncertain - inspect thoroughly" |
| 25-40% | Keyword-based fallback | "Uncertain - monitor and test" |
| 40-60% | Keyword-based fallback | "Possible issue - take preventive action" |
| > 60% | Keyword-based fallback | "Likely issue - apply treatment" |

## Warning Messages

All fallback diagnoses include clear warnings:

```
⚠️ Note: This is a general diagnosis based on limited information.
For accurate treatment, consult a professional agronomist or plant pathologist.
```

This ensures users understand the diagnosis is less certain than mapped conditions.

## Health Score Impact

Fallback categories affect health scores differently:

```python
category_multiplier = {
    "nutrient_deficiency": 1.0,    # Standard impact
    "environmental": 0.8,           # Reduced impact (less certain)
    "fungal": 1.3,
    "bacterial": 1.5,
    "viral": 1.7,
    "unknown": 1.0
}
```

Environmental stress (fallback) deducts less from health score since it's less certain.

## Recommendations Priority

Fallback recommendations are prioritized lower than specific diagnoses:

1. **Urgent**: Bacterial infections (if detected)
2. **High**: Fungal infections (if detected)
3. **Medium**: Specific nutrient deficiencies
4. **Medium**: General nutrient deficiency (fallback)
5. **Lower**: Environmental stress (fallback)

## Examples

### Example 1: Unknown Label "leaf_curl_virus" at 15% Confidence

**Input:**
```python
label = "leaf_curl_virus"
confidence = 0.15
```

**Fallback Selection:**
- Confidence < 25% → `general_plant_stress`

**Response:**
```json
{
  "raw_label": "leaf_curl_virus",
  "name": "General Plant Stress (Uncertain Diagnosis)",
  "category": "environmental",
  "confidence": 15.0,
  "severity": "mild",
  "treatment": [
    "⚠️ Note: This is a general diagnosis...",
    "Review and optimize growing conditions",
    "Inspect for pests or disease signs",
    ...
  ]
}
```

### Example 2: Unknown Label "zinc_deficiency" at 45% Confidence

**Input:**
```python
label = "zinc_deficiency"
confidence = 0.45
```

**Fallback Selection:**
- Contains "deficiency" keyword → `general_nutrient_deficiency`

**Response:**
```json
{
  "raw_label": "zinc_deficiency",
  "name": "Potential Nutrient Deficiency (Uncertain Diagnosis)",
  "category": "nutrient_deficiency",
  "subcategory": "general",
  "confidence": 45.0,
  "severity": "mild",
  "treatment": [
    "⚠️ Note: This is a general diagnosis...",
    "Conduct soil test to identify specific nutrient deficiencies",
    "Apply balanced NPK fertilizer",
    ...
  ]
}
```

### Example 3: Unknown Label "sunburn" at 65% Confidence

**Input:**
```python
label = "sunburn"
confidence = 0.65
```

**Fallback Selection:**
- Contains "burn" keyword → `general_plant_stress`

**Response:**
```json
{
  "raw_label": "sunburn",
  "name": "General Plant Stress (Uncertain Diagnosis)",
  "category": "environmental",
  "confidence": 65.0,
  "severity": "moderate",
  "treatment": [
    "⚠️ Note: This is a general diagnosis...",
    "Review and optimize growing conditions",
    "Check for excessive sun exposure",
    ...
  ]
}
```

## Adding Specific Mappings

When you notice common unmapped labels in logs, add specific mappings:

1. Check logs for `"Unmapped label detected: X"`
2. Research the condition
3. Add to `DISEASE_MAPPINGS` in `disease_mapping.py`:

```python
"zinc_deficiency": DiagnosisInfo(
    category="nutrient_deficiency",
    subcategory="Zn",
    name="Zinc Deficiency",
    symptoms=[...],
    causes=[...],
    treatment=[...],
    severity_indicators={...}
),
```

4. Update `LABEL_ALIASES` if needed
5. Restart server to use new mapping

## Monitoring Fallbacks

Track fallback usage in production:

```bash
# Count fallback occurrences in logs
grep "Unmapped label detected" logs/app.log | wc -l

# See most common unmapped labels
grep "Unmapped label detected" logs/app.log | \
  cut -d':' -f3 | sort | uniq -c | sort -rn
```

This helps identify which conditions need specific mappings.

## Benefits

1. **Always Actionable**: Users always get guidance, never "unknown error"
2. **Safe Defaults**: Conservative recommendations won't harm plants
3. **Clear Uncertainty**: Users know when diagnosis is uncertain
4. **Graceful Degradation**: System works even with unmapped labels
5. **Easy Expansion**: Add specific mappings as needed

## User Experience

**With Fallbacks:**
- ✅ "Potential Nutrient Deficiency - conduct soil test"
- ✅ Clear warning about uncertainty
- ✅ General but helpful advice

**Without Fallbacks:**
- ❌ "Unknown condition"
- ❌ No guidance
- ❌ User confusion

## Testing Fallbacks

Test with unmapped labels:

```python
# Test very low confidence
result = diagnose_image("test_image.jpg", confidence_threshold=0.1)
# Should see "General Plant Stress"

# Test nutrient-like label
# (simulate by checking logs for unmapped nutrient labels)
```

## Future Improvements

1. **Machine Learning**: Train on fallback cases to improve specific mappings
2. **User Feedback**: Collect feedback on fallback diagnoses
3. **Regional Variations**: Adjust fallbacks based on geographic region
4. **Seasonal Adjustments**: Consider time of year in fallback logic
5. **Multi-language**: Provide fallback guidance in multiple languages
