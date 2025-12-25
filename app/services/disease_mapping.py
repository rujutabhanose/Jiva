"""
Disease and Deficiency Mapping Layer
Maps model output labels to nutritional deficiencies and disease categories
Based on curated agronomy references and plant pathology knowledge
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DiagnosisInfo:
    """Structured diagnosis information"""
    category: str  # 'nutrient_deficiency', 'fungal', 'bacterial', 'viral', 'pest', 'environmental'
    subcategory: Optional[str]  # e.g., 'nitrogen', 'potassium', 'powdery_mildew'
    name: str
    symptoms: List[str]
    causes: List[str]
    treatment: List[str]
    severity_indicators: Dict[str, str]


# Comprehensive mapping from model labels to diagnosis categories
DISEASE_MAPPINGS = {
    # ==================== NUTRITIONAL DEFICIENCIES ====================

    # Nitrogen Deficiency
    "nitrogen_deficiency": DiagnosisInfo(
        category="nutrient_deficiency",
        subcategory="N",
        name="Nitrogen Deficiency",
        symptoms=[
            "Yellowing of older leaves (chlorosis) starting from leaf tips",
            "Stunted growth and smaller leaves",
            "Pale green to yellow color throughout plant",
            "Older leaves drop prematurely",
            "Reduced flowering and fruiting"
        ],
        causes=[
            "Poor soil fertility or depleted nitrogen levels",
            "Excessive rainfall leaching nitrogen from soil",
            "Sandy soils with low organic matter",
            "Over-cropping without replenishing nutrients",
            "Cold soil temperatures limiting nitrogen availability"
        ],
        treatment=[
            "Apply nitrogen-rich fertilizers (urea, ammonium nitrate, or blood meal)",
            "Use organic compost or well-rotted manure",
            "Apply foliar spray with diluted liquid nitrogen fertilizer",
            "Plant nitrogen-fixing cover crops (legumes)",
            "Mulch to reduce nitrogen leaching",
            "Split fertilizer applications for better absorption"
        ],
        severity_indicators={
            "mild": "Slight yellowing of lower leaves",
            "moderate": "Widespread yellowing and stunted growth",
            "severe": "Significant leaf drop and extremely stunted growth"
        }
    ),

    # Phosphorus Deficiency
    "phosphorus_deficiency": DiagnosisInfo(
        category="nutrient_deficiency",
        subcategory="P",
        name="Phosphorus Deficiency",
        symptoms=[
            "Dark green or purplish discoloration of leaves",
            "Purple or reddish tints on leaf undersides and stems",
            "Stunted root development",
            "Delayed maturity and flowering",
            "Small, dark leaves with reduced growth"
        ],
        causes=[
            "Low phosphorus availability in acidic or alkaline soils",
            "Cold soil temperatures reducing phosphorus uptake",
            "High iron or aluminum in soil binding phosphorus",
            "Compacted soil limiting root growth",
            "Insufficient organic matter"
        ],
        treatment=[
            "Apply phosphorus-rich fertilizers (superphosphate, bone meal)",
            "Add rock phosphate for long-term phosphorus supply",
            "Adjust soil pH to 6.0-7.0 for optimal phosphorus availability",
            "Incorporate compost to improve soil structure",
            "Use mycorrhizal inoculants to enhance phosphorus uptake",
            "Avoid over-watering which can leach phosphorus"
        ],
        severity_indicators={
            "mild": "Slight purple tinting on leaf edges",
            "moderate": "Widespread purpling and reduced growth",
            "severe": "Severe stunting with dark purple leaves"
        }
    ),

    # Potassium Deficiency
    "potassium_deficiency": DiagnosisInfo(
        category="nutrient_deficiency",
        subcategory="K",
        name="Potassium Deficiency",
        symptoms=[
            "Yellowing and browning of leaf margins (necrosis)",
            "Scorched or burnt appearance on leaf edges",
            "Weak stems prone to lodging or breaking",
            "Reduced disease resistance",
            "Poor fruit and flower development",
            "Curling or cupping of leaves"
        ],
        causes=[
            "Sandy or light soils with low potassium retention",
            "Excessive calcium or magnesium interfering with potassium uptake",
            "High nitrogen levels depleting potassium",
            "Heavy rainfall leaching potassium from soil",
            "Poor soil drainage"
        ],
        treatment=[
            "Apply potassium-rich fertilizers (potassium sulfate, potassium chloride)",
            "Use wood ash as organic potassium source",
            "Apply kelp meal or greensand for slow-release potassium",
            "Improve soil structure with compost",
            "Balance NPK ratio in fertilization program",
            "Mulch to reduce potassium leaching"
        ],
        severity_indicators={
            "mild": "Slight leaf margin yellowing",
            "moderate": "Brown edges with visible necrosis",
            "severe": "Extensive leaf death and weak plant structure"
        }
    ),

    # Iron Deficiency
    "iron_deficiency": DiagnosisInfo(
        category="nutrient_deficiency",
        subcategory="Fe",
        name="Iron Deficiency (Iron Chlorosis)",
        symptoms=[
            "Interveinal chlorosis on young leaves (veins remain green)",
            "New growth appears yellow or white",
            "Leaf tips may die back in severe cases",
            "Stunted growth of new shoots",
            "Reduced chlorophyll production"
        ],
        causes=[
            "High soil pH (above 7.5) limiting iron availability",
            "Waterlogged or poorly drained soil",
            "Excess phosphorus interfering with iron uptake",
            "High levels of competing metals (zinc, manganese, copper)",
            "Cold soil temperatures",
            "Compacted soil restricting root growth"
        ],
        treatment=[
            "Apply chelated iron (Fe-EDTA or Fe-EDDHA) for quick results",
            "Lower soil pH with sulfur or acidifying fertilizers",
            "Use iron sulfate as soil amendment",
            "Apply foliar iron spray for immediate relief",
            "Improve soil drainage",
            "Add organic matter to enhance iron availability",
            "Avoid over-watering"
        ],
        severity_indicators={
            "mild": "Slight yellowing between veins on new leaves",
            "moderate": "Pronounced interveinal chlorosis",
            "severe": "White or cream-colored new growth, severe stunting"
        }
    ),

    # Magnesium Deficiency
    "magnesium_deficiency": DiagnosisInfo(
        category="nutrient_deficiency",
        subcategory="Mg",
        name="Magnesium Deficiency",
        symptoms=[
            "Interveinal chlorosis on older leaves first",
            "Yellowing between veins while veins stay green",
            "Reddish or purple tints may develop",
            "Leaf curling or cupping downward",
            "Premature leaf drop",
            "Reduced photosynthesis efficiency"
        ],
        causes=[
            "Acidic soils with low magnesium levels",
            "Sandy soils prone to magnesium leaching",
            "Excess potassium or calcium blocking magnesium uptake",
            "Heavy rainfall washing away magnesium",
            "High nitrogen levels depleting magnesium",
            "Poor root development"
        ],
        treatment=[
            "Apply Epsom salt (magnesium sulfate) as foliar spray or soil drench",
            "Use dolomitic limestone to raise pH and add magnesium",
            "Apply magnesium-rich fertilizers",
            "Add compost with magnesium content",
            "Balance calcium-to-magnesium ratio in soil",
            "Reduce excess potassium if present",
            "Mulch to prevent nutrient leaching"
        ],
        severity_indicators={
            "mild": "Slight yellowing between veins on lower leaves",
            "moderate": "Widespread interveinal chlorosis",
            "severe": "Severe yellowing with brown necrotic spots"
        }
    ),

    # Calcium Deficiency
    "calcium_deficiency": DiagnosisInfo(
        category="nutrient_deficiency",
        subcategory="Ca",
        name="Calcium Deficiency",
        symptoms=[
            "Death of growing tips (apical meristem)",
            "Distorted or deformed new leaves",
            "Blossom end rot in fruits (tomatoes, peppers)",
            "Brown spots on leaf tips and margins",
            "Weak cell walls leading to tissue collapse",
            "Stunted root growth"
        ],
        causes=[
            "Acidic soil with low calcium availability",
            "Irregular watering causing calcium uptake issues",
            "High humidity reducing transpiration",
            "Excess nitrogen, potassium, or magnesium competing with calcium",
            "Poor soil structure limiting root development",
            "Salt accumulation in soil"
        ],
        treatment=[
            "Apply agricultural lime (calcium carbonate) to raise pH",
            "Use gypsum (calcium sulfate) in alkaline soils",
            "Ensure consistent and adequate watering",
            "Apply calcium chloride as foliar spray",
            "Maintain proper soil pH (6.0-7.0)",
            "Improve soil aeration and drainage",
            "Avoid excessive nitrogen fertilization"
        ],
        severity_indicators={
            "mild": "Slight tip burn on new leaves",
            "moderate": "Visible blossom end rot or tip death",
            "severe": "Widespread tissue death and fruit damage"
        }
    ),

    # ==================== FUNGAL DISEASES ====================

    # Powdery Mildew
    "powdery_mildew": DiagnosisInfo(
        category="fungal",
        subcategory="powdery_mildew",
        name="Powdery Mildew",
        symptoms=[
            "White powdery coating on leaves, stems, and flowers",
            "Yellowing and curling of infected leaves",
            "Stunted growth and distorted leaves",
            "Premature leaf drop",
            "Reduced photosynthesis",
            "Flower and fruit damage in severe cases"
        ],
        causes=[
            "High humidity with dry leaf surfaces",
            "Poor air circulation around plants",
            "Dense plant canopy trapping moisture",
            "Moderate temperatures (60-80°F)",
            "Shaded conditions",
            "Stressed plants more susceptible"
        ],
        treatment=[
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
        severity_indicators={
            "mild": "Small white patches on few leaves",
            "moderate": "Widespread white coating, some yellowing",
            "severe": "Extensive coverage, leaf drop, and stunted growth"
        }
    ),

    # Early Blight
    "early_blight": DiagnosisInfo(
        category="fungal",
        subcategory="early_blight",
        name="Early Blight",
        symptoms=[
            "Dark brown spots with concentric rings (target pattern)",
            "Yellowing around spots on older leaves first",
            "Spots enlarge and merge causing leaf death",
            "Stem lesions with dark rings",
            "Fruit spots near stem end",
            "Premature defoliation"
        ],
        causes=[
            "Alternaria fungi surviving in soil and plant debris",
            "Warm, humid weather (75-85°F)",
            "Poor air circulation",
            "Overhead watering wetting foliage",
            "Plants weakened by stress or other issues",
            "Dense planting increasing humidity"
        ],
        treatment=[
            "Remove and destroy infected leaves and debris",
            "Apply copper-based fungicide at first sign",
            "Use chlorothalonil fungicide for control",
            "Stake and prune plants for better airflow",
            "Mulch around base to prevent soil splash",
            "Rotate crops yearly (3-4 year rotation)",
            "Water at soil level, avoid wetting leaves",
            "Apply organic neem oil spray",
            "Improve soil health with compost"
        ],
        severity_indicators={
            "mild": "Few target spots on lower leaves",
            "moderate": "Multiple leaves affected, some defoliation",
            "severe": "Extensive defoliation, stem lesions, fruit damage"
        }
    ),

    # Late Blight
    "late_blight": DiagnosisInfo(
        category="fungal",
        subcategory="late_blight",
        name="Late Blight",
        symptoms=[
            "Dark, water-soaked lesions on leaves",
            "White fuzzy growth on leaf undersides (in humid conditions)",
            "Rapid browning and death of leaves",
            "Dark lesions on stems causing collapse",
            "Fruit rot with brown, greasy appearance",
            "Entire plant can die within days"
        ],
        causes=[
            "Phytophthora infestans (oomycete pathogen)",
            "Cool, wet weather (50-60°F) with high humidity",
            "Infected seed tubers or transplants",
            "Wind-blown spores from nearby infected plants",
            "Extended periods of leaf wetness",
            "Poor air circulation"
        ],
        treatment=[
            "Remove and destroy all infected plants immediately",
            "Apply copper fungicide preventatively",
            "Use mancozeb or chlorothalonil fungicide",
            "Destroy volunteer plants and cull piles",
            "Plant resistant varieties",
            "Ensure wide plant spacing",
            "Avoid overhead irrigation",
            "Apply fungicides before rain events",
            "Practice crop rotation (minimum 3 years)",
            "Monitor weather for blight-favorable conditions"
        ],
        severity_indicators={
            "mild": "Few water-soaked lesions",
            "moderate": "Rapid spread of lesions, stem involvement",
            "severe": "Extensive plant death, fruit rot"
        }
    ),

    # Leaf Spot Diseases
    "leaf_spot": DiagnosisInfo(
        category="fungal",
        subcategory="leaf_spot",
        name="Fungal Leaf Spot",
        symptoms=[
            "Circular or irregular brown, black, or tan spots",
            "Yellow halo around spots",
            "Spots may have dark borders",
            "Holes in leaves where spots fall out",
            "Premature leaf yellowing and drop",
            "Reduced plant vigor"
        ],
        causes=[
            "Various fungal pathogens (Septoria, Cercospora, etc.)",
            "Warm, humid conditions",
            "Overhead watering or rain splash",
            "Poor air circulation",
            "Infected plant debris in soil",
            "Stressed or weakened plants"
        ],
        treatment=[
            "Remove infected leaves and clean up fallen debris",
            "Apply copper or sulfur fungicide",
            "Use chlorothalonil for severe infections",
            "Improve air circulation through pruning",
            "Water at base of plants in morning",
            "Mulch to prevent soil splash onto leaves",
            "Rotate crops to break disease cycle",
            "Space plants adequately",
            "Apply neem oil as preventative"
        ],
        severity_indicators={
            "mild": "Few spots on lower leaves",
            "moderate": "Many spots, some defoliation",
            "severe": "Extensive spotting and significant leaf loss"
        }
    ),

    # ==================== GENERAL/UNKNOWN FALLBACKS ====================

    # General Nutrient Deficiency (Fallback for unidentified nutrient issues)
    "general_nutrient_deficiency": DiagnosisInfo(
        category="nutrient_deficiency",
        subcategory="general",
        name="Potential Nutrient Deficiency",
        symptoms=[
            "Discoloration or abnormal coloring of leaves",
            "Changes in leaf texture or appearance",
            "Reduced growth or vigor",
            "Leaf distortion or unusual patterns",
            "Changes in plant color or health"
        ],
        causes=[
            "General soil nutrient depletion",
            "Imbalanced fertilization",
            "Poor soil pH affecting nutrient availability",
            "Insufficient organic matter in soil",
            "Environmental stress affecting nutrient uptake",
            "Root damage limiting nutrient absorption"
        ],
        treatment=[
            "Conduct soil test to identify specific nutrient deficiencies",
            "Apply balanced NPK fertilizer (10-10-10 or similar)",
            "Add compost or well-rotted organic matter to improve soil health",
            "Check and adjust soil pH to 6.0-7.0 range",
            "Ensure adequate watering for nutrient uptake",
            "Consider foliar feeding with diluted liquid fertilizer",
            "Monitor plant response and adjust treatment based on results",
            "Consult local agricultural extension service for specific recommendations"
        ],
        severity_indicators={
            "mild": "Slight changes in appearance",
            "moderate": "Noticeable discoloration or growth issues",
            "severe": "Significant damage affecting plant health"
        }
    ),

    # General Plant Stress (Fallback for very low confidence)
    "general_plant_stress": DiagnosisInfo(
        category="environmental",
        subcategory="general_stress",
        name="General Plant Stress",
        symptoms=[
            "Overall decline in plant vigor",
            "Leaf abnormalities",
            "Color changes in foliage",
            "Reduced growth rate",
            "Plant appears unhealthy"
        ],
        causes=[
            "Environmental stress (temperature, light, or humidity)",
            "Watering issues (over or under-watering)",
            "Pest damage",
            "Disease presence",
            "Nutrient imbalances",
            "Soil quality problems",
            "Root zone issues"
        ],
        treatment=[
            "Review and optimize growing conditions (light, water, temperature)",
            "Check soil moisture levels - adjust watering schedule",
            "Inspect plant thoroughly for pests or disease signs",
            "Test soil for nutrient levels and pH",
            "Ensure proper drainage to prevent root rot",
            "Consider transplanting if root-bound",
            "Apply balanced fertilizer if nutritional deficiency suspected",
            "Remove any dead or severely damaged plant parts",
            "Monitor daily and document changes",
            "Consult a plant expert or agronomist if condition worsens"
        ],
        severity_indicators={
            "mild": "Minor changes in appearance",
            "moderate": "Visible stress symptoms",
            "severe": "Plant in distress, immediate action needed"
        }
    ),

    # ==================== BACTERIAL DISEASES ====================

    # Bacterial Spot
    "bacterial_spot": DiagnosisInfo(
        category="bacterial",
        subcategory="bacterial_spot",
        name="Bacterial Spot",
        symptoms=[
            "Small, dark, greasy-looking spots on leaves",
            "Spots may have yellow halo",
            "Raised corky lesions on fruit",
            "Leaf spots turn brown and papery",
            "Defoliation in severe cases",
            "Reduced fruit quality"
        ],
        causes=[
            "Xanthomonas bacteria spread by water splash",
            "Warm, wet weather conditions",
            "Infected seeds or transplants",
            "Rain or overhead irrigation spreading bacteria",
            "Wounds from insects or equipment",
            "High humidity promoting infection"
        ],
        treatment=[
            "Remove and destroy infected plant material",
            "Apply copper-based bactericide preventatively",
            "Use antibiotic sprays if available (streptomycin)",
            "Avoid overhead watering",
            "Disinfect tools between plants",
            "Plant resistant varieties",
            "Rotate crops (3-4 years)",
            "Ensure good air circulation",
            "Avoid working with wet plants",
            "Use certified disease-free seed"
        ],
        severity_indicators={
            "mild": "Few leaf spots, no defoliation",
            "moderate": "Many spots, some defoliation and fruit lesions",
            "severe": "Extensive defoliation and unmarketable fruit"
        }
    ),

    # Bacterial Wilt
    "bacterial_wilt": DiagnosisInfo(
        category="bacterial",
        subcategory="bacterial_wilt",
        name="Bacterial Wilt",
        symptoms=[
            "Sudden wilting of entire plant or sections",
            "Wilting that doesn't recover overnight",
            "Brown discoloration of vascular tissue",
            "Milky bacterial ooze from cut stems",
            "Leaf yellowing before wilting",
            "Plant death within days to weeks"
        ],
        causes=[
            "Bacteria transmitted by cucumber beetles or through soil",
            "Warm soil temperatures",
            "Infected plant debris",
            "Wounds providing entry points",
            "Contaminated tools or equipment",
            "Bacteria overwintering in beetles"
        ],
        treatment=[
            "Remove and destroy infected plants immediately",
            "Control cucumber beetles with insecticides or row covers",
            "Use resistant varieties when available",
            "Practice crop rotation",
            "Disinfect tools thoroughly",
            "Remove plant debris at end of season",
            "Apply beneficial nematodes for beetle larvae",
            "Avoid spreading soil from infected areas",
            "Plant trap crops to lure beetles away"
        ],
        severity_indicators={
            "mild": "Wilting of single leaves or stems",
            "moderate": "Wilting of multiple branches",
            "severe": "Entire plant collapse and death"
        }
    ),

    # Bacterial Blight
    "bacterial_blight": DiagnosisInfo(
        category="bacterial",
        subcategory="bacterial_blight",
        name="Bacterial Blight",
        symptoms=[
            "Water-soaked lesions on leaves",
            "Brown or black spots with yellow margins",
            "Wilting and death of shoots and branches",
            "Cankers on stems",
            "Bacterial exudate (ooze) may be visible",
            "Rapid tissue death"
        ],
        causes=[
            "Pseudomonas or Xanthomonas bacteria",
            "Warm, wet weather",
            "Rain splash or overhead irrigation",
            "Infected seeds or pruning wounds",
            "High humidity",
            "Stressed plants more susceptible"
        ],
        treatment=[
            "Prune out infected branches below symptoms",
            "Disinfect pruning tools between cuts (10% bleach solution)",
            "Apply copper bactericide",
            "Remove infected plants if severely affected",
            "Improve air circulation",
            "Avoid overhead watering",
            "Plant resistant cultivars",
            "Apply during dry weather only",
            "Destroy all infected debris"
        ],
        severity_indicators={
            "mild": "Few water-soaked spots",
            "moderate": "Branch dieback, visible cankers",
            "severe": "Extensive plant death, oozing lesions"
        }
    ),
}


# Additional mappings for common model label variations
LABEL_ALIASES = {
    "N_deficiency": "nitrogen_deficiency",
    "P_deficiency": "phosphorus_deficiency",
    "K_deficiency": "potassium_deficiency",
    "Fe_deficiency": "iron_deficiency",
    "Mg_deficiency": "magnesium_deficiency",
    "Ca_deficiency": "calcium_deficiency",
    "nitrogen": "nitrogen_deficiency",
    "phosphorus": "phosphorus_deficiency",
    "potassium": "potassium_deficiency",
    "iron": "iron_deficiency",
    "magnesium": "magnesium_deficiency",
    "calcium": "calcium_deficiency",
    "powdery": "powdery_mildew",
    "mildew": "powdery_mildew",
    "early_blight_tomato": "early_blight",
    "late_blight_tomato": "late_blight",
    "leaf_spot_fungal": "leaf_spot",
    "bacterial_spot_tomato": "bacterial_spot",

    # MobileNet plant-specific disease labels (format: "{plant}_with_{disease}")
    "grape_with_black_rot": "leaf_spot",  # Black rot causes leaf spots
    "tomato_with_late_blight": "late_blight",
    "squash_with_powdery_mildew": "powdery_mildew",
    "cedar_apple_rust": "leaf_spot",  # Rust causes spotting
    "grape_with_esca_(black_measles)": "leaf_spot",  # Esca causes leaf spots
    "tomato_with_early_blight": "early_blight",
    "potato_with_late_blight": "late_blight",
    "potato_with_early_blight": "early_blight",
    "apple_with_black_rot": "leaf_spot",
    "apple_with_cedar_apple_rust": "leaf_spot",
    "corn_with_common_rust": "leaf_spot",
    "tomato_with_bacterial_spot": "bacterial_spot",
    "pepper_with_bacterial_spot": "bacterial_spot",
}


def normalize_label(label: str) -> str:
    """Normalize model output label to standard format"""
    label = label.lower().replace(" ", "_").replace("-", "_")
    return LABEL_ALIASES.get(label, label)


def get_diagnosis_info(label: str) -> Optional[DiagnosisInfo]:
    """Get structured diagnosis information from model label"""
    normalized = normalize_label(label)
    return DISEASE_MAPPINGS.get(normalized)


def is_nutrient_deficiency(label: str) -> bool:
    """Check if label corresponds to nutrient deficiency"""
    info = get_diagnosis_info(label)
    return info is not None and info.category == "nutrient_deficiency"


def is_fungal_disease(label: str) -> bool:
    """Check if label corresponds to fungal disease"""
    info = get_diagnosis_info(label)
    return info is not None and info.category == "fungal"


def is_bacterial_disease(label: str) -> bool:
    """Check if label corresponds to bacterial disease"""
    info = get_diagnosis_info(label)
    return info is not None and info.category == "bacterial"


def get_nutrient_type(label: str) -> Optional[str]:
    """Get nutrient type (N, P, K, Fe, Mg, Ca) from label"""
    info = get_diagnosis_info(label)
    if info and info.category == "nutrient_deficiency":
        return info.subcategory
    return None
