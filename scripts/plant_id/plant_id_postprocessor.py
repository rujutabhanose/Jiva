#!/usr/bin/env python3
"""
Post-process plant identification results
- Enrich with USDA database (403K+ plants)
- Filter by confidence threshold
- Group similar species
- Provide alternative suggestions
"""

import json
from typing import List, Dict, Any
from pathlib import Path


class PlantIdentificationPostProcessor:
    """Post-process plant identification results with USDA enrichment"""

    def __init__(self):
        self.confidence_threshold = 35.0  # Only accept >35% confidence
        self.family_groups = self._load_family_groups()
        self.usda_plants = self._load_usda_database()

    def _load_usda_database(self) -> Dict[str, Any]:
        """Load USDA plant database (403K+ plants)"""
        usda_path = Path(__file__).parent / "app" / "data" / "plants_usda.json"

        if not usda_path.exists():
            # Try alternate path
            usda_path = Path(__file__).parent.parent / "backend" / "app" / "data" / "plants_usda.json"

        if usda_path.exists():
            try:
                with open(usda_path, 'r', encoding='utf-8') as f:
                    print(f"  Loaded USDA database from {usda_path}")
                    return json.load(f)
            except Exception as e:
                print(f"  Warning: Could not load USDA database: {e}")
                return {}
        else:
            print(f"  Warning: USDA database not found at {usda_path}")
            return {}

    def _normalize_scientific_name(self, scientific_name: str) -> str:
        """Normalize scientific name for USDA lookup"""
        # Remove parenthetical annotations
        sci = scientific_name.split("(")[0].strip()
        parts = sci.split()
        if len(parts) >= 2:
            key = f"{parts[0]}_{parts[1]}"
        else:
            key = sci.replace(" ", "_")
        return key.lower()

    def _enrich_with_usda(self, scientific_name: str, confidence: float) -> Dict[str, Any]:
        """Enrich plant data with USDA database info"""
        key = self._normalize_scientific_name(scientific_name)
        usda_info = self.usda_plants.get(key, {})

        return {
            "scientific_name": scientific_name,
            "common_name": usda_info.get("commonName") or self._get_common_name_fallback(scientific_name),
            "family": usda_info.get("family"),
            "usda_symbol": usda_info.get("symbol"),
            "confidence": round(confidence, 1),
            "usda_enriched": bool(usda_info)
        }

    def _get_common_name_fallback(self, scientific_name: str) -> str:
        """Fallback common names for popular houseplants"""
        fallback_names = {
            'Epipremnum aureum': 'Money Plant',
            'Dracaena trifasciata': 'Snake Plant',
            'Sansevieria trifasciata': 'Mother-in-law\'s Tongue',
            'Monstera deliciosa': 'Swiss Cheese Plant',
            'Ficus elastica': 'Rubber Plant',
            'Philodendron hederaceum': 'Heart Leaf Philodendron',
            'Scindapsus pictus': 'Satin Pothos',
            'Syngonium podophyllum': 'Arrowhead Plant',
            'Spathiphyllum wallisii': 'Peace Lily',
            'Tradescantia fluminensis': 'Wandering Jew',
            'Ficus lyrata': 'Fiddle Leaf Fig',
            'Chlorophytum comosum': 'Spider Plant',
            'Aloe vera': 'Aloe Vera',
            'Crassula ovata': 'Jade Plant',
        }
        return fallback_names.get(scientific_name, scientific_name)

    def _load_family_groups(self):
        """Plant families and similar species"""
        return {
            'Araceae': {
                'species': ['Epipremnum aureum', 'Monstera deliciosa', 'Syngonium podophyllum',
                           'Philodendron hederaceum', 'Scindapsus pictus', 'Spathiphyllum wallisii'],
                'common_traits': ['climbing vine', 'large leaves', 'prefers humidity']
            },
            'Asparagaceae': {
                'species': ['Dracaena trifasciata', 'Sansevieria trifasciata', 'Chlorophytum comosum'],
                'common_traits': ['upright growth', 'drought tolerant', 'variegated leaves']
            },
            'Moraceae': {
                'species': ['Ficus elastica', 'Ficus pumila', 'Ficus lyrata', 'Ficus benjamina'],
                'common_traits': ['latex sap', 'glossy leaves', 'tree-like growth']
            },
            'Commelinaceae': {
                'species': ['Tradescantia fluminensis', 'Tradescantia pallida', 'Tradescantia zebrina'],
                'common_traits': ['trailing vine', 'colorful leaves', 'fast growing']
            },
            'Crassulaceae': {
                'species': ['Crassula ovata', 'Echeveria elegans', 'Sedum morganianum'],
                'common_traits': ['succulent', 'drought tolerant', 'thick leaves']
            }
        }

    def process_result(self, raw_result: Dict, image_metadata: Dict = None) -> Dict:
        """
        Post-process identification result with USDA enrichment

        Args:
            raw_result: Raw output from ensemble model
            image_metadata: Image metadata (brightness, color, etc.)

        Returns:
            Processed result with USDA data, confidence and alternatives
        """

        primary_confidence = raw_result['primary']['confidence']

        # Reject if confidence too low
        if primary_confidence < self.confidence_threshold:
            return self._low_confidence_response(raw_result)

        # Enrich primary result with USDA data
        primary_enriched = self._enrich_with_usda(
            raw_result['primary']['scientific_name'],
            primary_confidence
        )

        # Also enrich alternatives with USDA data
        alternatives_enriched = []
        for alt in raw_result.get('alternatives', []):
            if alt.get('confidence', 0) > 15.0:
                enriched = self._enrich_with_usda(
                    alt['scientific_name'],
                    alt['confidence']
                )
                alternatives_enriched.append(enriched)

        # Limit alternatives
        alternatives_enriched = alternatives_enriched[:4]

        # Add confidence assessment
        confidence_level = self._assess_confidence_level(primary_confidence)

        # Get family from USDA or fallback
        family = primary_enriched.get('family')

        return {
            'primary': primary_enriched,
            'alternatives': alternatives_enriched,
            'confidence_level': confidence_level,
            'recommendation': self._get_recommendation(primary_enriched),
            'similar_species': self._get_similar_species(
                raw_result['primary']['scientific_name'],
                family
            ),
            'ensemble_info': {
                'models_used': raw_result.get('models_used', 1),
                'ensemble_confidence': raw_result.get('ensemble_confidence', 0)
            }
        }

    def _low_confidence_response(self, raw_result: Dict) -> Dict:
        """Handle low confidence predictions"""
        # Enrich alternatives even for low confidence
        alternatives_enriched = []
        for alt in raw_result.get('alternatives', [])[:3]:
            enriched = self._enrich_with_usda(
                alt.get('scientific_name', ''),
                alt.get('confidence', 0)
            )
            alternatives_enriched.append(enriched)

        return {
            'primary': None,
            'alternatives': alternatives_enriched,
            'confidence_level': 'very_low',
            'recommendation': 'Could not identify with high confidence. Please try: 1) Better lighting 2) Show full plant 3) Show leaves clearly',
            'similar_species': [],
            'user_action': 'request_more_images'
        }

    def _assess_confidence_level(self, confidence: float) -> str:
        """Assess confidence level"""
        if confidence >= 80:
            return 'very_high'
        elif confidence >= 60:
            return 'high'
        elif confidence >= 40:
            return 'medium'
        else:
            return 'low'

    def _get_recommendation(self, primary: Dict) -> str:
        """Get care recommendations"""
        species_name = primary.get('scientific_name', '')
        common_name = primary.get('common_name', species_name)

        # Care database
        care_db = {
            'Epipremnum aureum': 'Bright indirect light, water when soil is dry, loves humidity',
            'Dracaena trifasciata': 'Low light tolerant, water sparingly (once a month), very hardy',
            'Ficus elastica': 'Bright light, allow soil to dry between watering, wipe leaves regularly',
            'Monstera deliciosa': 'Bright indirect light, moderate watering, needs support as it grows',
            'Syngonium podophyllum': 'Medium light, keep soil moist, grows quickly',
            'Tradescantia fluminensis': 'Bright light for color, water regularly, pinch tips to encourage bushiness',
            'Philodendron hederaceum': 'Low to bright indirect light, water when top inch is dry',
            'Scindapsus pictus': 'Bright indirect light, moderate watering, slow grower',
            'Sansevieria trifasciata': 'Very low light tolerant, water every 3-4 weeks, nearly indestructible',
            'Spathiphyllum wallisii': 'Low to medium light, keep soil moist, will droop when thirsty',
            'Ficus lyrata': 'Bright indirect light, consistent watering, avoid moving',
            'Chlorophytum comosum': 'Indirect light, moderate watering, produces baby plants',
            'Aloe vera': 'Bright light, water deeply but infrequently, well-draining soil',
            'Crassula ovata': 'Bright light, water when completely dry, easy propagation',
        }

        return care_db.get(species_name, f'Plant identified as {common_name}. Research specific care for best results.')

    def _get_similar_species(self, scientific_name: str, family: str = None) -> List[Dict]:
        """Get similar species in same family"""

        # First check our curated groups
        for fam, data in self.family_groups.items():
            if scientific_name in data['species']:
                similar = [s for s in data['species'] if s != scientific_name]
                return [
                    {'name': s, 'reason': f'Same family: {fam}'}
                    for s in similar[:3]
                ]

        # If USDA family is known, search for others in same family
        if family and self.usda_plants:
            similar = []
            for key, info in self.usda_plants.items():
                if info.get('family') == family and info.get('commonName'):
                    sci_name = info.get('scientificName', '').split('(')[0].strip()
                    if sci_name and sci_name != scientific_name:
                        similar.append({
                            'name': sci_name,
                            'common_name': info.get('commonName'),
                            'reason': f'Same family: {family}'
                        })
                        if len(similar) >= 3:
                            break
            return similar

        return []


# Test
if __name__ == "__main__":
    processor = PlantIdentificationPostProcessor()

    print(f"\nUSDA database loaded: {len(processor.usda_plants)} entries")

    # Sample result
    sample = {
        'primary': {
            'scientific_name': 'Epipremnum aureum',
            'confidence': 85.5,
            'common_name': 'Money Plant'
        },
        'alternatives': [
            {'scientific_name': 'Scindapsus pictus', 'confidence': 12.3, 'common_name': 'Satin Pothos'},
            {'scientific_name': 'Monstera deliciosa', 'confidence': 8.9, 'common_name': 'Swiss Cheese Plant'},
        ],
        'models_used': 2,
        'ensemble_confidence': 0.82
    }

    result = processor.process_result(sample)
    print("\nProcessed result:")
    print(json.dumps(result, indent=2))

    # Test USDA lookup
    print("\n--- USDA Lookup Test ---")
    test_names = ['Abies alba', 'Monstera deliciosa', 'Ficus elastica', 'Unknown species']
    for name in test_names:
        enriched = processor._enrich_with_usda(name, 90.0)
        print(f"\n{name}:")
        print(f"  Common: {enriched['common_name']}")
        print(f"  Family: {enriched['family']}")
        print(f"  USDA Symbol: {enriched['usda_symbol']}")
        print(f"  Enriched: {enriched['usda_enriched']}")
