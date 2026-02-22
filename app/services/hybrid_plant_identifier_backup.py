"""
Hybrid plant identification that uses both local and cloud models.
Returns the result with highest confidence.
"""
import logging
from typing import Dict, Optional
from PIL import Image
import asyncio

logger = logging.getLogger(__name__)


class HybridPlantIdentifier:
    """
    Combines local (transformers) and cloud (HuggingFace InferenceClient) models.
    Uses whichever gives higher confidence, with fallback to local if offline.
    """

    def __init__(self):
        self.local_available = False
        self.cloud_available = False

        # Import local model
        try:
            from app.services.plant_identifier import identify_plant as local_identify
            from app.services.plant_identifier import processor, model
            if processor is not None and model is not None:
                self.local_identify = local_identify
                self.local_available = True
                logger.info("✅ Local plant identification model available")
        except Exception as e:
            logger.warning(f"⚠️  Local model not available: {e}")

        # Import cloud model
        try:
            import sys
            from pathlib import Path
            # Add backend root to path to import the cloud identifier
            backend_path = Path(__file__).parent.parent.parent
            if str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))

            from scripts.plant_id.plant_identifier import identifier as cloud_identifier
            self.cloud_identifier = cloud_identifier
            self.cloud_available = True
            logger.info("✅ Cloud plant identification (InferenceClient) available")
        except Exception as e:
            logger.warning(f"⚠️  Cloud model not available: {e}")

    def _check_local_model(self):
        """Re-check local model availability (handles late loading)."""
        if not self.local_available:
            try:
                from app.services.plant_identifier import identify_plant as local_identify
                from app.services.plant_identifier import processor, model
                if processor is not None and model is not None:
                    self.local_identify = local_identify
                    self.local_available = True
                    logger.info("✅ Local plant identification model now available")
            except Exception:
                pass

    async def identify(self, image_path: str, top_k: int = 3, region: str = "India") -> Dict:
        """
        Identify plant using hybrid approach.

        Returns:
            {
                "primary": {
                    "commonName": str,
                    "scientificName": str,
                    "confidence": float,
                    "source": "local" | "cloud" | "hybrid"
                },
                "alternatives": [...],
                "metadata": {
                    "local_confidence": float or None,
                    "cloud_confidence": float or None,
                    "method_used": str
                }
            }
        """
        # Re-check local model in case it was loaded after init
        self._check_local_model()

        local_result = None
        cloud_result = None

        # Try both models in parallel
        tasks = []

        if self.local_available:
            tasks.append(self._get_local_prediction(image_path, top_k))

        if self.cloud_available:
            tasks.append(self._get_cloud_prediction(image_path, top_k, region))

        if not tasks:
            raise RuntimeError("No plant identification models available")

        # Wait for both to complete (or timeout)
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse results
        if self.local_available:
            local_result = completed[0] if not isinstance(completed[0], Exception) else None
            if isinstance(completed[0], Exception):
                logger.warning(f"Local model failed: {completed[0]}")

        if self.cloud_available:
            cloud_idx = 1 if self.local_available else 0
            cloud_result = completed[cloud_idx] if not isinstance(completed[cloud_idx], Exception) else None
            if isinstance(completed[cloud_idx], Exception):
                logger.warning(f"Cloud model failed: {completed[cloud_idx]}")

        # Compare and choose best
        return self._merge_results(local_result, cloud_result)

    async def _get_local_prediction(self, image_path: str, top_k: int) -> Dict:
        """Get prediction from local transformers model."""
        try:
            # Run in thread pool since it's CPU-bound
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.local_identify,
                image_path,
                top_k
            )

            # Normalize format
            primary = result["primary"]
            return {
                "species": primary.get("commonName") or primary.get("scientificName"),
                "scientific_name": primary.get("scientificName"),
                "family": primary.get("family"),
                "confidence": primary["confidence"] / 100 if primary["confidence"] > 1 else primary["confidence"],
                "alternatives": result.get("alternatives", []),
                "source": "local"
            }
        except Exception as e:
            logger.error(f"Local prediction failed: {e}")
            raise

    async def _get_cloud_prediction(self, image_path: str, top_k: int, region: str) -> Dict:
        """Get prediction from HuggingFace InferenceClient."""
        try:
            # Run in thread pool since it's I/O-bound
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.cloud_identifier.identify,
                image_path,
                top_k,
                region,
                "unknown"
            )

            # Normalize format
            return {
                "species": result["plant"],
                "scientific_name": result["plant"],  # Cloud model returns scientific name
                "confidence": result["confidence"],
                "alternatives": result.get("top_k", []),
                "source": "cloud",
                "action": result.get("action")
            }
        except Exception as e:
            logger.error(f"Cloud prediction failed: {e}")
            raise

    def _merge_results(self, local_result: Optional[Dict], cloud_result: Optional[Dict]) -> Dict:
        """
        Merge results from both models and choose the best.
        Priority: highest confidence wins.
        """
        if not local_result and not cloud_result:
            raise RuntimeError("Both models failed to identify the plant")

        # If only one available, use it
        if local_result and not cloud_result:
            logger.info(f"Using local model only (confidence: {local_result['confidence']:.2%})")
            return self._format_response(local_result, None)

        if cloud_result and not local_result:
            logger.info(f"Using cloud model only (confidence: {cloud_result['confidence']:.2%})")
            return self._format_response(None, cloud_result)

        # Both available - choose higher confidence
        local_conf = local_result["confidence"]
        cloud_conf = cloud_result["confidence"]

        if cloud_conf > local_conf:
            logger.info(f"Cloud model has higher confidence ({cloud_conf:.2%} vs {local_conf:.2%})")
            chosen = cloud_result
        else:
            logger.info(f"Local model has higher confidence ({local_conf:.2%} vs {cloud_conf:.2%})")
            chosen = local_result

        return self._format_response(local_result, cloud_result, chosen)

    def _format_response(
        self,
        local_result: Optional[Dict],
        cloud_result: Optional[Dict],
        chosen: Optional[Dict] = None
    ) -> Dict:
        """Format the final response."""
        if chosen is None:
            chosen = local_result or cloud_result

        return {
            "primary": {
                "commonName": chosen["species"],
                "scientificName": chosen.get("scientific_name", chosen["species"]),
                "family": chosen.get("family"),
                "confidence": chosen["confidence"] * 100,  # Convert to percentage
                "source": chosen["source"]
            },
            "alternatives": chosen.get("alternatives", [])[:2],  # Top 2 alternatives
            "metadata": {
                "local_confidence": local_result["confidence"] * 100 if local_result else None,
                "cloud_confidence": cloud_result["confidence"] * 100 if cloud_result else None,
                "method_used": f"hybrid_{chosen['source']}" if local_result and cloud_result else chosen["source"],
                "both_models_used": bool(local_result and cloud_result)
            }
        }


# Global instance
hybrid_identifier = HybridPlantIdentifier()


async def identify_plant_hybrid(image_path: str, region: str = "India", top_k: int = 3) -> Dict:
    """Convenience function for hybrid identification."""
    return await hybrid_identifier.identify(image_path, top_k, region)
