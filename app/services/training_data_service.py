"""
Training Data Service

Handles saving confirmed training samples from user feedback,
managing training datasets, and preparing data for model retraining.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.training_data import TrainingData
from app.models.scan import Scan
from app.models.diagnosis_feedback import DiagnosisFeedback
from app.services.supabase_storage import supabase_storage
from app.core.config import settings

logger = logging.getLogger(__name__)


class TrainingDataService:
    """Service for managing training data from user feedback."""

    def __init__(self, db: Session):
        self.db = db

    async def save_confirmed_sample(
        self,
        user_id: int,
        scan_id: Optional[int],
        confirmed_label: str,
        model_type: str,
        image_source: str,  # File path or base64 data
        original_prediction: Optional[str] = None,
        original_confidence: Optional[float] = None,
    ) -> Optional[TrainingData]:
        """
        Save a confirmed training sample when user confirms diagnosis is correct.

        Args:
            user_id: ID of the user who confirmed
            scan_id: Optional scan ID this feedback relates to
            confirmed_label: The confirmed label (e.g., "nitrogen_deficiency")
            model_type: Model type ("coleaf", "disease", "plant_id")
            image_source: Path to image file or base64 data
            original_prediction: What the model originally predicted
            original_confidence: Original prediction confidence

        Returns:
            Created TrainingData record, or None if failed
        """
        try:
            # Upload image to Supabase
            image_url = None

            if image_source.startswith("data:") or len(image_source) > 500:
                # Looks like base64 data
                image_url = await supabase_storage.upload_training_image_from_base64(
                    base64_data=image_source,
                    model_type=model_type,
                    label=confirmed_label,
                    scan_id=scan_id or 0,
                )
            else:
                # Assume it's a file path
                image_url = await supabase_storage.upload_training_image(
                    image_path=image_source,
                    model_type=model_type,
                    label=confirmed_label,
                    scan_id=scan_id or 0,
                )

            if not image_url:
                logger.warning(
                    f"Failed to upload training image for scan {scan_id}. "
                    "Supabase may not be configured."
                )
                # Still save the record without image URL for tracking
                image_url = image_source  # Store original source as fallback

            # Create training data record
            training_data = TrainingData(
                user_id=user_id,
                scan_id=scan_id,
                image_url=image_url,
                confirmed_label=confirmed_label,
                model_type=model_type,
                original_prediction=original_prediction,
                original_confidence=original_confidence,
                is_used_for_training=False,
            )

            self.db.add(training_data)
            self.db.commit()
            self.db.refresh(training_data)

            logger.info(
                f"Saved training sample: {model_type}/{confirmed_label} "
                f"(scan_id={scan_id}, user_id={user_id})"
            )

            return training_data

        except Exception as e:
            logger.error(f"Error saving training sample: {e}")
            self.db.rollback()
            return None

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected training data.

        Returns:
            Dictionary with counts by model type and label
        """
        stats = {
            "total_samples": 0,
            "unused_samples": 0,  # Not yet used for training
            "by_model_type": {},
        }

        # Total count
        stats["total_samples"] = self.db.query(TrainingData).count()
        stats["unused_samples"] = (
            self.db.query(TrainingData)
            .filter(TrainingData.is_used_for_training == False)
            .count()
        )

        # Count by model type
        for model_type in ["coleaf", "disease", "plant_id"]:
            model_stats = {
                "total": 0,
                "unused": 0,
                "by_label": {},
            }

            # Total for this model type
            model_stats["total"] = (
                self.db.query(TrainingData)
                .filter(TrainingData.model_type == model_type)
                .count()
            )

            # Unused for this model type
            model_stats["unused"] = (
                self.db.query(TrainingData)
                .filter(
                    TrainingData.model_type == model_type,
                    TrainingData.is_used_for_training == False,
                )
                .count()
            )

            # Count by label
            label_counts = (
                self.db.query(
                    TrainingData.confirmed_label,
                    func.count(TrainingData.id).label("count"),
                )
                .filter(TrainingData.model_type == model_type)
                .group_by(TrainingData.confirmed_label)
                .all()
            )

            for label, count in label_counts:
                model_stats["by_label"][label] = count

            stats["by_model_type"][model_type] = model_stats

        return stats

    def get_unused_samples(
        self, model_type: str, limit: Optional[int] = None
    ) -> List[TrainingData]:
        """
        Get training samples that haven't been used for training yet.

        Args:
            model_type: Model type to get samples for
            limit: Optional limit on number of samples

        Returns:
            List of TrainingData records
        """
        query = (
            self.db.query(TrainingData)
            .filter(
                TrainingData.model_type == model_type,
                TrainingData.is_used_for_training == False,
            )
            .order_by(TrainingData.created_at.asc())
        )

        if limit:
            query = query.limit(limit)

        return query.all()

    def mark_samples_as_used(
        self, sample_ids: List[int], model_version_id: int
    ) -> int:
        """
        Mark training samples as used after a training run.

        Args:
            sample_ids: List of TrainingData IDs
            model_version_id: ID of the model version trained

        Returns:
            Number of samples marked
        """
        updated = (
            self.db.query(TrainingData)
            .filter(TrainingData.id.in_(sample_ids))
            .update(
                {
                    TrainingData.is_used_for_training: True,
                    TrainingData.trained_in_version: model_version_id,
                },
                synchronize_session=False,
            )
        )
        self.db.commit()
        return updated

    def can_retrain(self, model_type: str) -> bool:
        """
        Check if there are enough samples to trigger retraining.

        Args:
            model_type: Model type to check

        Returns:
            True if minimum threshold met
        """
        unused_count = (
            self.db.query(TrainingData)
            .filter(
                TrainingData.model_type == model_type,
                TrainingData.is_used_for_training == False,
            )
            .count()
        )

        return unused_count >= settings.MIN_TRAINING_SAMPLES

    def get_samples_for_training(
        self, model_type: str
    ) -> Dict[str, List[TrainingData]]:
        """
        Get all unused samples grouped by label for training.

        Args:
            model_type: Model type to get samples for

        Returns:
            Dictionary mapping labels to lists of TrainingData
        """
        samples = self.get_unused_samples(model_type)

        grouped = {}
        for sample in samples:
            if sample.confirmed_label not in grouped:
                grouped[sample.confirmed_label] = []
            grouped[sample.confirmed_label].append(sample)

        return grouped

    def get_label_distribution(self, model_type: str) -> Dict[str, int]:
        """
        Get the distribution of labels in unused training data.

        Args:
            model_type: Model type to check

        Returns:
            Dictionary mapping labels to counts
        """
        results = (
            self.db.query(
                TrainingData.confirmed_label,
                func.count(TrainingData.id).label("count"),
            )
            .filter(
                TrainingData.model_type == model_type,
                TrainingData.is_used_for_training == False,
            )
            .group_by(TrainingData.confirmed_label)
            .all()
        )

        return {label: count for label, count in results}
