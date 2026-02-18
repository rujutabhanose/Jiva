from sqlalchemy import String, DateTime, Integer, Float, ForeignKey, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.db.base import Base


class TrainingData(Base):
    """
    Stores confirmed training samples from user feedback.
    When a user confirms a diagnosis is correct, the image and label
    are stored here for model retraining.
    """
    __tablename__ = "training_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True)
    scan_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("scans.id"), nullable=True, index=True)

    # Image storage (Supabase URL)
    image_url: Mapped[str] = mapped_column(Text)

    # Label information
    confirmed_label: Mapped[str] = mapped_column(String, index=True)  # e.g., "nitrogen_deficiency", "powdery_mildew"
    model_type: Mapped[str] = mapped_column(String, index=True)  # "coleaf", "disease", "plant_id"

    # Original prediction (for analysis)
    original_prediction: Mapped[str | None] = mapped_column(String, nullable=True)
    original_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Training status
    is_used_for_training: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    trained_in_version: Mapped[int | None] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
