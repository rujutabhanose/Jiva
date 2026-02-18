from sqlalchemy import String, DateTime, Integer, Float, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.db.base import Base


class ModelVersion(Base):
    """
    Tracks different versions of ML models.
    Enables model versioning, rollback, and A/B testing.
    """
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Model identification
    model_type: Mapped[str] = mapped_column(String, index=True)  # "coleaf", "disease", "plant_id"
    version: Mapped[str] = mapped_column(String)  # e.g., "v1.0.0", "v1.1.0"

    # File location
    file_path: Mapped[str] = mapped_column(Text)  # Path to model file (local or Supabase URL)

    # Performance metrics
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Training info
    training_samples_count: Mapped[int] = mapped_column(Integer, default=0)
    epochs_trained: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)  # Original pre-trained model

    # Notes
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
