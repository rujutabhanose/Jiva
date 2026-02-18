from sqlalchemy import String, DateTime, Integer, ForeignKey, Text, Boolean, Float
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.db.base import Base


class DiagnosisFeedback(Base):
    __tablename__ = "diagnosis_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True)
    scan_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("scans.id"), nullable=True, index=True)

    # Original diagnosis
    original_condition: Mapped[str] = mapped_column(String)
    original_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # User feedback
    is_correct: Mapped[bool] = mapped_column(Boolean)
    corrected_condition: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Model information for continuous learning
    model_type: Mapped[str | None] = mapped_column(String, nullable=True, index=True)  # "coleaf", "disease", "plant_id"
    model_version_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=True
    )

    # Image reference for training (Supabase URL if saved for training)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Training status
    used_for_training: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
