from sqlalchemy import String, DateTime, Integer, Float, JSON, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.db.base import Base


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True)
    device_id: Mapped[str] = mapped_column(String, index=True)

    # Scan metadata
    mode: Mapped[str] = mapped_column(String)  # 'diagnosis' | 'identification'
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # Base64 or URL

    # Model versioning - tracks which model version made this prediction
    model_version_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=True, index=True
    )

    # Diagnosis/Identification results
    condition_name: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)

    # Detailed information (JSON)
    diagnosis_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # Full diagnosis object
    symptoms: Mapped[list | None] = mapped_column(JSON, nullable=True)  # List of symptoms
    causes: Mapped[list | None] = mapped_column(JSON, nullable=True)  # List of causes
    treatment: Mapped[list | None] = mapped_column(JSON, nullable=True)  # List of treatments

    # User notes
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Additional metadata
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    severity: Mapped[str | None] = mapped_column(String, nullable=True)
    health_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
