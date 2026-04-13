# app/models/user.py
from sqlalchemy import String, Boolean, DateTime, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import List
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)  # For analytics only, not unique
    platform: Mapped[str | None] = mapped_column(String, nullable=True)
    email: Mapped[str | None] = mapped_column(String, unique=True, index=True, nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    user_type: Mapped[str | None] = mapped_column(String, nullable=True)
    plant_types: Mapped[List[str] | None] = mapped_column(JSON, nullable=True)

    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    free_scans_left: Mapped[int] = mapped_column(Integer, default=1)
    scans_used: Mapped[int] = mapped_column(Integer, default=0)

    # ✅ EMAIL VERIFICATION
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # India free-tier fields (set at registration based on IP geolocation)
    registration_ip: Mapped[str | None] = mapped_column(String, nullable=True)
    ip_country: Mapped[str | None] = mapped_column(String, nullable=True)
    india_free_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    premium_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # None = permanent pro

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)