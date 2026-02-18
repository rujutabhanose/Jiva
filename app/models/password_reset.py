# app/models/password_reset.py
from sqlalchemy import String, Boolean, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timedelta
from app.db.base import Base


class PasswordResetToken(Base):
    """Model for storing password reset OTPs"""
    __tablename__ = "password_reset_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String, index=True, nullable=False)
    otp: Mapped[str] = mapped_column(String(6), nullable=False)
    reset_token: Mapped[str | None] = mapped_column(String, nullable=True)  # Token issued after OTP verification
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False)

    @classmethod
    def create_otp(cls, email: str, otp: str, expires_minutes: int = 15):
        """Factory method to create a new OTP token"""
        return cls(
            email=email,
            otp=otp,
            expires_at=datetime.utcnow() + timedelta(minutes=expires_minutes)
        )

    def is_expired(self) -> bool:
        """Check if the OTP has expired"""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if the OTP is valid (not used and not expired)"""
        return not self.used and not self.is_expired()
