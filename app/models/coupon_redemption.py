from sqlalchemy import String, DateTime, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.db.base import Base


class CouponRedemption(Base):
    __tablename__ = "coupon_redemptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    coupon_id: Mapped[int] = mapped_column(Integer, ForeignKey("coupons.id"))
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    device_id: Mapped[str] = mapped_column(String, index=True)
    redeemed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
