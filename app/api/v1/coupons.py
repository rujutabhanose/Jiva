from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.api.deps import get_db, get_current_user
from app.models.coupon import Coupon
from app.models.coupon_redemption import CouponRedemption
from app.models.user import User

router = APIRouter()


class ValidateCouponRequest(BaseModel):
    code: str


class ValidateCouponResponse(BaseModel):
    valid: bool
    offer_tag: str | None = None  # RevenueCat offer tag to apply on Android


class RedeemCouponRequest(BaseModel):
    code: str


class RedeemCouponResponse(BaseModel):
    message: str
    plan_type: str


@router.post("/validate", response_model=ValidateCouponResponse)
def validate_coupon(
    request: ValidateCouponRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Check if a coupon is valid without redeeming it. Returns the associated offer tag."""
    code = request.code.strip().upper()

    coupon = db.query(Coupon).filter(Coupon.code == code).first()
    if not coupon or not coupon.is_active:
        return {"valid": False}

    if coupon.expires_at and coupon.expires_at < datetime.utcnow():
        return {"valid": False}

    if coupon.max_uses is not None and coupon.current_uses >= coupon.max_uses:
        return {"valid": False}

    already_redeemed = db.query(CouponRedemption).filter(
        CouponRedemption.coupon_id == coupon.id,
        CouponRedemption.user_id == current_user.id,
    ).first()
    if already_redeemed:
        return {"valid": False}

    return {"valid": True, "offer_tag": coupon.offer_tag if hasattr(coupon, "offer_tag") else None}


@router.post("/redeem", response_model=RedeemCouponResponse)
def redeem_coupon(
    request: RedeemCouponRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    code = request.code.strip().upper()

    coupon = db.query(Coupon).filter(Coupon.code == code).first()
    if not coupon or not coupon.is_active:
        raise HTTPException(status_code=404, detail="Invalid or expired coupon code")

    if coupon.expires_at and coupon.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="This coupon has expired")

    if coupon.max_uses is not None and coupon.current_uses >= coupon.max_uses:
        raise HTTPException(status_code=400, detail="This coupon has reached its usage limit")

    already_redeemed = db.query(CouponRedemption).filter(
        CouponRedemption.coupon_id == coupon.id,
        CouponRedemption.user_id == current_user.id,
    ).first()
    if already_redeemed:
        raise HTTPException(status_code=400, detail="You have already redeemed this coupon")

    # Grant premium access (timed or permanent)
    current_user.is_premium = True
    current_user.free_scans_left = -1  # unlimited
    if coupon.duration_days is not None:
        current_user.premium_expires_at = datetime.utcnow() + timedelta(days=coupon.duration_days)
    else:
        current_user.premium_expires_at = None  # permanent

    # Record redemption
    redemption = CouponRedemption(
        coupon_id=coupon.id,
        user_id=current_user.id,
        device_id=current_user.device_id or "",
    )
    coupon.current_uses += 1

    db.add(redemption)
    db.commit()

    return {"message": "Coupon redeemed! You now have Pro access.", "plan_type": coupon.plan_type}
