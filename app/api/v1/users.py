from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.api.deps import get_db
from app.models import user as user_model
from app.models.coupon import Coupon
from app.models.coupon_redemption import CouponRedemption
from app.models import scan as scan_model
from app.core.security import get_current_user_email

router = APIRouter()


class UserProfile(BaseModel):
    id: int
    email: EmailStr
    name: str
    country: Optional[str] = None
    userType: Optional[str] = None  # 'Home gardener' | 'Nursery' | 'Farmer' | 'Other'
    plantTypes: Optional[List[str]] = None  # Array of plant types
    onboardingCompleted: Optional[bool] = False
    created_at: str
    is_premium: bool = False
    free_scans_left: int = 1


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    country: Optional[str] = None
    userType: Optional[str] = None
    plantTypes: Optional[List[str]] = None


class DeviceUserRequest(BaseModel):
    device_id: str


class ScanCountRequest(BaseModel):
    device_id: str
    scan_count: int


class UpgradeRequest(BaseModel):
    device_id: str
    plan: str  # 'monthly' | 'yearly'


class CouponRedeemRequest(BaseModel):
    device_id: str
    coupon_code: str


@router.get("/me", response_model=UserProfile)
async def get_current_user(db: Session = Depends(get_db)):
    """Get current user profile"""
    # TODO: Add authentication to get current user ID
    # For now, get the first user or create a test user
    user = db.query(user_model.User).first()

    if not user:
        # Create a test user if none exists
        user = user_model.User(
            email="test@example.com",
            hashed_password="not_implemented",
            name="Test User",
            free_scans_left=1
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "country": user.country,
        "userType": user.user_type,
        "plantTypes": user.plant_types,
        "onboardingCompleted": bool(user.user_type or user.plant_types),
        "created_at": user.created_at.isoformat(),
        "is_premium": user.is_premium,
        "free_scans_left": user.free_scans_left
    }


@router.put("/me", response_model=UserProfile)
async def update_profile(request: UpdateProfileRequest, db: Session = Depends(get_db)):
    """Update user profile"""
    # TODO: Add authentication to get current user ID
    user = db.query(user_model.User).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update user fields
    if request.name is not None:
        user.name = request.name
    if request.email is not None:
        user.email = request.email
    if request.country is not None:
        user.country = request.country
    if request.userType is not None:
        user.user_type = request.userType
    if request.plantTypes is not None:
        user.plant_types = request.plantTypes

    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "country": user.country,
        "userType": user.user_type,
        "plantTypes": user.plant_types,
        "onboardingCompleted": bool(user.user_type or user.plant_types),
        "created_at": user.created_at.isoformat(),
        "is_premium": user.is_premium,
        "free_scans_left": user.free_scans_left
    }


@router.delete("/me")
async def delete_account(
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user_email)
):
    """Delete user account and all associated data"""
    # Get current user from email
    current_user = db.query(user_model.User).filter(user_model.User.email == email).first()

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete all user's scans
    db.query(scan_model.Scan).filter(
        scan_model.Scan.user_id == current_user.id
    ).delete(synchronize_session=False)

    # Delete all user's coupon redemptions
    db.query(CouponRedemption).filter(
        CouponRedemption.user_id == current_user.id
    ).delete(synchronize_session=False)

    # Delete the user
    db.delete(current_user)
    db.commit()

    return {"message": "Account deleted successfully"}


@router.post("/device")
async def get_or_create_device_user(request: DeviceUserRequest, db: Session = Depends(get_db)):
    """Get or create a user by device ID (for guest users)"""
    # Check if user exists with this device_id
    user = db.query(user_model.User).filter(user_model.User.device_id == request.device_id).first()

    if not user:
        # Create a new guest user
        user = user_model.User(
            device_id=request.device_id,
            name=f"Guest User",
            free_scans_left=1,
            scans_used=0,
            is_premium=False
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"[DEBUG] Created new device user {user.id} with is_premium={user.is_premium}")
    else:
        print(f"[DEBUG] Found existing device user {user.id} with is_premium={user.is_premium}")

    response_data = {
        "user_id": user.id,
        "device_id": user.device_id,
        "is_premium": user.is_premium,
        "free_scans_left": user.free_scans_left,
        "scans_used": user.scans_used,
        "scans_limit": 999999 if user.is_premium else 1  # Use large number instead of infinity for JSON compatibility
    }
    print(f"[DEBUG] Returning device user data: is_premium={response_data['is_premium']}")

    return response_data


@router.post("/scan-count")
async def update_scan_count(request: ScanCountRequest, db: Session = Depends(get_db)):
    """Update scan count for a device user"""
    user = db.query(user_model.User).filter(user_model.User.device_id == request.device_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update scan counts
    user.scans_used = request.scan_count

    if not user.is_premium:
        user.free_scans_left = max(0, 1 - request.scan_count)

    db.commit()
    db.refresh(user)

    return {
        "user_id": user.id,
        "device_id": user.device_id,
        "is_premium": user.is_premium,
        "free_scans_left": user.free_scans_left,
        "scans_used": user.scans_used
    }


@router.post("/upgrade")
async def upgrade_to_pro(request: UpgradeRequest, db: Session = Depends(get_db)):
    """Upgrade a user to pro/premium"""
    user = db.query(user_model.User).filter(user_model.User.device_id == request.device_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # TODO: Integrate with payment provider (Stripe, RevenueCat, etc.)
    # For now, just upgrade the user
    user.is_premium = True
    user.free_scans_left = -1  # Unlimited

    db.commit()
    db.refresh(user)

    return {
        "user_id": user.id,
        "device_id": user.device_id,
        "is_premium": user.is_premium,
        "plan": request.plan,
        "message": "Successfully upgraded to Pro!"
    }


@router.post("/redeem-coupon")
async def redeem_coupon(request: CouponRedeemRequest, db: Session = Depends(get_db)):
    """Redeem a coupon code to upgrade to premium"""
    # Get user
    user = db.query(user_model.User).filter(user_model.User.device_id == request.device_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if user is already premium
    if user.is_premium:
        raise HTTPException(status_code=400, detail="User is already premium")

    # Get coupon
    coupon = db.query(Coupon).filter(
        Coupon.code == request.coupon_code.upper()
    ).first()

    if not coupon:
        raise HTTPException(status_code=404, detail="Invalid coupon code")

    # Validate coupon
    if not coupon.is_active:
        raise HTTPException(status_code=400, detail="Coupon is no longer active")

    # Check expiration
    if coupon.expires_at and coupon.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Coupon has expired")

    # Check max uses
    if coupon.max_uses and coupon.current_uses >= coupon.max_uses:
        raise HTTPException(status_code=400, detail="Coupon has reached maximum uses")

    # Check if user already redeemed this coupon
    existing_redemption = db.query(CouponRedemption).filter(
        CouponRedemption.coupon_id == coupon.id,
        CouponRedemption.device_id == request.device_id
    ).first()

    if existing_redemption:
        raise HTTPException(status_code=400, detail="Coupon already redeemed")

    # Redeem coupon
    user.is_premium = True
    user.free_scans_left = -1  # Unlimited

    # Record redemption
    redemption = CouponRedemption(
        coupon_id=coupon.id,
        user_id=user.id,
        device_id=request.device_id
    )
    db.add(redemption)

    # Increment coupon usage
    coupon.current_uses += 1

    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "user_id": user.id,
        "device_id": user.device_id,
        "is_premium": user.is_premium,
        "plan_type": coupon.plan_type,
        "message": f"Successfully redeemed coupon! You now have {coupon.plan_type} premium access."
    }


@router.post("/validate-coupon")
async def validate_coupon(request: CouponRedeemRequest, db: Session = Depends(get_db)):
    """Validate a coupon code without redeeming it"""
    # Get coupon
    coupon = db.query(Coupon).filter(
        Coupon.code == request.coupon_code.upper()
    ).first()

    if not coupon:
        return {
            "valid": False,
            "message": "Invalid coupon code"
        }

    # Check if active
    if not coupon.is_active:
        return {
            "valid": False,
            "message": "Coupon is no longer active"
        }

    # Check expiration
    if coupon.expires_at and coupon.expires_at < datetime.utcnow():
        return {
            "valid": False,
            "message": "Coupon has expired"
        }

    # Check max uses
    if coupon.max_uses and coupon.current_uses >= coupon.max_uses:
        return {
            "valid": False,
            "message": "Coupon has reached maximum uses"
        }

    # Check if already redeemed by this device
    existing_redemption = db.query(CouponRedemption).filter(
        CouponRedemption.coupon_id == coupon.id,
        CouponRedemption.device_id == request.device_id
    ).first()

    if existing_redemption:
        return {
            "valid": False,
            "message": "Coupon already redeemed"
        }

    return {
        "valid": True,
        "plan_type": coupon.plan_type,
        "description": coupon.description or f"Upgrade to {coupon.plan_type} premium",
        "message": "Coupon is valid!"
    }


class CreateCouponRequest(BaseModel):
    code: str
    description: Optional[str] = None
    plan_type: str  # 'monthly' | 'yearly' | 'lifetime'
    max_uses: Optional[int] = None
    expires_in_days: Optional[int] = None  # Days until expiration, None = never expires


@router.post("/coupons/create")
async def create_coupon(request: CreateCouponRequest, db: Session = Depends(get_db)):
    """Create a new coupon code (Admin endpoint)"""
    # TODO: Add authentication/authorization check for admin users
    
    # Check if coupon code already exists
    existing = db.query(Coupon).filter(Coupon.code == request.code.upper()).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Coupon code '{request.code.upper()}' already exists")
    
    # Calculate expiration date if provided
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
    
    # Create coupon
    coupon = Coupon(
        code=request.code.upper(),
        description=request.description or f"{request.code} promotional code",
        plan_type=request.plan_type,
        max_uses=request.max_uses,
        current_uses=0,
        is_active=True,
        expires_at=expires_at
    )
    
    db.add(coupon)
    db.commit()
    db.refresh(coupon)
    
    return {
        "success": True,
        "coupon": {
            "id": coupon.id,
            "code": coupon.code,
            "description": coupon.description,
            "plan_type": coupon.plan_type,
            "max_uses": coupon.max_uses,
            "expires_at": coupon.expires_at.isoformat() if coupon.expires_at else None,
            "message": f"Coupon '{coupon.code}' created successfully"
        }
    }
