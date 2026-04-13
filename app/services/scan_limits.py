from fastapi import HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.models.user import User

def enforce_and_consume_scan(user: User, db: Session):
    """
    Enforces scan limits and consumes one scan atomically.
    Backend is the single source of truth.
    """
    # Expire timed premium grants
    if user.is_premium and user.premium_expires_at and user.premium_expires_at < datetime.utcnow():
        user.is_premium = False
        user.free_scans_left = max(0, 1 - user.scans_used)
        db.commit()

    if user.is_premium:
        return

    # Users whose registration IP was in India get free access for 6 months
    if user.india_free_expires_at and user.india_free_expires_at > datetime.utcnow():
        return

    if user.free_scans_left <= 0:
        raise HTTPException(
            status_code=402,
            detail="Free scans exhausted. Upgrade to continue."
        )

    user.free_scans_left -= 1
    user.scans_used += 1
    db.commit()