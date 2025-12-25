from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models.user import User

def enforce_and_consume_scan(user: User, db: Session):
    """
    Enforces scan limits and consumes one scan atomically.
    Backend is the single source of truth.
    """
    if user.is_premium:
        return

    if user.free_scans_left <= 0:
        raise HTTPException(
            status_code=402,
            detail="Free scans exhausted. Upgrade to continue."
        )

    user.free_scans_left -= 1
    user.scans_used += 1
    db.commit()