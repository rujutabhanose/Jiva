from typing import Generator
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from app.db.session import SessionLocal
from app.core.security import get_current_user_email
from app.models.user import User


def get_db() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI routes.

    Usage:
        @router.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db session here
            pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user_email)
) -> User:
    """
    Get the current authenticated user from JWT token.

    Usage:
        @router.get("/endpoint")
        def endpoint(current_user: User = Depends(get_current_user)):
            # current_user is authenticated User object
            pass
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user