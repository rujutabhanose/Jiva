"""
Grant permanent pro access to specific users by email.
Usage: python -m app.scripts.grant_pro_access
"""
from app.db.session import SessionLocal
from app.models.user import User

PRO_EMAILS = [
    "rujuta.bhanose@gmail.com",
    "maheshathalye@hotmail.com",
]


def grant_pro_access():
    db = SessionLocal()
    try:
        for email in PRO_EMAILS:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                print(f"⚠️  User not found: {email} (they may not have registered yet)")
                continue

            if user.is_premium:
                print(f"ℹ️  Already pro: {email}")
            else:
                user.is_premium = True
                user.free_scans_left = -1  # -1 = unlimited
                print(f"✅ Granted pro access: {email}")

        db.commit()
        print("\nDone.")
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    grant_pro_access()
