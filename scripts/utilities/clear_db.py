"""
Script to clear all database entries except coupons
"""
from app.db.session import SessionLocal
from app.models.user import User
from app.models.scan import Scan
from app.models.coupon_redemption import CouponRedemption

def clear_database():
    db = SessionLocal()
    try:
        # Delete in order to respect foreign key constraints
        # 1. Delete coupon redemptions first (has FK to both user and coupon)
        redemptions_count = db.query(CouponRedemption).count()
        db.query(CouponRedemption).delete()
        print(f"✅ Deleted {redemptions_count} coupon redemptions")

        # 2. Delete scans (has FK to user)
        scans_count = db.query(Scan).count()
        db.query(Scan).delete()
        print(f"✅ Deleted {scans_count} scans")

        # 3. Delete users (no dependencies on it anymore)
        users_count = db.query(User).count()
        db.query(User).delete()
        print(f"✅ Deleted {users_count} users")

        # Commit all deletions
        db.commit()
        print("\n✨ Database cleared successfully! Coupons preserved.")

    except Exception as e:
        db.rollback()
        print(f"❌ Error clearing database: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("🗑️  Clearing database (preserving coupons)...\n")
    clear_database()
