"""
Script to check database contents
"""
from app.db.session import SessionLocal
from app.models.user import User
from app.models.scan import Scan
from app.models.coupon import Coupon
from app.models.coupon_redemption import CouponRedemption

def check_database():
    db = SessionLocal()
    try:
        users_count = db.query(User).count()
        scans_count = db.query(Scan).count()
        coupons_count = db.query(Coupon).count()
        redemptions_count = db.query(CouponRedemption).count()

        print("📊 Database Contents:")
        print(f"   Users: {users_count}")
        print(f"   Scans: {scans_count}")
        print(f"   Coupons: {coupons_count}")
        print(f"   Coupon Redemptions: {redemptions_count}")

        if coupons_count > 0:
            print(f"\n✅ Coupons preserved successfully!")
            coupons = db.query(Coupon).all()
            for coupon in coupons:
                print(f"   - {coupon.code}: {coupon.reward_type} ({coupon.uses_left}/{coupon.max_uses} uses left)")

    finally:
        db.close()

if __name__ == "__main__":
    check_database()
