"""
Seed the database with sample coupon codes for testing
"""
from datetime import datetime, timedelta
from app.db.session import SessionLocal
from app.models.coupon import Coupon


def seed_coupons():
    db = SessionLocal()

    try:
        # Check if coupons already exist
        existing = db.query(Coupon).first()
        if existing:
            print("Coupons already exist. Skipping seed.")
            return

        # Create sample coupons
        coupons = [
            Coupon(
                code="WELCOME50",
                description="Welcome offer - 50 free users",
                plan_type="monthly",
                max_uses=50,
                current_uses=0,
                is_active=True,
                expires_at=datetime.utcnow() + timedelta(days=30)
            ),
            Coupon(
                code="BETA2024",
                description="Beta tester lifetime access",
                plan_type="lifetime",
                max_uses=100,
                current_uses=0,
                is_active=True,
                expires_at=None  # No expiration
            ),
            Coupon(
                code="NURSERY100",
                description="Nursery professional yearly access",
                plan_type="yearly",
                max_uses=None,  # Unlimited uses
                current_uses=0,
                is_active=True,
                expires_at=datetime.utcnow() + timedelta(days=90)
            ),
            Coupon(
                code="EARLYBIRD",
                description="Early bird special - unlimited yearly",
                plan_type="yearly",
                max_uses=20,
                current_uses=0,
                is_active=True,
                expires_at=datetime.utcnow() + timedelta(days=7)
            ),
            Coupon(
                code="TESTCODE",
                description="Test coupon for development",
                plan_type="monthly",
                max_uses=None,
                current_uses=0,
                is_active=True,
                expires_at=None
            ),
        ]

        db.add_all(coupons)
        db.commit()

        print(f"✅ Successfully seeded {len(coupons)} coupons:")
        for coupon in coupons:
            print(f"   - {coupon.code}: {coupon.description}")
            print(f"     Plan: {coupon.plan_type}, Max uses: {coupon.max_uses or 'Unlimited'}")
            if coupon.expires_at:
                print(f"     Expires: {coupon.expires_at.strftime('%Y-%m-%d')}")
            else:
                print(f"     Expires: Never")
            print()

    except Exception as e:
        print(f"❌ Error seeding coupons: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    seed_coupons()
