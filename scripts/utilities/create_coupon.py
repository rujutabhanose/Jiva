#!/usr/bin/env python3
"""
Quick script to create a new coupon code

Usage:
    python create_coupon.py

Or run interactively to create custom coupons
"""

from datetime import datetime, timedelta
from app.db.session import SessionLocal
from app.models.coupon import Coupon


def create_coupon_interactive():
    """Interactive coupon creation"""
    print("\n=== Jiva Plants Coupon Creator ===\n")

    # Get coupon code
    code = input("Coupon code (will be uppercased): ").strip().upper()
    if not code:
        print("❌ Coupon code is required!")
        return

    # Get description
    description = input("Description: ").strip() or f"{code} promotional code"

    # Get plan type
    print("\nPlan types:")
    print("  1. Monthly")
    print("  2. Yearly")
    print("  3. Lifetime")
    plan_choice = input("Select plan type (1-3): ").strip()

    plan_map = {"1": "monthly", "2": "yearly", "3": "lifetime"}
    plan_type = plan_map.get(plan_choice, "monthly")

    # Get max uses
    max_uses_input = input("Max uses (press Enter for unlimited): ").strip()
    max_uses = int(max_uses_input) if max_uses_input else None

    # Get expiration
    expires_input = input("Days until expiration (press Enter for never): ").strip()
    if expires_input:
        expires_at = datetime.utcnow() + timedelta(days=int(expires_input))
    else:
        expires_at = None

    # Confirm
    print("\n--- Coupon Summary ---")
    print(f"Code: {code}")
    print(f"Description: {description}")
    print(f"Plan Type: {plan_type}")
    print(f"Max Uses: {max_uses or 'Unlimited'}")
    print(f"Expires: {expires_at.strftime('%Y-%m-%d') if expires_at else 'Never'}")

    confirm = input("\nCreate this coupon? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled")
        return

    # Create coupon
    db = SessionLocal()
    try:
        # Check if code already exists
        existing = db.query(Coupon).filter(Coupon.code == code).first()
        if existing:
            print(f"❌ Coupon code '{code}' already exists!")
            return

        coupon = Coupon(
            code=code,
            description=description,
            plan_type=plan_type,
            max_uses=max_uses,
            current_uses=0,
            is_active=True,
            expires_at=expires_at
        )

        db.add(coupon)
        db.commit()

        print(f"\n✅ Successfully created coupon: {code}")
        print(f"   Users can now redeem this code for {plan_type} access!")

    except Exception as e:
        print(f"❌ Error creating coupon: {e}")
        db.rollback()
    finally:
        db.close()


def create_quick_coupon(code, plan_type="monthly", days=None, max_uses=None):
    """Quick coupon creation with defaults.

    Args:
        code:      Coupon code (uppercased automatically).
        plan_type: 'monthly' | 'yearly' | 'lifetime'.
        days:      Days until expiration, or None for no expiry.
        max_uses:  Maximum redemptions, or None for unlimited.
    """
    db = SessionLocal()
    try:
        existing = db.query(Coupon).filter(Coupon.code == code.upper()).first()
        if existing:
            print(f"❌ Coupon '{code}' already exists")
            return False

        expires_at = datetime.utcnow() + timedelta(days=days) if days else None

        coupon = Coupon(
            code=code.upper(),
            description=f"{code} promotional code",
            plan_type=plan_type,
            max_uses=max_uses,
            current_uses=0,
            is_active=True,
            expires_at=expires_at,
        )

        db.add(coupon)
        db.commit()

        expiry_str = f"expires in {days} days" if days else "no expiry"
        uses_str = f"{max_uses} uses" if max_uses else "unlimited uses"
        print(f"✅ Created: {code.upper()} ({plan_type}, {uses_str}, {expiry_str})")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Quick mode: python create_coupon.py MYCODE [plan_type] [days_or_0_for_none] [max_uses_or_0_for_unlimited]
        code = sys.argv[1]
        plan = sys.argv[2] if len(sys.argv) > 2 else "monthly"
        days_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        max_uses_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        # 0 means "no limit / no expiry"
        days = days_arg if days_arg > 0 else None
        max_uses = max_uses_arg if max_uses_arg > 0 else None

        create_quick_coupon(code, plan, days, max_uses)
    else:
        # Interactive mode
        create_coupon_interactive()
