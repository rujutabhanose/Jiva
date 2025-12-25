# Coupon Code System

## Overview

The coupon code system allows users to upgrade to premium by redeeming promotional codes. This is an alternative payment method alongside traditional payment plans (monthly/yearly subscriptions).

## Features

- **Flexible Plan Types**: Monthly, Yearly, or Lifetime access
- **Usage Limits**: Set maximum number of redemptions per coupon
- **Expiration Dates**: Optional expiration for time-limited promotions
- **One-time Redemption**: Each device can only redeem a specific coupon once
- **Validation API**: Check coupon validity before redemption

## Database Models

### Coupon Model

Location: `backend/app/models/coupon.py`

```python
class Coupon:
    id: int                      # Primary key
    code: str                    # Unique coupon code (uppercase)
    description: str             # Human-readable description
    plan_type: str               # 'monthly' | 'yearly' | 'lifetime'
    max_uses: int | None         # Maximum redemptions (None = unlimited)
    current_uses: int            # Current redemption count
    is_active: bool              # Whether coupon is active
    expires_at: datetime | None  # Expiration date (None = never expires)
    created_at: datetime         # Creation timestamp
```

### CouponRedemption Model

Location: `backend/app/models/coupon_redemption.py`

```python
class CouponRedemption:
    id: int                 # Primary key
    coupon_id: int          # Foreign key to Coupon
    user_id: int            # Foreign key to User
    device_id: str          # Device identifier
    redeemed_at: datetime   # Redemption timestamp
```

## API Endpoints

### 1. Redeem Coupon

**Endpoint**: `POST /api/v1/users/redeem-coupon`

**Request**:
```json
{
  "device_id": "1234567890-abcdef",
  "coupon_code": "WELCOME50"
}
```

**Success Response** (200):
```json
{
  "success": true,
  "user_id": 42,
  "device_id": "1234567890-abcdef",
  "is_premium": true,
  "plan_type": "monthly",
  "message": "Successfully redeemed coupon! You now have monthly premium access."
}
```

**Error Responses**:
- `404`: Invalid coupon code
- `400`: Coupon expired, inactive, max uses reached, or already redeemed
- `400`: User is already premium

### 2. Validate Coupon

**Endpoint**: `POST /api/v1/users/validate-coupon`

**Request**:
```json
{
  "device_id": "1234567890-abcdef",
  "coupon_code": "BETA2024"
}
```

**Valid Coupon Response** (200):
```json
{
  "valid": true,
  "plan_type": "lifetime",
  "description": "Beta tester lifetime access",
  "message": "Coupon is valid!"
}
```

**Invalid Coupon Response** (200):
```json
{
  "valid": false,
  "message": "Coupon has expired"
}
```

## Mobile App Integration

### 1. UpgradeModal Component

Location: `mobile/src/components/UpgradeModal.tsx`

The upgrade modal now includes a coupon code input section with:
- Text input for coupon code (auto-uppercase)
- Apply button with loading state
- Error and success messages
- Automatic modal close on successful redemption

### 2. AppContext

Location: `mobile/src/context/AppContext.tsx`

Added `redeemCoupon` function:

```typescript
const redeemCoupon = async (couponCode: string): Promise<{ success: boolean; message: string }> => {
  // Calls API, updates local state, and returns result
}
```

## Sample Coupons

Run the seeding script to add sample coupons:

```bash
cd backend
python -m app.scripts.seed_coupons
```

This creates the following test coupons:

| Code | Plan Type | Max Uses | Expires | Description |
|------|-----------|----------|---------|-------------|
| `WELCOME50` | Monthly | 50 | 30 days | Welcome offer |
| `BETA2024` | Lifetime | 100 | Never | Beta tester access |
| `NURSERY100` | Yearly | Unlimited | 90 days | Nursery professional |
| `EARLYBIRD` | Yearly | 20 | 7 days | Early bird special |
| `TESTCODE` | Monthly | Unlimited | Never | Development testing |

## Database Migration

After creating the coupon models, run database migrations:

```bash
cd backend
alembic revision --autogenerate -m "Add coupon and coupon_redemption tables"
alembic upgrade head
```

Then seed the database:

```bash
python -m app.scripts.seed_coupons
```

## Validation Rules

The system validates coupons based on:

1. **Existence**: Coupon code must exist in database
2. **Active Status**: `is_active` must be `true`
3. **Expiration**: Current time must be before `expires_at` (if set)
4. **Usage Limit**: `current_uses` must be less than `max_uses` (if set)
5. **Duplicate Prevention**: Device cannot redeem same coupon twice
6. **User Status**: User must not already be premium

## User Flow

1. User exhausts free scan
2. Upgrade modal appears with payment options
3. User scrolls down to "OR USE COUPON CODE" section
4. User enters coupon code (automatically uppercased)
5. User clicks "Apply" button
6. System validates coupon and shows result:
   - **Valid**: Success message, user upgraded, modal closes after 2 seconds
   - **Invalid**: Error message displayed

## Admin Management

### Creating New Coupons

```python
from app.db.session import SessionLocal
from app.models.coupon import Coupon
from datetime import datetime, timedelta

db = SessionLocal()

coupon = Coupon(
    code="PROMO2024",
    description="2024 Promotion",
    plan_type="yearly",
    max_uses=100,
    current_uses=0,
    is_active=True,
    expires_at=datetime.utcnow() + timedelta(days=60)
)

db.add(coupon)
db.commit()
```

### Deactivating Coupons

```python
coupon = db.query(Coupon).filter(Coupon.code == "OLDCODE").first()
coupon.is_active = False
db.commit()
```

### Viewing Redemptions

```python
from app.models.coupon_redemption import CouponRedemption

redemptions = db.query(CouponRedemption).filter(
    CouponRedemption.coupon_id == coupon.id
).all()

for redemption in redemptions:
    print(f"User {redemption.user_id} redeemed on {redemption.redeemed_at}")
```

## Security Considerations

1. **Case Insensitive**: Codes are converted to uppercase before lookup
2. **One-time Use Per Device**: Prevents abuse by same user
3. **Usage Tracking**: `current_uses` incremented atomically
4. **Expiration**: Time-based expiration prevents indefinite validity
5. **Active Flag**: Allows instant deactivation without deletion

## Testing

### Backend Tests

```bash
# Test coupon redemption
curl -X POST http://localhost:8000/api/v1/users/redeem-coupon \
  -H "Content-Type: application/json" \
  -d '{"device_id": "test-device", "coupon_code": "TESTCODE"}'

# Test coupon validation
curl -X POST http://localhost:8000/api/v1/users/validate-coupon \
  -H "Content-Type: application/json" \
  -d '{"device_id": "test-device", "coupon_code": "BETA2024"}'
```

### Mobile App Testing

1. Start backend server
2. Run mobile app
3. Use free scan
4. Open upgrade modal
5. Enter test coupon code: `TESTCODE`
6. Verify successful redemption

## Future Enhancements

Potential improvements:

1. **Discount Coupons**: Percentage or fixed amount discounts instead of full access
2. **Referral Codes**: Auto-generated codes for user referrals
3. **Analytics Dashboard**: Track coupon performance and usage
4. **Batch Creation**: Generate multiple unique codes at once
5. **Email Integration**: Send codes via email campaigns
6. **A/B Testing**: Test different coupon offers
7. **Geographic Restrictions**: Limit coupons by region
8. **User Type Restrictions**: Restrict to specific user types (nursery, farmer, etc.)

## Monitoring

Track these metrics:

- Total coupons created
- Active vs inactive coupons
- Redemption rate per coupon
- Most popular coupons
- Expiration approaching
- Failed redemption attempts
- Revenue impact (for discount coupons)
