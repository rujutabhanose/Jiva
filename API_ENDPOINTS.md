# Backend API Endpoints Documentation

## Base URL
- Development: `http://localhost:8000`
- Production: Configure in settings

## All Available Endpoints

### Authentication (`/api/v1/auth`)
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/logout` - User logout

### Users (`/api/v1/users`)
- `POST /api/v1/users/device` - Get or create device user ✅ (Used by mobile)
- `POST /api/v1/users/scan-count` - Update scan count ✅ (Used by mobile)
- `POST /api/v1/users/upgrade` - Upgrade to pro ✅ (Used by mobile)
- `POST /api/v1/users/redeem-coupon` - Redeem coupon code ✅ (Used by mobile)
- `POST /api/v1/users/validate-coupon` - Validate coupon code ✅ (Used by mobile)
- `GET /api/v1/users/me` - Get current user profile
- `PUT /api/v1/users/me` - Update user profile
- `DELETE /api/v1/users/me` - Delete user account

### Plant Identification (`/api/v1/identify`)
- `POST /api/v1/identify` - Identify plant (FREE, no scan limit) ✅ (Used by mobile)

### Plant Diagnosis (`/api/v1/diagnose`)
- `POST /api/v1/diagnose` - Diagnose plant health (PAID, consumes scan) ✅ (Used by mobile)

### Scans (`/api/v1/scans`)
- `POST /api/v1/scans` - Create new scan ✅ (Used by mobile)
- `GET /api/v1/scans` - Get scan history ✅ (Used by mobile)
- `GET /api/v1/scans/{scan_id}` - Get specific scan ✅ (Used by mobile)
- `PUT /api/v1/scans/{scan_id}/notes` - Update scan notes ✅ (Used by mobile)
- `DELETE /api/v1/scans/{scan_id}` - Delete scan ✅ (Used by mobile)
- `GET /api/v1/scans/stats/summary` - Get scan statistics ✅ (Used by mobile)

## Health Check
- `GET /` - Root endpoint
- `GET /health` - Health check endpoint

## Mobile App Endpoint Mapping

All mobile app endpoints are properly mapped:

| Mobile Endpoint | Backend Endpoint | Status |
|----------------|------------------|--------|
| `GET_OR_CREATE_DEVICE_USER` | `POST /api/v1/users/device` | ✅ |
| `UPDATE_SCAN_COUNT` | `POST /api/v1/users/scan-count` | ✅ |
| `UPGRADE_TO_PRO` | `POST /api/v1/users/upgrade` | ✅ |
| `REDEEM_COUPON` | `POST /api/v1/users/redeem-coupon` | ✅ |
| `VALIDATE_COUPON` | `POST /api/v1/users/validate-coupon` | ✅ |
| `DIAGNOSE` | `POST /api/v1/diagnose` | ✅ |
| `IDENTIFY` | `POST /api/v1/identify` | ✅ |
| `CREATE_SCAN` | `POST /api/v1/scans` | ✅ |
| `GET_SCANS` | `GET /api/v1/scans` | ✅ |
| `GET_SCAN(scanId)` | `GET /api/v1/scans/{scan_id}` | ✅ |
| `UPDATE_SCAN_NOTES(scanId)` | `PUT /api/v1/scans/{scan_id}/notes` | ✅ |
| `DELETE_SCAN(scanId)` | `DELETE /api/v1/scans/{scan_id}` | ✅ |
| `GET_SCAN_STATS` | `GET /api/v1/scans/stats/summary` | ✅ |

## Notes

1. **Plant Identification** is FREE - does not consume scans
2. **Plant Diagnosis** is PAID - consumes one scan per request
3. All endpoints require `device_id` for device-based authentication
4. Scan endpoints support both `diagnosis` and `identification` modes

