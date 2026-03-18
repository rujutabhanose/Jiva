import httpx
import logging

logger = logging.getLogger(__name__)


async def get_country_from_ip(ip: str) -> str | None:
    """
    Returns the ISO 3166-1 alpha-2 country code for the given IP address,
    or None if the lookup fails or the IP is private/local.
    Uses ip-api.com (free, no API key, 45 req/min limit).
    """
    # Skip private / loopback addresses
    if not ip or ip in ("127.0.0.1", "::1") or ip.startswith(("10.", "192.168.", "172.")):
        return None

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "status,countryCode"},
            )
            data = resp.json()
            if data.get("status") == "success":
                return data.get("countryCode")
    except Exception as exc:
        logger.warning("IP geolocation lookup failed for %s: %s", ip, exc)

    return None


def extract_client_ip(request) -> str | None:
    """
    Extracts the real client IP from the request, respecting common proxy headers.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can be a comma-separated list; take the first (original client)
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    if request.client:
        return request.client.host

    return None
