import hmac
from typing import Annotated, Optional
from fastapi import Header, HTTPException, Query, status
from app.config import get_settings


def valid_api_key(candidate: str | None) -> bool:
    if not candidate:
        return False
    return hmac.compare_digest(candidate, get_settings().api_key)


async def require_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None,
    token: Annotated[Optional[str], Query()] = None,
) -> None:
    """API key via X-API-Key header, or ?token= for <img>/video consumers."""
    if not valid_api_key(x_api_key) and not valid_api_key(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
