from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


async def require_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Valida a API Key enviada no header X-API-Key.
    Se API_KEYS estiver vazio, a autenticação é desabilitada (útil em desenvolvimento).
    """
    if not settings.API_KEYS:
        return "dev"

    if not api_key or api_key not in settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key.",
        )
    return api_key
