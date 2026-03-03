import os
from functools import lru_cache
from typing import Generator

from fastapi import Header, HTTPException

try:
    from src.llm_compass.config import get_settings
except ImportError:  # pragma: no cover - fallback for alternate import layout
    from llm_compass.config import get_settings


def _load_key_set(env_var: str, default: set[str]) -> set[str]:
    raw = os.getenv(env_var, "")
    parsed = {item.strip() for item in raw.split(",") if item.strip()}
    return parsed or default


@lru_cache(maxsize=1)
def _api_keys() -> set[str]:
    return _load_key_set("LLM_COMPASS_API_KEYS", {"dev-api-key"})


@lru_cache(maxsize=1)
def _admin_api_keys() -> set[str]:
    return _load_key_set("LLM_COMPASS_ADMIN_API_KEYS", set())


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    if not x_api_key or x_api_key not in _api_keys():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


async def require_admin_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    if not x_api_key or x_api_key not in _admin_api_keys():
        raise HTTPException(status_code=403, detail="Admin access required")
    return x_api_key


@lru_cache(maxsize=1)
def _get_database() -> object | None:
    try:
        from src.llm_compass.data.database import Database
    except Exception:  # pragma: no cover - fallback for alternate import layout
        try:
            from llm_compass.data.database import Database
        except Exception:
            return None

    try:
        settings = get_settings()
        return Database(settings)
    except Exception:
        return None


def get_db() -> Generator[object | None, None, None]:
    db = _get_database()
    if db is None:
        yield None
        return

    yield from db.get_session()
