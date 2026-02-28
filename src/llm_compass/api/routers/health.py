from fastapi import APIRouter


router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/info")
async def info() -> dict[str, str]:
    return {"name": "llm-compass-api", "version": "0.1.0"}

