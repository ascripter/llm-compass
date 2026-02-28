from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import health_router, query_router
from .schemas.common import APIError, ErrorDetail


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = APIError(errors=[ErrorDetail(code=code, message=message)]).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


app = FastAPI(
    title="LLM Compass API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(query_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    del request

    code_map = {
        400: "VALIDATION_ERROR",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "UNPROCESSABLE_ENTITY",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    code = code_map.get(exc.status_code, "HTTP_ERROR")

    if isinstance(exc.detail, dict):
        detail_code = str(exc.detail.get("code", code))
        detail_message = str(exc.detail.get("message", "Request failed"))
        return _error_response(exc.status_code, detail_code, detail_message)

    return _error_response(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    del request
    message = "; ".join(
        f"{'.'.join(str(part) for part in err.get('loc', []))}: {err.get('msg', 'Invalid value')}"
        for err in exc.errors()
    )
    return _error_response(422, "UNPROCESSABLE_ENTITY", message or "Invalid request body")


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception) -> JSONResponse:
    del request
    return _error_response(500, "INTERNAL_ERROR", str(exc))

