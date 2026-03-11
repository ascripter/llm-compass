from typing import List

from pydantic import BaseModel, Field


class Pagination(BaseModel):
    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)


class PaginatedMeta(BaseModel):
    total: int
    page: int
    per_page: int
    pages: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class APIError(BaseModel):
    errors: List[ErrorDetail]

