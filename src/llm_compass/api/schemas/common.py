from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class SpeedClass(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    SLOW = "slow"


class DeploymentType(str, Enum):
    ANY = "any"
    CLOUD = "cloud"
    LOCAL = "local"


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

