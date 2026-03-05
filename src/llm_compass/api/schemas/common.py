from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from llm_compass.common.types import DeploymentType

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class SpeedClass(str, Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


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

