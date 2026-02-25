"""
Defines the database schema using SQLAlchemy and pgvector.
Maps directly to Product Requirements Section 1.2 (A, B, C, D).
"""

from typing import List, Optional, Literal
from datetime import datetime
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    JSON,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column as col, relationship

from pydantic import BaseModel, ValidationError, field_validator


Modality = Literal["text", "image", "audio", "video"]  # Extendable for future modalities
SpeedClass = Literal["fast", "medium", "slow"]  # For categorizing model inference speed


def _comma_separated_list_validator(v: str, allowed: Optional[tuple[str]] = None) -> list[str]:
    """Pydantic validator to convert a comma-separated string into a list of strings."""
    if isinstance(v, str):
        out = [item.strip() for item in v.split(",") if item.strip()]
    elif isinstance(v, list):
        out = v
    else:
        raise ValidationError(f"Value must be a comma-separated list of strings: {v}")
    for value in out:
        if allowed and value not in allowed:
            raise ValidationError(f"Value '{value}' is not in the allowed list: {allowed}")
    return out


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class BenchmarkDictionary(Base):
    """
    Stores semantic definitions of benchmarks.
    Req 1.2.B: Supports 'Smart Lookup' via vector embeddings.
    """

    __tablename__ = "benchmark_dictionary"

    id: Mapped[int] = col(Integer, primary_key=True)  # FAISS index, required to be set a priori
    name_normalized: Mapped[str] = col(String, index=True, nullable=False)
    variant: Mapped[str] = col(String, default=None, nullable=True)
    # the (non-embedded) description string (English)
    description: Mapped[str] = col(String, nullable=False)
    categories: Mapped[List[str]] = col(JSON, default=[], nullable=False)

    # Relationship to BenchmarkScore
    benchmark_scores: Mapped[List["BenchmarkScore"]] = relationship(back_populates="benchmark")

    # Constraints
    __table_args__ = (
        UniqueConstraint("name_normalized", "variant", name="_name_variant_unique"),
    )


class BenchmarkDictionarySchema(BaseModel):
    """Pydantic schema for validating BenchmarkDictionary entries."""

    id: int
    name_normalized: str
    variant: Optional[str] = None
    description: str
    categories: List[str]

    @classmethod
    @field_validator("categories", mode="before")
    def validate_categories(cls, v: str) -> list[str]:
        return _comma_separated_list_validator(v)


class LLMMetadata(Base):
    """
    Stores static attributes for filtering and tradeoff analysis.
    Req 1.2.D: Tracks parameters, modality, and costs.
    """

    __tablename__ = "llm_metadata"

    id: Mapped[Optional[int]] = col(Integer, primary_key=True)
    name_normalized: Mapped[str] = col(String, index=True, nullable=False)
    provider: Mapped[str] = col(String, nullable=False)  # e.g. "OpenAI", "Anthropic", "Meta"
    parameter_count: Mapped[Optional[int]] = col(Integer, nullable=True)
    architecture: Mapped[Optional[str]] = col(String, nullable=True)  # e.g. "transformer"
    quantization: Mapped[Optional[str]] = col(String, nullable=True)  # e.g. "fp8", "q4_k_m"
    distillation_source: Mapped[Optional[str]] = col(String, nullable=True)  # if distilled
    modality_input: Mapped[List[Modality]] = col(JSON, default=[], nullable=False)
    modality_output: Mapped[List[Modality]] = col(JSON, default=[], nullable=False)
    context_window: Mapped[int] = col(Integer, nullable=False)
    cost_input_1m: Mapped[float] = col(Float, nullable=False)
    cost_output_1m: Mapped[float] = col(Float, nullable=False)
    speed_class: Mapped[SpeedClass] = col(String, nullable=False)
    speed_tps: Mapped[Optional[float]] = col(Float, nullable=True)
    is_open_weights: Mapped[bool] = col(Boolean, nullable=False)
    is_reasoning_model: Mapped[bool] = col(Boolean, nullable=False)
    has_tool_calling: Mapped[bool] = col(Boolean, nullable=False)
    is_outdated: Mapped[bool] = col(Boolean, default=False, nullable=False)
    superseded_by_model_id: Mapped[Optional[int]] = col(
        Integer, ForeignKey("llm_metadata.id"), nullable=True
    )
    # Relationship to BenchmarkScore
    benchmark_scores: Mapped[List["BenchmarkScore"]] = relationship(back_populates="model")

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "name_normalized",
            "provider",
            "quantization",
            name="_name_provider_quantization_unique",
        ),
    )


class LLMMetadataSchema(BaseModel):
    """Pydantic schema for validating LLMMetadata entries."""

    id: Optional[int] = None
    name_normalized: str
    provider: str
    parameter_count: Optional[int] = None
    architecture: Optional[str] = None
    quantization: Optional[str] = None
    distillation_source: Optional[str] = None
    modality_input: List[Modality] = []
    modality_output: List[Modality] = []
    context_window: int
    cost_input_1m: float
    cost_output_1m: float
    speed_class: SpeedClass
    speed_tps: Optional[float] = None
    is_open_weights: bool
    is_reasoning_model: bool
    has_tool_calling: bool
    is_outdated: bool = False
    superseded_by_model_id: Optional[int] = None

    @classmethod
    @field_validator("modality_input", "modality_output", mode="before")
    def validate_modalities(cls, v):
        return _comma_separated_list_validator(v, Modality.__args__)

    @classmethod
    @field_validator("is_open_weights", "is_reasoning_model", "has_tool_calling", mode="before")
    def validate_bool_fields(cls, v):
        """Convert CSV boolean strings (TRUE/FALSE, 1/0, yes/no) to Python bool."""
        if isinstance(v, str):
            v = v.strip().upper()
            if v in ("TRUE", "1", "YES"):
                return True
            elif v in ("FALSE", "0", "NO"):
                return False
        return bool(v)


class BenchmarkScore(Base):
    """
    The core repository of raw performance data.
    Req 1.2.C: Links Models to Benchmarks with scores.
    """

    __tablename__ = "benchmark_scores"

    id: Mapped[Optional[int]] = col(Integer, primary_key=True)
    model_id: Mapped[int] = col(Integer, ForeignKey("llm_metadata.id"), nullable=False)
    benchmark_id: Mapped[int] = col(
        Integer, ForeignKey("benchmark_dictionary.id"), nullable=False
    )
    score_value: Mapped[float] = col(Float, nullable=False)
    metric_unit: Mapped[str] = col(String, nullable=False)  # e.g. "%", "elo", "pass@1"
    source_name: Mapped[str] = col(String, nullable=False)  # e.g. "vals.ai"
    source_url: Mapped[str] = col(String, nullable=False)  # full url to source
    date_published: Mapped[Optional[datetime]] = col(DateTime, nullable=True)
    date_ingested: Mapped[datetime] = col(DateTime, default=datetime.utcnow, nullable=False)
    original_model_name: Mapped[str] = col(String, nullable=False)  # For audit (Req 1.3.A)
    original_benchmark_name: Mapped[str] = col(String, nullable=False)  # For audit (Req 1.3.A)
    original_benchmark_variant: Mapped[str] = col(String, nullable=True)  # For audit (Req 1.3.A)

    # Relationships
    model: Mapped["LLMMetadata"] = relationship(back_populates="benchmark_scores")
    benchmark: Mapped["BenchmarkDictionary"] = relationship(back_populates="benchmark_scores")


class BenchmarkScoreSchema(BaseModel):
    """Pydantic schema for validating BenchmarkScore entries."""

    id: Optional[int] = None
    model_id: Optional[int] = None  # Optional during validation; to be filled after FK resolution
    benchmark_id: Optional[int] = None  # Optional during validation; t.b.f. after FK resolution
    score_value: float
    metric_unit: str
    source_name: str
    source_url: str
    date_published: Optional[datetime] = None
    date_ingested: Optional[datetime] = None
    original_model_name: str
    original_benchmark_name: str
    original_benchmark_variant: Optional[str] = None

    @classmethod
    @field_validator("date_published", mode="before")
    def validate_date_published(cls, v):
        """Convert CSV date strings to datetime objects."""
        if v is None or v == "":
            return None
        if isinstance(v, datetime):
            return v
        # Try common date formats
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(v, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date: {v}")
