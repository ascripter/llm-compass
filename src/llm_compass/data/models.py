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
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, col as col, relationship

from pgvector.sqlalchemy import Vector

EMBED_MODEL_NAME = "qwen/qwen3-embedding-8b"
EMBED_MODEL_VERSION = "openrouter"
EMBED_DIM = 4096  # Qwen3-Embedding-8B default benchmark dimension

Modality = Literal["text", "image", "audio", "video"]  # Extendable for future modalities
SpeedClass = Literal["fast", "medium", "slow"]  # For categorizing model inference speed


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class BenchmarkDictionary(Base):
    """
    Stores semantic definitions of benchmarks.
    Req 1.2.B: Supports 'Smart Lookup' via vector embeddings.
    """

    __tablename__ = "benchmark_dictionary"

    id: Mapped[Optional[int]] = col(Integer, primary_key=True)
    name_normalized: Mapped[str] = col(String, index=True, nullable=False)
    variant: Mapped[str] = col(String, default=None, nullable=True)
    # the (non-embedded) description string (English)
    description: Mapped[str] = col(String, nullable=False)
    # Embedding for semantic search
    description_embedding: Mapped[List[float]] = col(Vector(EMBED_DIM), nullable=False)
    # To track which embedding model was used
    embedding_model_name: Mapped[str] = col(String, default=EMBED_MODEL_NAME, nullable=False)
    # For future-proofing against model updates
    embedding_model_version: Mapped[str] = col(
        String, default=EMBED_MODEL_VERSION, nullable=False
    )
    embedding_timestamp: Mapped[datetime] = col(
        String, default=datetime.utcnow().isoformat(), nullable=False
    )
    categories: Mapped[List[str]] = col(ARRAY(String), default=[], nullable=False)

    # Relationship to BenchmarkScore
    benchmark_scores: Mapped[List["BenchmarkScore"]] = relationship(back_populates="benchmark")

    # Constraints
    __table_args__ = (
        UniqueConstraint("name_normalized", "variant", name="_name_variant_unique"),
    )


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
    modality_input: Mapped[List[Modality]] = col(ARRAY(String), default=[], nullable=False)
    modality_output: Mapped[List[Modality]] = col(ARRAY(String), default=[], nullable=False)
    context_window: Mapped[Optional[int]] = col(Integer, nullable=False)
    cost_input_1m: Mapped[float] = col(Float, nullable=False)
    cost_output_1m: Mapped[float] = col(Float, nullable=False)
    speed_class: Mapped[Optional[SpeedClass]] = col(String, nullable=False)
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
    score_value: Mapped[float] = col(Float)
    metric_unit: Mapped[str] = col(String, nullable=True)  # e.g. "%", "elo", "pass@1"
    source_name: Mapped[Optional[str]] = col(String, nullable=True)  # e.g. "Hugging Face"
    source_url: Mapped[Optional[str]] = col(String, nullable=True)
    date_published: Mapped[Optional[datetime]] = col(DateTime, nullable=True)
    date_ingested: Mapped[datetime] = col(DateTime, default=datetime.utcnow, nullable=False)
    original_model_name: Mapped[str] = col(String, nullable=False)  # For audit (Req 1.3.A)
    original_benchmark_name: Mapped[str] = col(String, nullable=False)  # For audit (Req 1.3.A)

    # Relationships
    model: Mapped["LLMMetadata"] = relationship(back_populates="benchmark_scores")
    benchmark: Mapped["BenchmarkDictionary"] = relationship(back_populates="benchmark_scores")
