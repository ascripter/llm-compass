"""
Defines the database schema using SQLModel and pgvector.
Maps directly to Product Requirements Section 1.2 (A, B, C, D).
"""

from typing import List, Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column


class BenchmarkDictionary(SQLModel, table=True):
    """
    Stores semantic definitions of benchmarks.
    Req 1.2.B: Supports 'Smart Lookup' via vector embeddings.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name_normalized: str = Field(index=True)
    variant: str = Field(default="default")
    description: str
    # Embedding for semantic search (e.g., OpenAI text-embedding-3-small is 1536 dims)
    embedding: List[float] = Field(sa_column=Column(Vector(1536)))


class LLMMetadata(SQLModel, table=True):
    """
    Stores static attributes for filtering and tradeoff analysis.
    Req 1.2.D: Tracks parameters, modality, and costs.
    """

    model_id: Optional[int] = Field(default=None, primary_key=True)
    name_normalized: str = Field(unique=True)
    provider: str
    is_open_weights: bool = Field(default=False)
    # Stored as JSON strings or using a JSON column type in real impl
    modality_input: str  # e.g. "text,image"
    cost_input_1m: float
    cost_output_1m: float
    # ... other fields from reqs (speed_class, context_window, etc.)


class BenchmarkScore(SQLModel, table=True):
    """
    The core repository of raw performance data.
    Req 1.2.C: Links Models to Benchmarks with scores.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: int = Field(foreign_key="llmmetadata.model_id")
    benchmark_id: int = Field(foreign_key="benchmarkdictionary.id")
    score_value: float
    original_model_name: str  # For audit (Req 1.3.A)
    source_url: Optional[str] = None
