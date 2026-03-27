"""Pydantic schemas for Node 4 (Scoring & Ranking) output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from llm_compass.common.types import SpeedClass


class BenchmarkResult(BaseModel):
    benchmark_id: int
    benchmark_name: str
    benchmark_variant: Optional[str] = None
    score: float
    metric_unit: str
    weight_used: float
    is_estimated: bool = False
    source_url: Optional[str] = None
    estimation_note: Optional[str] = None

    @model_validator(mode="after")
    def check_source_url(self) -> "BenchmarkResult":
        if not self.is_estimated and not self.source_url:
            raise ValueError("source_url is required for non-estimated benchmark results")
        return self


class PerformanceCI(BaseModel):
    """Confidence interval for performance_index reflecting uncertainty from missing benchmark data.

    Models with full benchmark coverage will have low == mid == high (point estimate).
    For models missing some benchmarks, the interval widens: low assumes 0.25 and high
    assumes 0.75 for each missing benchmark's normalized score. Use mid (0.5 assumption)
    for sorting and blended score computation.
    """

    low: float = Field(ge=0.0, le=1.0, description="Pessimistic: missing benchmarks scored 0.25")
    mid: float = Field(ge=0.0, le=1.0, description="Neutral midpoint, used for sorting (0.50 for missing)")
    high: float = Field(ge=0.0, le=1.0, description="Optimistic: missing benchmarks scored 0.75")


class RankMetrics(BaseModel):
    performance_index: PerformanceCI
    blended_cost_index: float
    blended_score: float


class RankedModel(BaseModel):
    model_id: int
    name_normalized: str
    provider: str
    speed_class: Optional[SpeedClass] = None
    speed_tps: Optional[int] = None
    cost_null_fraction: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of weighted cost data that was missing (0=fully known, 1=all null)",
    )
    rank_metrics: RankMetrics
    benchmark_results: List[BenchmarkResult]
    reason_for_ranking: str


class RankedLists(BaseModel):
    top_performance: List[RankedModel] = []
    balanced: List[RankedModel] = []
    budget: List[RankedModel] = []
    metadata: Dict[str, Any] = {}
