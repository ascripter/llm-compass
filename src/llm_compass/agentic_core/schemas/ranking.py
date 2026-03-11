"""Pydantic schemas for Node 4 (Scoring & Ranking) output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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


class RankMetrics(BaseModel):
    performance_index: float
    blended_cost_index: float
    blended_score: float


class RankedModel(BaseModel):
    model_id: int
    name_normalized: str
    provider: str
    speed_class: Optional[SpeedClass] = None
    speed_tps: Optional[float] = None
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
