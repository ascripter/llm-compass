"""
Structured output schema for Req 2.3
- Node 3 (b): Weighting pre-filtered benchmarks by LLM
"""

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator

RelevanceClass = Literal[
    "perfect_match",
    "strong_match",
    "partial_match",
    "weak_match",
    "no_match",
]

CLASS_TO_WEIGHT = {
    "perfect_match": 1.00,
    "strong_match": 0.75,
    "partial_match": 0.50,
    "weak_match": 0.25,
    "no_match": 0.00,
}


class BenchmarkJudgment(BaseModel):
    benchmark_id: int = Field(description="Unique benchmark identifier from your DB.")
    relevance_class: RelevanceClass = Field(
        description=(
            "perfect_match = direct benchmark fit for the task; "
            "strong_match = clearly relevant but not exact; "
            "partial_match = relevant sub-capability only; "
            "weak_match = only loose or indirect relation; "
            "no_match = not meaningfully relevant."
        )
    )
    short_rationale: str = Field(
        description="1-2 short sentences explaining the main reason for the assigned class."
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def relevance_weight(self) -> float:
        return CLASS_TO_WEIGHT[self.relevance_class]


class BenchmarkJudgments(BaseModel):
    judgments: list[BenchmarkJudgment] = Field(
        description="Judgment for each candidate benchmark."
    )

    model_config = ConfigDict(extra="forbid")
