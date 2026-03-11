"""Pydantic schemas for Node 5 (Synthesis) output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reusable UI building-blocks (moved from api/schemas/query.py)
# ---------------------------------------------------------------------------

class ComparisonTable(BaseModel):
    title: str
    columns: List[str]
    rows: List[List[Any]]


class RecommendationCard(BaseModel):
    category: str
    model_name: str
    reason: str


class Citation(BaseModel):
    id: str
    label: str
    url: str


class Warning(BaseModel):
    code: str
    message: str


# ---------------------------------------------------------------------------
# LLM structured output (natural-language parts only)
# ---------------------------------------------------------------------------

class SynthesisLLMOutput(BaseModel):
    """LLM structured output for Node 5.

    The LLM generates ONLY natural-language parts; deterministic components
    are built by the node function.
    """

    task_summary: str = Field(
        description="1-2 sentence rephrasing of the user's intended task."
    )
    executive_summary: str = Field(
        description=(
            "3-5 sentence markdown highlighting top performance winner, budget winner, "
            "key trade-offs, and most relevant benchmarks."
        )
    )
    recommendation_reasons: Dict[str, str] = Field(
        description=(
            "One reason per category explaining why the model wins, referencing benchmarks. "
            "Keys: 'top_performance', 'balanced', 'budget'. "
            "Only include keys for categories that have at least one model."
        )
    )
    offset_calibration_note: Optional[str] = Field(
        None,
        description=(
            "If any is_estimated=true scores exist, note which models/benchmarks "
            "were estimated and why. Null otherwise."
        ),
    )


# ---------------------------------------------------------------------------
# Full synthesis output stored in AgentState
# ---------------------------------------------------------------------------

class SynthesisOutput(BaseModel):
    """Complete synthesis output. Combines LLM text with deterministic components."""

    llm_output: Optional[SynthesisLLMOutput] = None
    summary_markdown: str
    comparison_table: Optional[ComparisonTable] = None
    recommendation_cards: List[RecommendationCard] = []
    citations: List[Citation] = []
    warnings: List[Warning] = []
