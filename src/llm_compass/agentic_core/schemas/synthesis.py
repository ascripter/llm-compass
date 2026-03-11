"""Pydantic schemas for Node 5 (Synthesis) output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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

    task_summary: str
    executive_summary: str
    recommendation_reasons: Dict[str, str]  # keys: top_performance, balanced, budget
    offset_calibration_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Full synthesis output stored in AgentState
# ---------------------------------------------------------------------------

class SynthesisOutput(BaseModel):
    """Complete synthesis output. Combines LLM text with deterministic components."""

    summary_markdown: str
    comparison_table: Optional[ComparisonTable] = None
    recommendation_cards: List[RecommendationCard] = []
    citations: List[Citation] = []
    warnings: List[Warning] = []
