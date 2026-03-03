from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .common import DeploymentType, ErrorDetail, Modality, SpeedClass


class Constraints(BaseModel):
    """Maps 1:1 to UI constraint panel (Req 3.2)."""

    min_context_window: int = Field(0, ge=0)
    modality_input: List[Modality] = ["text"]
    modality_output: List[Modality] = ["text"]
    deployment: DeploymentType = DeploymentType.ANY
    require_reasoning: bool = False
    require_tool_calling: bool = False
    min_speed_class: Optional[SpeedClass] = None


class QueryRequest(BaseModel):
    """Primary request body for the /query endpoint."""

    user_query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        examples=["I need a model for RAG on legal documents"],
    )
    constraints: Constraints = Constraints()
    session_id: Optional[str] = None


class ClarifyRequest(BaseModel):
    """Follow-up message when the agent asks for clarification."""

    user_reply: str = Field(..., min_length=1, max_length=2000)


class TraceEvent(BaseModel):
    stage: str
    message: str
    data: Dict[str, Any] = {}


class BenchmarkResult(BaseModel):
    benchmark_id: str
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
    model_id: str
    name_normalized: str
    provider: str
    speed_class: Optional[SpeedClass] = None
    speed_tps: Optional[float] = None
    rank_metrics: RankMetrics
    benchmark_results: List[BenchmarkResult]
    reason_for_ranking: str


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


class UIComponents(BaseModel):
    summary_markdown: str
    comparison_table: Optional[ComparisonTable] = None
    recommendation_cards: List[RecommendationCard] = []
    citations: List[Citation] = []
    warnings: List[Warning] = []


class RankedLists(BaseModel):
    top_performance: List[RankedModel]
    balanced: List[RankedModel]
    budget: List[RankedModel]
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Matches the AgentResponse JSON schema from Req 2.1.B."""

    session_id: str
    user_query: str
    applied_constraints: Dict[str, Any]
    status: Literal["ok", "needs_clarification", "error"]
    clarification_question: Optional[str] = None
    traceability: Dict[str, List[TraceEvent]] = {"events": []}
    ranked_data: Optional[RankedLists] = None
    ui_components: Optional[UIComponents] = None
    errors: List[ErrorDetail] = []

