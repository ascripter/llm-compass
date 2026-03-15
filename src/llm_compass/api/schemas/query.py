from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ...common.schemas import Constraints
from .common import ErrorDetail
from llm_compass.agentic_core.schemas.ranking import RankedLists
from llm_compass.agentic_core.schemas.synthesis import SynthesisOutput


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
    constraints: Constraints = Constraints()


class TraceEvent(BaseModel):
    stage: str
    message: str
    data: Dict[str, Any] = {}


class StreamEvent(BaseModel):
    """Single NDJSON line emitted by the streaming query endpoint."""

    event: Literal["node_complete", "error", "complete"]
    node: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    logs: Optional[List[str]] = None


class QueryResponse(BaseModel):
    """Matches the AgentResponse JSON schema from Req 2.1.B."""

    session_id: str
    user_query: str
    applied_constraints: Dict[str, Any]
    status: Literal["ok", "needs_clarification", "error"]
    clarification_question: Optional[str] = None
    traceability: Dict[str, List[TraceEvent]] = {"events": []}
    ranked_data: Optional[RankedLists] = None
    ui_components: Optional[SynthesisOutput] = None
    debug_summary: Optional[str] = None
    errors: List[ErrorDetail] = []
