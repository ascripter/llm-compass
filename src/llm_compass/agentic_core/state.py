"""
Defines the LangGraph state.
Req 2.3: Carries UI inputs and Agent intermediate steps.
"""

from typing import TypedDict, List, Dict, Any, Annotated, Optional, Literal
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState  # implies `messages` attribute
from langgraph.graph.message import add_messages

from .schemas import IntentExtraction, TokenRatioEstimation
from ..common.schemas import Constraints
from ..common.types import Modality


class AgentState(MessagesState):
    user_query: str
    constraints: Constraints  # UI Constraints (Req 3.2: Context, Modality, Deployment, etc.)

    # From intent validation node (Req 2.3 Node 1)
    clarification_count: int  # Tracks cycles; max 3 before terminal error
    clarification_limit_exceeded: bool
    intent_extraction: Optional[IntentExtraction]
    token_ratio_estimation: Optional[TokenRatioEstimation]

    # From Query Refinement (Req 2.3 Node 2)
    search_queries: List[str]

    # rest t.b.d.
    # Results
    ranked_results: Dict[str, List[Any]]  # Top, Balanced, Budget lists
    final_response: Optional[Dict]  # The JSON schema for UI

    # Traceability (Req 3.3.A)
    logs: Annotated[List[str], add_messages]
