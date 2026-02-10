"""
Defines the LangGraph state.
Req 2.3: Carries UI inputs and Agent intermediate steps.
"""

from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator


class AgentState(TypedDict):
    user_query: str
    # UI Constraints (Req 3.2: Context, Modality, Deployment, etc.)
    constraints: Dict[str, Any]

    # Internal Logic State
    clarification_needed: bool
    clarification_question: Optional[str]
    predicted_io_ratio: Dict[str, float]  # Req 2.3 Node 2
    search_queries: List[str]

    # Results
    ranked_results: Dict[str, List[Any]]  # Top, Balanced, Budget lists
    final_response: Optional[Dict]  # The JSON schema for UI

    # Traceability (Req 3.3.A)
    logs: Annotated[List[str], operator.add]
