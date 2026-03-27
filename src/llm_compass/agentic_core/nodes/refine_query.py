"""
Req 2.3 Node 2 (a): Query Refiner (LLM)

This node runs after intent validation succeeds (parallel to Token Ratio Estimation).
It performs search query generation (3 to 5 benchmark-discovery queries).
"""

from typing import Any
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm_compass.config import Settings
from llm_compass.common.schemas import Constraints
from ..schemas import IntentExtraction, QueryExpansion
from ..state import AgentState

logger = logging.getLogger(__name__)

QUERY_EXPANSION_SYSTEM_PROMPT = """You are a benchmark-discovery query generation assistant.

Given the validated user task and active UI constraints:
1. Generate 3-5 distinct semantic search queries for benchmark retrieval.
2. Queries should focus on measurable evaluation needs (reasoning, coding, summarization, multimodal understanding, etc.).
3. Keep queries concise and specific enough for vector retrieval.
4. Follow the output schema strictly.
"""


def _ensure_query_count(search_queries: list[str], user_query: str) -> tuple[list[str], bool]:
    unique: list[str] = []
    seen: set[str] = set()
    for query in search_queries:
        cleaned = query.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            unique.append(cleaned)
            seen.add(key)

    seed = user_query.strip() or "LLM benchmark recommendation"
    fallback_candidates = [
        f"{seed} benchmark",
        f"{seed} evaluation",
        f"{seed} performance benchmark",
        "llm benchmark leaderboard",
        "model evaluation benchmark suite",
    ]
    used_fallback = False
    for candidate in fallback_candidates:
        if len(unique) >= 3:
            break
        key = candidate.lower()
        if key in seen:
            continue
        unique.append(candidate)
        seen.add(key)
        used_fallback = True

    return unique[:5], used_fallback


def query_refiner_node(state: AgentState, *, settings: Settings) -> dict[str, Any]:
    """
    Refines a validated query (Req 2.3 Node 2).

    Returns:
        dict[str, Any]:
            - search_queries: list[str]
            - logs: list[str]
    """
    # patch: use 4o-mini since gpt-oss-120b doesn't adhere to schema consistently
    llm = settings.make_llm("openai/gpt-5-mini", temperature=0.7)
    query_expander = llm.with_structured_output(QueryExpansion)

    constraints_raw = state.get("constraints")
    constraints = (
        Constraints(**constraints_raw) if isinstance(constraints_raw, dict) else constraints_raw
    )
    if constraints is None:
        constraints = Constraints(min_context_window=0)

    intent_raw = state.get("intent_extraction")
    intent = IntentExtraction(**intent_raw) if isinstance(intent_raw, dict) else intent_raw

    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=str(state.get("user_query", "")).strip())]

    context_lines = [
        f"User query: {state.get('user_query', '')}",
        f"UI constraints: {constraints.model_dump_json()}",
    ]
    if intent is not None:
        context_lines.append(
            "Validated intent modalities:"
            f" input={intent.intended_input_modalities}, output={intent.intended_output_modalities}"
        )
    context = "\n".join(context_lines)

    query_messages = (
        [SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT)]
        + messages
        + [HumanMessage(content=context)]
    )
    query_expansion: QueryExpansion = query_expander.invoke(query_messages)  # type: ignore[assignment]

    search_queries, used_fallback = _ensure_query_count(
        query_expansion.search_queries,
        str(state.get("user_query", "")),
    )

    logger.debug(
        "query_refiner_node EXIT | search_queries=%s | used_fallback=%s",
        search_queries,
        used_fallback,
    )

    logs = [
        f"{len(search_queries)} queries",
    ]
    if used_fallback:
        logs.append("Original list had < 3 unique entries => fallback queries appended.")

    return {
        "search_queries": search_queries,
        "logs": logs,
    }
