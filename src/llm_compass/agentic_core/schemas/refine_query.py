"""
Structured output schemas for Req 2.3 Node 2 (Query Refiner).
- Node 2 (a): Refine Query to get 3-5 search queries.
"""

from pydantic import BaseModel, Field, field_validator


class QueryExpansion(BaseModel):
    """LLM output schema for generating semantic benchmark search queries."""

    reasoning: str = Field(
        description=(
            "Briefly explain how the queries were derived from the user task and constraints."
        )
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description=(
            "A list of 3 to 5 distinct semantic search queries for benchmark discovery."
        ),
    )

    @field_validator("search_queries", mode="before")
    @classmethod
    def normalize_queries(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for entry in value:
            query = str(entry).strip()
            key = query.lower()
            if query and key not in seen:
                normalized.append(query)
                seen.add(key)
        return normalized
