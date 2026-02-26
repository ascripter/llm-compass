"""
Req 1.3.A: Write-Time Normalization.
Uses a cheap LLM to map raw strings to standardized entities.
"""

from collections.abc import Sequence
from typing import Optional

from llm_compass.config import Settings


class Normalizer:
    """
    Centralizes normalization logic for model and benchmark names.
    This is a critical step before FK resolution and database insertion.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # Initialize LLM client here (e.g. Openrouter) with a small model for normalization tasks.

    def normalize_model_names(self, raw_names: list[str]) -> list[str]:
        """
        Uses an LLM call to standardize names.
        e.g., "llama-2-7b-chat-hf" -> "Llama 2 7B Chat"
        """
        # implementation using Openrouter client with a small model (e.g. gpt-3.5-turbo) to do the normalization.
        return [n.strip() for n in raw_names]  # Placeholder

    def normalize_benchmark_names(
        self, raw_names: list[str], raw_variants: Sequence[str | None] | None = None
    ) -> list[tuple[str, Optional[str]]]:
        """
        Uses an LLM call to standardize names and split into base name + benchmark variant if applicable.
        e.g., "mmlu-5shot" -> ("MMLU", "5 Shot")
        """
        # implementation using Openrouter client with a small model (e.g. gpt-3.5-turbo) to do the normalization.
        if raw_variants is None:
            raw_variants = [None] * len(raw_names)
        assert len(raw_names) == len(raw_variants), "raw_names and raw_variants must be same len"
        # Placeholder: no modification for MVP
        return [(n.strip(), v) for n, v in zip(raw_names, raw_variants)]
