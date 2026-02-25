"""
Req 1.3.A: Write-Time Normalization.
Uses a cheap LLM to map raw strings to standardized entities.
"""


class Normalizer:
    """
    Centralizes normalization logic for model and benchmark names.
    This is a critical step before FK resolution and database insertion.
    """

    def __init__(self, settings):
        self.settings = settings
        # Initialize LLM client here (e.g. Openrouter) with a small model for normalization tasks.

    def normalize_model_name(self, raw_name: str) -> str:
        """
        Uses an LLM call to standardize names.
        e.g., "llama-2-7b-chat-hf" -> "Llama 2 7B Chat"
        """
        # implementation using Openrouter client with a small model (e.g. gpt-3.5-turbo) to do the normalization.
        return raw_name.strip().title()  # Placeholder: simple title case for MVP

    def normalize_benchmark_name(self, raw_name: str) -> str:
        """
        Uses an LLM call to standardize names.
        e.g., "mmlu-5shot" -> "MMLU 5 Shot"
        """
        # implementation using Openrouter client with a small model (e.g. gpt-3.5-turbo) to do the normalization.
        return raw_name  # Placeholder: no modification for MVP
