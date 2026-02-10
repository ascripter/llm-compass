"""
Req 1.3.A: Write-Time Normalization.
Uses a cheap LLM to map raw strings to standardized entities.
"""


def normalize_entity_name(raw_name: str, entity_type: str) -> str:
    """
    Uses an LLM call to standardize names.
    e.g., "llama-2-7b-chat-hf" -> "Llama 2 7B Chat"
    """
    # implementation using OpenAI/Anthropic client
    pass
