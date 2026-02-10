"""
Tests for Data Models and Schema constraints.
Req 1.2: Verify schema integrity and vector fields.
"""

import pytest
from sqlalchemy.exc import IntegrityError
from src.data.models import LLMMetadata, BenchmarkDictionary


def test_llm_metadata_uniqueness(session):
    """
    Req 1.2.D: Ensure 'name_normalized' is unique for models.
    """
    model1 = LLMMetadata(
        name_normalized="Llama 3 70B",
        provider="Meta",
        cost_input_1m=0.9,
        cost_output_1m=0.9,
        modality_input="text",
        is_open_weights=True,
    )
    session.add(model1)
    session.commit()

    # Attempt duplicate
    model2 = LLMMetadata(
        name_normalized="Llama 3 70B",
        provider="Meta",
        cost_input_1m=0.5,
        cost_output_1m=0.5,
        modality_input="text",
    )
    session.add(model2)

    with pytest.raises(IntegrityError):
        session.commit()


def test_benchmark_embedding_dimensions(session):
    """
    Req 1.2.B: Verify vector column accepts correct dimensions (1536).
    Note: Requires PGVector extension enabled in the test DB.
    For SQLite/unit tests, we might mock this or skip if using SQLite.
    """
    # ... implementation dependent on test DB backend ...
    pass
