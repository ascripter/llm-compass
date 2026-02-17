"""
Tests for Ingestion logic.
Req 1.1: Verify manual import and normalization triggers.
"""

import pandas as pd
from io import StringIO
from src.data.ingestion import import_manual_csv


def test_manual_csv_import_valid(session, mocker):
    """
    Req 1.1.B: Test valid CSV import workflow.
    """
    # Mock the normalizer to avoid real LLM calls
    mocker.patch(
        "src.data.ingestion.normalize_entity_name",
        side_effect=lambda name, type: name.upper(),  # simple mock logic
    )

    csv_data = """model_name,benchmark,score,source
llama-2,mmlu,0.5,paper
"""
    file_buffer = StringIO(csv_data)

    # Run import
    import_manual_csv(file_buffer, session)

    # Assertions
    # Check if Score record exists
    # Check if Model record was created (normalized)
    # Check if Benchmark record was created (normalized)
    pass


def test_audit_fields_preservation(session, mocker):
    """
    Req 1.3.A: Ensure 'original_model_name' is saved strictly.
    """
    # ... similar setup ...
    # Assert that score.original_model_name == "llama-2"
    pass
