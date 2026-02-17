"""
Handles data ingestion strategies.
Req 1.1: Supports Manual Import (CSV) and Scheduled Aggregation.
"""

import pandas as pd
from .database import get_session
from .normalizer import normalize_entity_name


def import_manual_csv(file_buffer):
    """
    Req 1.1.B: Parses uploaded CSV, normalizes names, and saves to DB.

    Steps:
    1. Read CSV.
    2. For each row:
       - Call normalize_entity_name() for model/benchmark.
       - Create/Get LLMMetadata and BenchmarkDictionary records.
       - Insert BenchmarkScore.
    """
    pass


def run_scheduled_aggregation():
    """
    Req 1.1.A: Fetches from configured sources (mocked for MVP).
    """
    pass
