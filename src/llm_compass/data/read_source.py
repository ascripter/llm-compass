"""Read csv / online table sources of data to be ingested into the database
(Req 1.1.B). This includes parsing and basic preprocessing, but not
name normalization or FK resolution which are part of the ingestion step (Req 1.3.A).

NOTE: All methods for manual MVP ingestion.
"""

import csv
import io
from datetime import datetime
from urllib.parse import urlparse
from typing import Any

import httpx
from pydantic._internal._model_construction import ModelMetaclass

from .models import (
    BenchmarkDictionarySchema,
    BenchmarkScoreSchema,
    LLMMetadataSchema,
)


def _get_google_sheet_url(gid: int | str) -> str:
    """Return the exportable csv url for a given gid (sheet number)
    in the google sheet benchmark dictionary source.
    """
    root = "https://docs.google.com/spreadsheets/d/"
    sheet_id = "1a_4LmfIuKhhHzzZ3-xcO6jStHCIxbhgF60rRxoPfiMU"
    url = f"{root}{sheet_id}/export?format=csv&gid={str(gid)}"
    return url


def _fetch_url_as_csv_reader(url: str, skip_rows: int = 0) -> csv.DictReader:
    """Fetch CSV content from URL and parse with csv.DictReader.

    Returns:
        Iterator of dictionaries, one per row, with keys matching CSV headers.
    """
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    csv_content = io.StringIO(response.text)
    for _ in range(skip_rows):
        next(csv_content)  # Skip rows if needed
    reader = csv.DictReader(csv_content)

    return reader


def _validate_rows(
    reader: csv.DictReader | list[dict[str, Any]], validation_class: ModelMetaclass
) -> list[dict[str, Any]]:
    """Validate each row using the provided Pydantic model."""
    validated_rows = []
    for row in reader:
        for k, v in row.items():
            if v == "":
                row[k] = None  # set empty strings to None for optional fields
        validated = validation_class(**row)
        validated_rows.append(validated.model_dump())
    return validated_rows


def benchmark_dictionary_from_googlesheet() -> list[dict[str, Any]]:
    """Get benchmark dictionary table from online source.

    Returns:
        List of validated dictionaries ready for database insertion.
    """
    url = _get_google_sheet_url(gid=0)
    reader = _fetch_url_as_csv_reader(url)
    rows = _validate_rows(reader, BenchmarkDictionarySchema)

    # Sort by name_normalized and variant
    rows.sort(key=lambda x: (x["name_normalized"], x["variant"] or ""))

    # Add designated FAISS  and id for DB insertion
    for idx, record in enumerate(rows):
        record["id"] = idx

    return rows


def llm_metadata_from_googlesheet() -> list[dict[str, Any]]:
    """Get LLM metadata table from online source.

    Returns:
        List of validated dictionaries ready for database insertion.
    """
    url = _get_google_sheet_url(gid=1019571654)
    reader = _fetch_url_as_csv_reader(url, skip_rows=1)  # Skip first row (header)
    rows = _validate_rows(reader, LLMMetadataSchema)

    # Sort by provider and name_normalized
    rows.sort(key=lambda x: (x["provider"], x["name_normalized"]))

    return rows


def benchmark_scores_from_googlesheet() -> list[dict[str, Any]]:
    """Get benchmark scores table from online source.

    Returns partly preprocessed list of dictionaries with the following steps missing:
    - name normalization
    - FK resolution for model_id and benchmark_id

    Returns:
        List of dictionaries with source_name and date_ingested derived.
    """
    url = _get_google_sheet_url(gid=984276769)
    reader = _fetch_url_as_csv_reader(url)
    rows = []
    for row in reader:
        # Derive source_name from source_url
        row["source_name"] = (
            urlparse(row["source_url"]).netloc.lower() if row["source_url"] else None
        )
        row["date_ingested"] = datetime.utcnow()

        # Rename original columns for audit purposes
        row["original_model_name"] = row.pop("model")
        row["original_benchmark_name"] = row.pop("benchmark_name")
        row["original_benchmark_variant"] = row.pop("benchmark_variant", None)
        rows.append(row)

    rows = _validate_rows(rows, BenchmarkScoreSchema)

    # Returned list has missing name normalization and columns that are required for DB insertion:
    # - model_id (int, FK to LLMMetadata)
    # - benchmark_id (int, FK to BenchmarkDictionary)
    return rows
