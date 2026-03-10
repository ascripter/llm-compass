"""
Req 1.1: Supports Manual Import (CSV) and Scheduled Aggregation.
Handles data ingestion strategies for SQLite *and* FAISS
"""

from typing import Any

from sqlalchemy import insert, delete
from .database import Database
from .embedding import Embedding
from .models import BenchmarkDictionary, LLMMetadata, BenchmarkScore, ModelNormalized, ModelNormalizedSchema
from .normalizer import Normalizer, normalize


def ingest_benchmark_dictionary(
    *,
    records: list[dict[str, Any]],
    database: Database,
    normalizer: Normalizer,
    embedding: Embedding,
    update: bool = True,
):
    """Req 1.1.B: Parses benchmark dictionary from source, normalizes names,
    saves to DB + FAISS index.
    If `update = True` rows will be appended to existing table,
    otherwise all rows will be deleted and recreated (dev usage).
    """

    # Normalization step: (noop in MVP)
    bench_names = [r["name_normalized"] for r in records]
    bench_variants = [r["variant"] for r in records]
    bench_norm = normalizer.normalize_benchmark_names(bench_names, bench_variants)
    for row, (bench_name_norm, bench_variant_norm) in zip(records, bench_norm):
        row["name_normalized"] = bench_name_norm
        row["variant"] = bench_variant_norm

    # Write to DB
    with database.SessionLocal() as session:
        if update:
            n_updated = 0
            for row in records:
                exists = (
                    session.query(BenchmarkDictionary)
                    .filter_by(name_normalized=row["name_normalized"], variant=row["variant"])
                    .first()
                )
                if not exists:
                    print(f"inserting benchmark {row['name_normalized']} {row['variant']}")
                    n_updated += 1
                    session.execute(insert(BenchmarkDictionary).values(**row))
            session.commit()
            all_records = session.query(BenchmarkDictionary).all()
            all_records = [dict(_.__dict__) for _ in all_records]  # Convert to list of dicts
        else:
            # bulk recreate
            session.execute(delete(BenchmarkDictionary))
            n_updated = len(records)
            session.execute(insert(BenchmarkDictionary).values(records))
            session.commit()
            all_records = records

    # (re)write FAISS index
    # tbd: allow update instead of full rewrite, but for MVP we can live with this
    if n_updated > 0:
        embedding.generate_index(all_records, text_key="name_normalized", id_key="id")


def ingest_llm_metadata(
    *,
    records: list[dict[str, Any]],
    database: Database,
    normalizer: Normalizer,
    update: bool = True,
):
    """Req 1.1.B: Parses LLM metadata from source, normalizes names, saves to DB.
    If `update = True` rows will be appended to existing table,
    otherwise all rows will be deleted and recreated (dev usage).
    """
    # Normalization step: Should return the same name since it's already normalized
    # in the source sheet, but we call it for consistency and future-proofing.
    model_names = [r["name_normalized"] for r in records]
    model_norm = normalizer.normalize_model_names(model_names)
    for row, model_name_norm in zip(records, model_norm):
        row["name_normalized"] = model_name_norm

    # Write to DB
    with database.SessionLocal() as session:
        if update:
            for row in records:
                exists = (
                    session.query(LLMMetadata)
                    .filter_by(name_normalized=row["name_normalized"])
                    .first()
                )
                if not exists:
                    print(f"inserting LLM {row['name_normalized']}")
                    session.execute(insert(LLMMetadata).values(**row))
            session.commit()
        else:
            session.execute(delete(LLMMetadata))
            session.execute(insert(LLMMetadata).values(records))
            session.commit()


def ingest_benchmark_scores(
    *,
    records: list[dict[str, Any]],
    database: Database,
    normalizer: Normalizer,
    update: bool = True,
    skip_fk: bool = False,
):
    """
    Req 1.1.B: Parses uploaded CSV and saves to DB + FAISS index.
    If `update = True` rows will be appended to existing table,
    otherwise all rows will be deleted and recreated (dev usage).
    If `skip_fk = True` FK relations are NOT resolved (dev usage).

    NOTE: Has to be called after ingest_benchmark_dictionary and ingest_llm_metadata
    since it relies on normalized names for FK resolution (if `dummy_fk = False`)
    """
    # Normalization step: Needed especially for model names to resolve FKs.
    # MVL: Manual collection in score table should assure benchmark names *are* normalized
    bench_names = [r["original_benchmark_name"] for r in records]
    bench_variants = [r["original_benchmark_variant"] for r in records]
    bench_norm = normalizer.normalize_benchmark_names(bench_names, bench_variants)
    model_names = [r["original_model_name"] for r in records]
    model_norm = normalizer.normalize_model_names(model_names)

    # For FK resolution create lookup tables by unique-constraints
    fk_lookup_benchmark = {
        f"{_.name_normalized}#{_.variant if _.variant else ''}": _
        for _ in database.SessionLocal().query(BenchmarkDictionary).all()
    }
    fk_lookup_llm = {
        f"{_.name_normalized}": _ for _ in database.SessionLocal().query(LLMMetadata).all()
    }

    _z = zip(records, bench_norm, model_norm)
    for row, (bench_name_norm, bench_variant_norm), model_name_norm in _z:
        bench_variant_str = bench_variant_norm if bench_variant_norm else ""

        # Skip FK resolution if set
        if skip_fk:
            row["benchmark_id"] = 0
            row["model_id"] = 0
            continue

        # BenchmarkDictionary: We use normalized names for FK resolution in existing tables
        benchmark_fk = fk_lookup_benchmark.get(f"{bench_name_norm}#{bench_variant_str}")
        if not benchmark_fk:
            raise ValueError(
                f"Could not find benchmark FK for {bench_name_norm} {bench_variant_str}"
            )
        row["benchmark_id"] = benchmark_fk.id

        # LLMMetadata: We use normalized names for FK resolution in existing tables
        model_fk = fk_lookup_llm.get(f"{model_name_norm}")
        if not model_fk:
            raise ValueError(f"Could not find model FK for {model_name_norm}")
        row["model_id"] = model_fk.id

    with database.SessionLocal() as session:
        if update:
            for row in records:
                exists = (
                    session.query(BenchmarkScore)
                    .filter_by(
                        benchmark_id=row["benchmark_id"],
                        model_id=row["model_id"],
                        source_url=row["source_url"],
                        date_published=row["date_published"],
                    )
                    .first()
                )
                if not exists:
                    print(
                        f"inserting score for model_id {row['model_id']} on "
                        f"benchmark_id {row['benchmark_id']}"
                    )
                    session.execute(insert(BenchmarkScore).values(**row))
            session.commit()
        else:
            session.execute(delete(BenchmarkScore))
            session.execute(insert(BenchmarkScore).values(records))
            session.commit()


def ingest_model_normalized(
    *,
    raw_model_names: list[str],
    database: Database,
    update: bool = True,
) -> list[dict]:
    """
    Normalize raw model name strings and persist results to the
    `model_normalized` PostgreSQL table.

    One DB row is written per entry in `raw_model_names` — including duplicates —
    mirroring the source Google Sheet exactly.

    Args:
        raw_model_names:  Flat list of raw name strings, one per sheet row.
                          Blank strings are skipped; duplicates are kept.
        database:         Injected Database instance (provides SessionLocal).
        update:           If True  → append rows to existing table.
                          If False → wipe the table first, then bulk-insert.

    Returns:
        List of dicts (normalize() output) for all inserted rows.

    Raises:
        ValueError: If a normalized record fails Pydantic validation.

    Usage example:
        from .ingest import ingest_model_normalized

        ingest_model_normalized(
            raw_model_names=[r["model"] for r in rows],
            database=db,
            update=True,
        )
    """
    # ── 1. Normalize — one record per sheet row, blanks skipped ──────────────
    normalized: list[dict] = [
        normalize(name.strip())
        for name in raw_model_names
        if name.strip()
    ]

    # ── 2. Validate with Pydantic (catches bad data early) ───────────────────
    validated: list[dict] = []
    for record in normalized:
        try:
            schema = ModelNormalizedSchema(**record)
            validated.append(schema.model_dump())
        except Exception as exc:
            raise ValueError(
                f"Validation failed for raw='{record.get('raw')}': {exc}"
            ) from exc

    # ── 3. Write to DB ────────────────────────────────────────────────────────
    with database.SessionLocal() as session:
        if not update:
            # Dev / reset mode: wipe table first
            session.execute(delete(ModelNormalized))
            print("[model_normalized] Table wiped.")

        session.execute(insert(ModelNormalized).values(validated))
        session.commit()
        print(f"[model_normalized] Inserted {len(validated)} rows.")

    return validated