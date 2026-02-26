"""Script to manually (re)create the SQLite database and the FAISS index"""

import sys
from pathlib import Path

if __name__ == "__main__":
    # Add the project root to sys.path to allow running this script directly
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)

    from llm_compass.config import get_settings
    from llm_compass.data.database import Database
    from llm_compass.data.embedding import Embedding
    from llm_compass.data.normalizer import Normalizer
    from llm_compass.data.ingestion import (
        ingest_benchmark_dictionary,
        ingest_llm_metadata,
        ingest_benchmark_scores,
    )
    from llm_compass.data.read_source import (
        benchmark_dictionary_from_googlesheet,
        llm_metadata_from_googlesheet,
        benchmark_scores_from_googlesheet,
    )

    # No need to change directory manually with sys.path fix
    settings = get_settings()
    db = Database(settings)
    db.init_db()
    norm = Normalizer(settings)
    emb = Embedding(settings)

    records_benchmark = benchmark_dictionary_from_googlesheet()
    ingest_benchmark_dictionary(
        records=records_benchmark, database=db, normalizer=norm, embedding=emb, update=True
    )

    records_models = llm_metadata_from_googlesheet()
    ingest_llm_metadata(records=records_models, database=db, normalizer=norm, update=True)

    records_scores = benchmark_scores_from_googlesheet()
    ingest_benchmark_scores(
        records=records_scores, database=db, normalizer=norm, update=True, skip_fk=True
    )
