# llm-compass

## Setup (dev notes)
rename `.env.example` to `.env` and fill variables accordingly

## Data (dev notes)
Build database initially via
- `python src/llm_compass/scripts/build_data.py` (Linux)
- `python src\llm_compass\scripts\build_data.py` (Windows)

Will store data in `storage/` (depending on settings in `.env`)

**NOTE:** FK resolution from scores to benchmark / models still t.b.d.

To fetch data from SQLite:

```python
    from llm_compass.config import get_settings
    from llm_compass.data.database import Database
    from llm_compass.data.models import BenchmarkDictionary, LLMMetadata, BenchmarkScore

    settings = get_settings()  # load_dotenv() and storage path
    db = Database(settings)
    with db.SessionLocal() as session:
        # SQLAlchemy code, i.e.
        results = session.query(BenchmarkDictionary).filter_by(name_normalized="GPQA")
        for row in results:
            print(row.name_normalized, row.variant)
```

