# llm-compass

## Setup (dev notes)
- rename `.env.example` to `.env` and fill variables accordingly
- `pip install -e .` in the repo root

## Generata Database (dev notes)
For MVP, data is not updated automatically. Import manually curated data into the project via (from repo root):
- `python src/llm_compass/scripts/build_data.py` (Linux)
- `python src\llm_compass\scripts\build_data.py` (Windows)

Will store data in `storage/` (depending on settings in `.env`)

## Lauch App
- Run the following commands in 2 terminals (from repo root):
  - `streamlit run src\llm_compass\ui\app.py`
  - `fastapi dev src\llm_compass\api\main.py`

