import pandas as pd
import re
import urllib.request
from io import StringIO
from collections import Counter


from collections.abc import Sequence
from typing import Optional
from llm_compass.config import Settings


# ── 0. INTERFACE (Andreas) ─────────────────────────────────────────────────────────
class Normalizer:
    """
    Centralizes normalization logic for model and benchmark names.
    This is a critical step before FK resolution and database insertion.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # Initialize LLM client here (e.g. Openrouter) with a small model for normalization tasks.

    def normalize_model_names(self, raw_names: list[str]) -> list[str]:
        """
        Uses an LLM call to standardize names.
        e.g., "llama-2-7b-chat-hf" -> "Llama 2 7B Chat"
        """
        # implementation using Openrouter client with a small model (e.g. gpt-3.5-turbo) to do the normalization.
        return [n.strip() for n in raw_names]  # Placeholder

    def normalize_benchmark_names(
        self, raw_names: list[str], raw_variants: Sequence[str | None] | None = None
    ) -> list[tuple[str, Optional[str]]]:
        """
        Uses an LLM call to standardize names and split into base name + benchmark variant if applicable.
        e.g., "mmlu-5shot" -> ("MMLU", "5 Shot")
        """
        # implementation using Openrouter client with a small model (e.g. gpt-3.5-turbo) to do the normalization.
        if raw_variants is None:
            raw_variants = [None] * len(raw_names)
        assert len(raw_names) == len(raw_variants), "raw_names and raw_variants must be same len"
        # Placeholder: no modification for MVP
        return [(n.strip(), v) for n, v in zip(raw_names, raw_variants)]


### Code section Nidhi ###
# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
SHEET_ID = "1a_4LmfIuKhhHzzZ3-xcO6jStHCIxbhgF60rRxoPfiMU"
GID = "984276769"
OUTPUT_FILE = "Normalized-data.xlsx"

# ── 2. EXPANDED PROVIDER KNOWLEDGE BASE ──────────────────────────────────────
PROVIDER_PATTERNS = [
    (r"claude|anthropic", "anthropic"),
    (r"gpt|openai|o[1-9]", "openai"),
    (r"gemini|google|gemma|palm", "google"),
    (r"llama|meta", "meta"),
    (r"mistral|mixtral|pixtral", "mistral"),
    (r"qwen|qwq|alibaba", "alibaba"),
    (r"deepseek", "deepseek"),
    (r"grok|xai", "xai"),
    (r"phi|microsoft", "microsoft"),
    (r"command|cohere", "cohere"),
    (r"kimi|moonshot", "moonshot"),
    (r"minimax", "minimax"),
    (r"glm|chatglm|zhipu|thudm", "zhipu"),
    (r"yi|01\.ai", "01-ai"),
    (r"baichuan", "baichuan"),
    (r"arctic|snowflake", "snowflake"),
    (r"dbrx|databricks", "databricks"),
    (r"solar|upstage", "upstage"),
    (r"granite|ibm", "ibm"),
    (r"samba|sambanova", "sambanova"),
    (r"cerebras", "cerebras"),
    (r"jamba|ai21", "ai21"),
    (r"starcoder|bigcode", "bigcode"),
    (r"falcon|tii", "tii"),
]

# ── 3. NORMALIZATION ENGINE ──────────────────────────────────────────────────

_CAMEL_PASSES = [
    (re.compile(r"([a-z])([A-Z])"), r"\1 \2"),
    (re.compile(r"([A-Z]{2,})([A-Z][a-z])"), r"\1 \2"),
    (re.compile(r"(\d)([A-Z][a-z])"), r"\1 \2"),
    (re.compile(r"([a-np-z])(\d)"), r"\1 \2"),
]

DATE_PATTERNS = [
    (r"\((\d{4})(\d{2})(\d{2})\)", lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    (r"\((\d{4}-\d{2}-\d{2})\)", lambda m: m.group(1)),
    (r"\b(\d{4}-\d{2}-\d{2})\b", lambda m: m.group(1)),
    (
        r"\((\d{1,2})/(\d{2,4})\)",
        lambda m: (
            f"{m.group(2) if len(m.group(2)) == 4 else '20' + m.group(2)}-{m.group(1).zfill(2)}"
        ),
    ),
]


def normalize_model(raw_name: str):
    if not raw_name or pd.isna(raw_name):
        return "n/a", "n/a", "unknown"
    name = str(raw_name).strip()

    for pat, repl in _CAMEL_PASSES:
        name = pat.sub(repl, name)

    # 1. Date
    date_val = None
    for pat, fmt in DATE_PATTERNS:
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            date_val = fmt(m)
            name = name.replace(m.group(0), " ")
            break

    # 2. Variant & Forced Reasoning Defaulting
    variant_val = "standard"
    if re.search(r"\bo\d|deepseek[- ]?r1", raw_name, re.IGNORECASE):
        variant_val = "reasoning"
    else:
        for pat, var in [
            (r"\bthinking\b", "thinking"),
            (r"\binstruct\b", "instruct"),
            (r"\bpreview\b", "preview"),
        ]:
            if re.search(pat, name, re.IGNORECASE):
                variant_val = var
                name = re.sub(pat, " ", name, flags=re.IGNORECASE)
                break

    # 3. Size Extraction
    size_val = None
    size_match = re.search(r"\b(\d+x?\d*(?:\.\d+)?[bBmMkK](?:[/-][aA]?\d+[bBmMkK])?)\b", name)
    if size_match:
        size_val = size_match.group(1).lower()
        name = name.replace(size_match.group(0), " ")

    # 4. Version Extraction
    v_match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", name)
    version_val = v_match.group(1) if v_match else None
    if version_val:
        name = name.replace(version_val, " ")

    # 5. Provider Anchoring with Fallback
    provider_val = None
    for pat, prov in PROVIDER_PATTERNS:
        if re.search(pat, raw_name, re.IGNORECASE):
            provider_val = prov
            name = re.sub(pat, "", name, flags=re.IGNORECASE)
            break

    if not provider_val:
        first_word_match = re.match(r"([a-zA-Z0-9]+)", str(raw_name).strip())
        provider_val = first_word_match.group(1).lower() if first_word_match else "unknown"

    # 6. ID Construction
    clean_family = re.sub(r"\(.*?\)|[^a-z0-9]+", "-", name.lower()).strip("-")
    base_parts = [provider_val, clean_family, version_val, size_val]
    base_id = re.sub(r"-+", "-", "-".join([str(p) for p in base_parts if p]).lower())

    full_parts = [base_id, variant_val, date_val]
    full_id = re.sub(
        r"-standard$", "", re.sub(r"-+", "-", "-".join([str(p) for p in full_parts if p]).lower())
    )

    return full_id, base_id, provider_val


# ── 4. EXECUTION & VALIDATION ─────────────────────────────────────────────


def main():
    csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
    try:
        print(f"📥 Loading {SHEET_ID}...")
        response = urllib.request.urlopen(csv_url)
        df = pd.read_csv(StringIO(response.read().decode("utf-8")))

        print(f"🔄 Processing 1,457 entries...")
        results = df["model"].apply(normalize_model)
        df["normalized_name"] = [r[0] for r in results]
        df["base_model_id"] = [r[1] for r in results]
        df["provider_found"] = [r[2] for r in results]

        # Validation Logic
        unique_providers = df["provider_found"].unique()
        unique_bases = df["base_model_id"].nunique()

        print("\n" + "=" * 40)
        print("📊 VALIDATION SUMMARY")
        print("=" * 40)
        print(f"Total Rows Processed:   {len(df)}")
        print(f"Unique Providers Found: {len(unique_providers)}")
        print(f"Unique Base Model IDs:  {unique_bases}")
        print("-" * 40)
        print("Top Providers by Entry Count:")
        for prov, count in Counter(df["provider_found"]).most_common(5):
            print(f" - {prov:<15}: {count}")
        print("=" * 40 + "\n")

        # Cleanup and Save
        output_df = df[["model", "normalized_name", "base_model_id"]]
        output_df.columns = ["model_name", "normalized_name", "base_model_id"]
        output_df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
        print(f"✅ Final Excel mapping saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
