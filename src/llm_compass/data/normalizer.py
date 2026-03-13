"""
Model Name Normalizer  v3
=========================
Normalizes raw model name strings into structured canonical records
ready for PostgreSQL insertion via the ModelNormalized schema.

Output fields per record:
    raw · canonical_id · base_id · provider · family · version · size
    · variant · reasoning_effort · date · is_latest_alias

Design principles:
  1. Dual-path architecture: slug inputs pass through unchanged; human-readable
     inputs are converted to slug form first
  2. Brand-name protection: compound names like MiniMax, DeepSeek are never split
  3. canonical_id preserves input token order — no reordering
  4. variant is "" when not explicitly present (never "standard")
  5. Inference levels (xhigh/high/medium/low) are preserved in reasoning_effort
  6. "V"/"v" version prefixes are preserved in slugs (e.g. deepseek-v3.1)
  7. Subfamily tokens (Lite, Flash, Pro, Mini) are preserved in family
  8. Provider prefixes (Meta, Google, etc.) stripped only from human-readable input

Usage:
    from .normalizer import normalize, Normalizer

    # Single name
    record = normalize("Claude 3.7 Sonnet (Thinking)")

    # Batch — via Normalizer class (used by ingest pipeline)
    normalizer = Normalizer(settings)
    names = normalizer.normalize_model_names(raw_names)
"""

import re
from collections.abc import Sequence
from typing import Optional

from llm_compass.config import Settings


# ── 0. INTERFACE ───────────────────────────────────────────────────────────────


class Normalizer:
    """
    Centralizes normalization logic for model and benchmark names.
    This is a critical step before FK resolution and database insertion.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def normalize_model_names(self, raw_names: list[str]) -> list[str]:
        """
        Uses an LLM call to standardize names.
        e.g., "llama-2-7b-chat-hf" -> "Llama 2 7B Chat"
        """
        return [n.strip() for n in raw_names]  # Placeholder

    def normalize_benchmark_names(
        self, raw_names: list[str], raw_variants: Sequence[str | None] | None = None
    ) -> list[tuple[str, Optional[str]]]:
        """
        Uses an LLM call to standardize names and split into base name + variant.
        e.g., "mmlu-5shot" -> ("MMLU", "5 Shot")
        """
        if raw_variants is None:
            raw_variants = [None] * len(raw_names)
        assert len(raw_names) == len(raw_variants), "raw_names and raw_variants must be same len"
        return [(n.strip(), v) for n, v in zip(raw_names, raw_variants)]  # Placeholder


# =============================================================================
# PART 1 — BRAND PROTECTION
# =============================================================================
# Applied BEFORE CamelCase splitting to prevent compound brand names from being
# torn apart. Order matters: longer patterns first.

_BRAND_PROTECTIONS: list[tuple[re.Pattern, str]] = [
    # Compound brand names — prevent CamelCase splitting
    (re.compile(r"Hyper\s*Clova\s*X?", re.I), "hyperclovax"),
    (re.compile(r"Step\s*Fun", re.I), "stepfun"),
    (re.compile(r"Chat\s*GPT", re.I), "chatgpt"),
    (re.compile(r"Deep\s*Seek", re.I), "deepseek"),
    (re.compile(r"Mini\s*Max", re.I), "minimax"),
    (re.compile(r"Mi\s*Mo", re.I), "mimo"),
    (re.compile(r"Big\s*Code", re.I), "bigcode"),
    (re.compile(r"Star\s*Coder", re.I), "starcoder"),
    (re.compile(r"Chat\s*GLM", re.I), "chatglm"),
    (re.compile(r"Open\s*Chat", re.I), "openchat"),
    (re.compile(r"Bai\s*Chuan", re.I), "baichuan"),
    (re.compile(r"GPT[\s\-_]*Oss", re.I), "gpt-oss"),
    # Versioned brands: some brands fuse version into the name (qwen3, olmo3)
    # while others separate it (llama-3). Handle both patterns here.
    (re.compile(r"\bQwen[\s\-_]*(\d+(?:\.\d+)?)", re.I), r"qwen\1"),
    (re.compile(r"\bLlama[\s\-_]*(\d+(?:\.\d+)?)", re.I), r"llama-\1"),
    (re.compile(r"\bOLMo[\s\-_]*(\d+(?:\.\d+)?)", re.I), r"olmo-\1"),
    (re.compile(r"\bLFM[\s\-_]*(\d+(?:\.\d+)?)", re.I), r"lfm\1"),
]


# =============================================================================
# PART 2 — SLUG DETECTION
# =============================================================================

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9.\-:_]*$")


def _is_slug(s: str) -> bool:
    """True if s is already in slug format (lowercase, hyphens/dots/digits)."""
    return bool(_SLUG_RE.match(s))


# =============================================================================
# PART 3 — CAMELCASE SPLITTING (with brand protection already applied)
# =============================================================================

_CAMEL_PASSES = [
    (re.compile(r"([A-Z]{2,})([A-Z][a-z])"), r"\1 \2"),  # GPTOss -> GPT Oss
    (re.compile(r"([a-z])([A-Z])"), r"\1 \2"),  # camelCase -> camel Case
    (re.compile(r"(\d)([A-Z][a-z])"), r"\1 \2"),  # 4Turbo -> 4 Turbo
]


def _split_camel(s: str) -> str:
    """Split CamelCase into separate words, preserving already-protected brands."""
    for pat, repl in _CAMEL_PASSES:
        s = pat.sub(repl, s)
    return s


# =============================================================================
# PART 4 — PREPROCESSING & PARENTHESIZED CONTENT EXTRACTION
# =============================================================================

_SAMPLING_RE = re.compile(r"\(T\s*=\s*[\d.]+\)|\bT\s*=\s*[\d.]+\b", re.I)

# Inference-level qualifiers: (high reasoning), (low reasoning), (xhigh reasoning), etc.
_INFERENCE_RE = re.compile(
    r"\(\s*(max|xhigh|high|medium|low|default|adaptive)\s+(reasoning)\s*\)", re.I
)

# Standalone reasoning in parens: (reasoning)
_REASONING_PAREN_RE = re.compile(r"\(\s*reasoning\s*\)", re.I)

# Variant triggers in parentheses
_PAREN_VARIANT_RE = re.compile(
    r"\(\s*(thinking|think|non[-_]?thinking|nonthinking|instruct|preview|"
    r"experimental|exp|codex|reasoner|reasoning|base)\s*\)",
    re.I,
)

# Date in parentheses: (20250514), (2025-05-06), (2512), etc.
_DATE_PAREN_RE = re.compile(r"\(\s*(\d{4}-\d{2}-\d{2}|\d{8}|\d{4}-\d{2}|\d{1,2}/\d{2,4})\s*\)")

# Month abbreviation in parentheses: (Jan), (Feb), etc.
_MONTH_PAREN_RE = re.compile(r"\(\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\)", re.I)


def _preprocess(raw: str) -> str:
    """Strip quotes, whitespace, sampling annotations."""
    s = raw.strip().strip('"').strip("'").strip()
    s = _SAMPLING_RE.sub(" ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _extract_parenthesized(s: str) -> tuple[str, str, Optional[str], Optional[str]]:
    """
    Extract parenthesized content from human-readable input.

    Returns:
        (main_text, variant, reasoning_effort, paren_date)
    """
    variant = ""
    reasoning_effort = None
    paren_date = None

    # 1. Extract inference-level reasoning: "(High Reasoning)" -> variant=reasoning, effort=high
    m = _INFERENCE_RE.search(s)
    if m:
        reasoning_effort = m.group(1).lower()
        if reasoning_effort == "default":
            reasoning_effort = "medium"
        variant = "reasoning"
        s = s[: m.start()] + " " + s[m.end() :]

    # 2. Extract standalone "(reasoning)"
    if not variant:
        m = _REASONING_PAREN_RE.search(s)
        if m:
            variant = "reasoning"
            reasoning_effort = "medium"  # default when no level specified
            s = s[: m.start()] + " " + s[m.end() :]

    # 3. Extract other variant triggers: "(Thinking)", "(Instruct)", etc.
    if not variant:
        m = _PAREN_VARIANT_RE.search(s)
        if m:
            raw_variant = m.group(1).lower().replace("-", "").replace("_", "")
            variant = _VARIANT_MAP.get(raw_variant, raw_variant)
            s = s[: m.start()] + " " + s[m.end() :]

    # 4. Extract date from parentheses
    m = _DATE_PAREN_RE.search(s)
    if m:
        paren_date = m.group(1)
        s = s[: m.start()] + " " + s[m.end() :]

    # 4b. Extract month abbreviation from parentheses: (Jan), (Feb), etc.
    if not paren_date:
        m = _MONTH_PAREN_RE.search(s)
        if m:
            paren_date = m.group(1).lower()
            s = s[: m.start()] + " " + s[m.end() :]

    # 4c. Extract standalone reasoning effort: (xhigh), (high), (low), etc.
    if reasoning_effort is None:
        m = re.search(r"\(\s*(max|xhigh|high|medium|low|adaptive)\s*\)", s, re.I)
        if m:
            reasoning_effort = m.group(1).lower()
            s = s[: m.start()] + " " + s[m.end() :]

    # 5. Strip any remaining parenthesized content (unknown annotations)
    s = re.sub(r"\([^)]*\)", " ", s)

    return re.sub(r"\s{2,}", " ", s).strip(), variant, reasoning_effort, paren_date


# =============================================================================
# PART 5 — VARIANT DETECTION
# =============================================================================

# Map raw variant tokens to normalized form
_VARIANT_MAP: dict[str, str] = {
    "thinking": "thinking",
    "think": "thinking",
    "nonthinking": "non-thinking",
    "non-thinking": "non-thinking",
    "instruct": "instruct",
    "reasoning": "thinking",
    "reasoner": "thinking",
    "non-reasoning": "non-thinking",
    "nonreasoning": "non-thinking",
    "base": "base",
}

# Variant tokens that can appear as trailing segments in slugs
_VARIANT_SLUG_TOKENS = frozenset(_VARIANT_MAP.values()) | frozenset(
    {
        "it",  # gemma instruction-tuned suffix
    }
)

# Multi-segment variants that appear in reference names
_MULTI_SEGMENT_VARIANTS = [
    ("fast", "non", "reasoning"),
    ("fast", "reasoning"),
    ("thinking", "preview"),
    ("non", "thinking"),
    ("non", "reasoning"),
    ("deep", "research"),
]


def _detect_variant_in_slug(tokens: list[str]) -> tuple[str, Optional[str], list[int]]:
    """
    Detect variant from slug tokens (trailing position).

    Returns: (variant, reasoning_effort, list of consumed token indices)
    """
    n = len(tokens)
    reasoning_effort = None

    # Check multi-segment variants first (longest match)
    for pattern in _MULTI_SEGMENT_VARIANTS:
        plen = len(pattern)
        if n >= plen:
            tail = tuple(tokens[n - plen :])
            if tail == pattern:
                variant = "-".join(pattern)
                # Normalize compound variants
                if variant == "fast-reasoning":
                    variant = "fast-reasoning"
                elif variant == "fast-non-reasoning":
                    variant = "fast-non-reasoning"
                elif variant == "non-thinking":
                    variant = "non-thinking"
                elif variant == "non-reasoning":
                    variant = "non-reasoning"
                elif variant == "thinking-preview":
                    variant = "thinking"
                elif variant == "deep-research":
                    variant = "deep-research"
                indices = list(range(n - plen, n))
                return variant, reasoning_effort, indices

    # Check single-segment variants at the end
    if n >= 1 and tokens[-1] in _VARIANT_SLUG_TOKENS:
        variant = tokens[-1]
        # "it" suffix -> instruct (e.g. gemma-3-12b-it)
        if variant == "it":
            variant = "instruct"
        return variant, reasoning_effort, [n - 1]

    return "", None, []


# =============================================================================
# PART 6 — SIZE DETECTION
# =============================================================================

_SIZE_RE = re.compile(
    r"^(?:\d+x)?\d+(?:\.\d+)?[tbmk]$",
    re.I,
)
_MOE_SIZE_RE = re.compile(
    r"^(\d+(?:\.\d+)?[tbmk])[-]?(a\d+(?:\.\d+)?[tbmk])$",
    re.I,
)
_CONTEXT_SIZE_RE = re.compile(r"^\d+k$", re.I)  # e.g. "80k" — context window, NOT model size


def _is_size_token(tok: str) -> bool:
    """Check if token looks like a model size (70b, 8x22b, 235b-a22b)."""
    if _CONTEXT_SIZE_RE.match(tok):
        # Heuristic: "80k" is context window, "70b" is size
        # Only 'b', 't', 'm' suffixes are model sizes; 'k' is context window
        return tok[-1].lower() != "k"
    return bool(_SIZE_RE.match(tok)) or bool(_MOE_SIZE_RE.match(tok))


def _detect_size_in_tokens(tokens: list[str], skip: set[int]) -> tuple[Optional[str], list[int]]:
    """Find the first size token, return (size_string, consumed_indices)."""
    for i, tok in enumerate(tokens):
        if i in skip:
            continue
        # Check for fused MoE: "235b-a22b" as single token (from slug) or consecutive
        if _MOE_SIZE_RE.match(tok):
            return tok.lower(), [i]
        if _SIZE_RE.match(tok) and tok[-1].lower() != "k":
            # Check if next token is an active-param suffix: a22b, a3b
            if (
                i + 1 < len(tokens)
                and i + 1 not in skip
                and re.match(r"^a\d+(?:\.\d+)?[tbmk]$", tokens[i + 1], re.I)
            ):
                return f"{tok.lower()}-{tokens[i+1].lower()}", [i, i + 1]
            return tok.lower(), [i]
    return None, []


# =============================================================================
# PART 7 — VERSION DETECTION
# =============================================================================

# Patterns that look like versions but are actually family identifiers
_FAMILY_VERSION_RE = re.compile(r"^[ork]\d+(?:\.\d+)?$", re.I)


def _detect_version_in_tokens(
    tokens: list[str], skip: set[int]
) -> tuple[Optional[str], list[int]]:
    """
    Detect version token. Returns (version_string, consumed_indices).
    Preserves "v" prefix when present.
    """
    for i, tok in enumerate(tokens):
        if i in skip:
            continue
        # Skip family-version patterns like o3, r1, k2
        if _FAMILY_VERSION_RE.match(tok):
            continue
        # Skip size tokens
        if _is_size_token(tok):
            continue
        # Skip date-like tokens
        if _is_date_token(tok):
            continue
        # "v3", "v3.1", "v0.3" — version with v-prefix
        if re.match(r"^v\d+(?:\.\d+)*$", tok, re.I):
            return tok.lower(), [i]
        # "3.7", "2.5", "1.5" — dotted version (must have at least one dot)
        if re.match(r"^\d+\.\d+(?:\.\d+)*$", tok):
            return tok, [i]
        # Single digit only if preceded by a text-only token (e.g. "llama" "3")
        # but NOT if it's a very common family number pattern
        if re.match(r"^\d+$", tok) and i > 0 and not tokens[i - 1][-1].isdigit():
            # Avoid matching things like model series numbers that are part of the name
            # Only match if it looks like a version: 1-digit or 2-digit
            if len(tok) <= 2 and i not in skip:
                return tok, [i]
    return None, []


# =============================================================================
# PART 8 — DATE EXTRACTION
# =============================================================================

_DATE_PATTERNS = [
    # Full ISO date: 2025-05-14
    (
        re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
        lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
    ),
    # 8-digit date: 20250514 — preserve as-is
    (
        re.compile(r"\b(20\d{6})\b"),
        lambda m: m.group(1),
    ),
    # Preview date: preview 05-14
    (re.compile(r"[Pp]review\s+(\d{2})-(\d{2})\b"), lambda m: f"2025-{m.group(1)}-{m.group(2)}"),
    # Year-month: 2025-05
    (re.compile(r"\b(\d{4})-(\d{2})\b"), lambda m: f"{m.group(1)}-{m.group(2)}"),
    # Compact YYMM: -2512, -0324 (only if >= 20xx range)
    (
        re.compile(r"[\-](\d{2})(\d{2})(?:\b|$)"),
        lambda m: f"20{m.group(1)}-{m.group(2)}" if 20 <= int(m.group(1)) <= 30 else None,
    ),
    # Slash dates: (1/25), (4/2025)
    (re.compile(r"\((\d{1,2})/(\d{2})\)"), lambda m: f"20{m.group(2)}-{m.group(1).zfill(2)}"),
    (re.compile(r"\((\d{1,2})/(\d{4})\)"), lambda m: f"{m.group(2)}-{m.group(1).zfill(2)}"),
    # Trailing MMDD: 0905 (month 01-12, day 01-31) — preserve as-is
    (
        re.compile(r"(?:^|[\s\-_])(\d{4})$"),
        lambda m: (
            m.group(1)
            if 1 <= int(m.group(1)[:2]) <= 12 and 1 <= int(m.group(1)[2:]) <= 31
            else None
        ),
    ),
]

_DATE_SLUG_RE = re.compile(
    r"^(20\d{2})-?(\d{2})-?(\d{2})$"  # 2025-05-14 or 20250514
    r"|^(\d{4})$"  # 2512 (compact YYMM)
)


def _is_date_token(tok: str) -> bool:
    """Check if a token looks like a date component."""
    if re.match(r"^20\d{2}$", tok):
        return True  # year: 2025
    if re.match(r"^20\d{6}$", tok):
        return True  # 20250514
    if re.match(r"^20\d{2}-\d{2}-\d{2}$", tok):
        return True  # 2025-05-14
    if re.match(r"^20\d{2}-\d{2}$", tok):
        return True  # 2025-05
    # 4-digit YYMM: only if in plausible date range
    if re.match(r"^\d{4}$", tok):
        try:
            yy = int(tok[:2])
            mm = int(tok[2:])
            if 20 <= yy <= 30 and 1 <= mm <= 12:
                return True
        except ValueError:
            pass
        # 4-digit MMDD: month 01-12, day 01-31
        try:
            mm = int(tok[:2])
            dd = int(tok[2:])
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return True
        except ValueError:
            pass
    return False


def _extract_date(raw: str) -> Optional[str]:
    """Extract a date from the raw string."""
    for pat, fmt in _DATE_PATTERNS:
        m = pat.search(raw)
        if m:
            try:
                date = fmt(m)
                if date:
                    return date
            except Exception:
                pass
    return None


def _strip_date_suffix(slug: str, date: Optional[str]) -> str:
    """Remove trailing date suffix from a slug to produce base_id."""
    if not date:
        return slug
    # Try removing various date formats from the end
    # Full date: -2025-05-14
    if slug.endswith(f"-{date}"):
        return slug[: -len(date) - 1]
    # Compact formats at end
    clean_date = date.replace("-", "")
    if slug.endswith(f"-{clean_date}"):
        return slug[: -len(clean_date) - 1]
    # YYMM format: -2512
    if len(date) == 7 and date[4] == "-":
        yymm = date[2:4] + date[5:7]
        if slug.endswith(f"-{yymm}"):
            return slug[: -len(yymm) - 1]
    # YYMMDD format
    if len(date) == 10:
        yymmdd = date[2:4] + date[5:7] + date[8:10]
        if slug.endswith(f"-{yymmdd}"):
            return slug[: -len(yymmdd) - 1]
    return slug


def _detect_date_tokens(tokens: list[str]) -> list[int]:
    """Find indices of date-shaped tokens at the END of the token list."""
    indices = []
    # Walk backwards from the end; stop at first non-date token
    for i in range(len(tokens) - 1, -1, -1):
        if _is_date_token(tokens[i]):
            indices.append(i)
        else:
            break
    return indices


# =============================================================================
# PART 9 — PROVIDER DETECTION
# =============================================================================

PROVIDER_PATTERNS = [
    (re.compile(r"^claude$|^anthropic$", re.I), "anthropic"),
    (re.compile(r"^gptoss$", re.I), "openai"),
    (re.compile(r"^gpt$|^openai$|^chatgpt$", re.I), "openai"),
    (re.compile(r"^o\d+$", re.I), "openai"),
    (re.compile(r"^gemini$|^google$|^gemma$", re.I), "google"),
    (re.compile(r"^llama$|^meta$", re.I), "meta"),
    (
        re.compile(
            r"^mistral$|^mixtral$|^devstral$|^ministral$|^magistral$|^pixtral$|^codestral$|^voxtral$",
            re.I,
        ),
        "mistral",
    ),
    (re.compile(r"^qwen\d*$|^qwq$", re.I), "alibaba"),
    (re.compile(r"^deepseek$", re.I), "deepseek"),
    (re.compile(r"^grok$|^xai$", re.I), "xai"),
    (re.compile(r"^phi$|^microsoft$", re.I), "microsoft"),
    (re.compile(r"^command$|^cohere$", re.I), "cohere"),
    (re.compile(r"^kimi$|^moonshot$", re.I), "moonshot"),
    (re.compile(r"^minimax$", re.I), "minimax"),
    (re.compile(r"^glm$|^chatglm$|^zhipu$", re.I), "zhipu"),
    (re.compile(r"^yi$", re.I), "01-ai"),
    (re.compile(r"^baichuan$", re.I), "baichuan"),
    (re.compile(r"^arctic$|^snowflake$", re.I), "snowflake"),
    (re.compile(r"^dbrx$|^databricks$", re.I), "databricks"),
    (re.compile(r"^solar$|^upstage$", re.I), "upstage"),
    (re.compile(r"^granite$|^ibm$", re.I), "ibm"),
    (re.compile(r"^falcon$|^tii$", re.I), "tii"),
    (re.compile(r"^cerebras$", re.I), "cerebras"),
    (re.compile(r"^jamba$", re.I), "ai21"),
    (re.compile(r"^starcoder$|^bigcode$", re.I), "bigcode"),
    (re.compile(r"^nova$|^amazon$", re.I), "amazon"),
    (re.compile(r"^nemotron$|^nvidia$", re.I), "nvidia"),
    (re.compile(r"^mimo$", re.I), "xiaomi"),
    (re.compile(r"^ernie$|^baidu$", re.I), "baidu"),
    (re.compile(r"^doubao$|^seed$", re.I), "bytedance"),
    (re.compile(r"^step$|^stepfun$", re.I), "stepfun"),
    (re.compile(r"^ling$|^ring$", re.I), "bytedance"),
    (re.compile(r"^reka$", re.I), "reka"),
    (re.compile(r"^sonar$", re.I), "perplexity"),
    (re.compile(r"^cogito$", re.I), "prgx"),
    (re.compile(r"^hermes$", re.I), "nousresearch"),
    (re.compile(r"^olmo$|^olmo3$", re.I), "allenai"),
    (re.compile(r"^lfm\d*$", re.I), "liquid"),
    (re.compile(r"^openchat$", re.I), "openchat"),
    (re.compile(r"^sora$", re.I), "openai"),
    (re.compile(r"^dall$", re.I), "openai"),
    (re.compile(r"^flux$", re.I), "bfl"),
    (re.compile(r"^stable$", re.I), "stability"),
    (re.compile(r"^whisper$", re.I), "openai"),
    (re.compile(r"^hunyuan$", re.I), "tencent"),
]

# Provider prefixes to strip in human-readable path
# Maps provider_prefix -> set of family tokens that follow it
_PROVIDER_PREFIX_STRIP = {
    "meta": {"llama"},
    "google": {"gemini", "gemma"},
    "anthropic": {"claude"},
    "microsoft": {"phi"},
    "openai": {"gpt"},
    "nvidia": {"nemotron"},
    "amazon": {"nova"},
    "tencent": {"hunyuan"},
    "stepfun": {"step", "step3"},
    "rekaai": {"reka"},
}


def _detect_provider(tokens: list[str], raw: str = "") -> str:
    """Detect provider from token list."""
    for pat, provider in PROVIDER_PATTERNS:
        for tok in tokens:
            if pat.match(tok):
                return provider
    # Slug fallback: use first token
    if tokens:
        first = re.sub(r"[^a-z0-9]", "", tokens[0].lower())
        if first:
            return first
    return "unknown"


# =============================================================================
# PART 10 — FAMILY EXTRACTION
# =============================================================================

# True noise words to discard from family — NOT subfamily tokens
_FAMILY_NOISE = frozenset(
    {
        "model",
        "ai",
        "the",
        "by",
        "from",
        "new",
        "latest",
        "updated",
        "adaptive",
        "chat",
    }
)


def _build_family(
    tokens: list[str],
    provider: str,
    skip_indices: set[int],
) -> str:
    """
    Build family slug from remaining tokens after variant/size/version/date removal.
    Provider-prefix token at index 0 is also excluded if it matches provider.
    """
    # Determine if first token is a provider prefix to skip
    provider_skip = set()
    if tokens:
        for pat, prov in PROVIDER_PATTERNS:
            if pat.match(tokens[0]) and prov == provider:
                # Check if this is JUST the provider name (not a brand-family like "deepseek")
                # Only skip if there's a distinct family token after it
                if len(tokens) > 1 and tokens[0].lower() != tokens[1].lower():
                    # Special case: tokens like "deepseek" ARE the family — don't skip
                    tok_lower = tokens[0].lower()
                    is_family_brand = tok_lower in {
                        "deepseek",
                        "minimax",
                        "qwen",
                        "grok",
                        "claude",
                        "gemini",
                        "gemma",
                        "mistral",
                        "mixtral",
                        "llama",
                        "phi",
                        "jamba",
                        "falcon",
                        "granite",
                        "reka",
                        "sonar",
                        "cogito",
                        "hermes",
                        "olmo",
                        "olmo3",
                        "devstral",
                        "ministral",
                        "magistral",
                        "pixtral",
                        "codestral",
                        "voxtral",
                        "mimo",
                        "ernie",
                        "kimi",
                        "glm",
                        "chatglm",
                        "dbrx",
                        "arctic",
                        "command",
                        "nova",
                        "nemotron",
                        "sora",
                    }
                    if not is_family_brand:
                        provider_skip.add(0)
                break

    parts = []
    seen = set()
    for i, tok in enumerate(tokens):
        if i in skip_indices or i in provider_skip:
            continue
        clean = tok.lower().strip("-.")
        if not clean or clean in _FAMILY_NOISE:
            continue
        if clean not in seen:
            seen.add(clean)
            parts.append(clean)

    return "-".join(parts) if parts else provider


# =============================================================================
# PART 11 — HUMAN-READABLE TO SLUG CONVERSION
# =============================================================================


def _to_slug(s: str) -> str:
    """
    Convert a human-readable model name to slug format.
    Assumes brand protection has already been applied.
    """
    # CamelCase split
    s = _split_camel(s)

    # Normalize separators: spaces, underscores, slashes, colons, plus -> hyphens
    s = re.sub(r"[\s_/\\:]+", "-", s)

    # Lowercase
    s = s.lower()

    # Collapse multiple hyphens, strip leading/trailing
    s = re.sub(r"-{2,}", "-", s).strip("-")

    # Remove dots that are separators (but keep dots in versions like 3.5)
    # A dot between two letters is a separator; between digits is a version
    s = re.sub(r"(?<=[a-z])\.(?=[a-z])", "-", s)

    return s


def _strip_provider_prefix(slug: str) -> str:
    """Strip leading provider prefix from a slug when a known family follows."""
    parts = slug.split("-")
    if len(parts) < 2:
        return slug

    first = parts[0]
    second = parts[1]

    for prefix, families in _PROVIDER_PREFIX_STRIP.items():
        if first == prefix:
            # Check if second token (or second with digits attached) matches a family
            second_base = re.sub(r"\d+.*$", "", second)  # "llama3" -> "llama"
            if second in families or second_base in families:
                return "-".join(parts[1:])
    return slug


# =============================================================================
# PART 12 — MASTER normalize() FUNCTION
# =============================================================================


def normalize(raw: str) -> dict:
    """
    Normalize a single raw model name into a structured record matching
    the ModelNormalized DB schema.

    Args:
        raw: A single raw model name string.

    Returns:
        Dict with keys:
            raw, canonical_id, base_id, provider, family,
            version, size, variant, reasoning_effort, date, is_latest_alias
    """
    cleaned = _preprocess(raw)
    if not cleaned:
        return {
            "raw": raw,
            "canonical_id": "",
            "base_id": "",
            "provider": "unknown",
            "family": "unknown",
            "version": None,
            "size": None,
            "variant": "",
            "reasoning_effort": None,
            "date": None,
            "is_latest_alias": True,
        }

    # Extract date from original string
    date = _extract_date(raw)

    if _is_slug(cleaned):
        # ── SLUG PATH ──────────────────────────────────────────────────────
        canonical_id = cleaned
        base_id = _strip_date_suffix(cleaned, date)
        tokens = cleaned.split("-")

        # Detect variant from slug tokens
        variant, reasoning_effort, variant_indices = _detect_variant_in_slug(tokens)

        # Build skip set for family extraction
        skip = set(variant_indices)

        # Detect size
        size, size_indices = _detect_size_in_tokens(tokens, skip)
        skip.update(size_indices)

        # Detect version
        version, version_indices = _detect_version_in_tokens(tokens, skip)
        skip.update(version_indices)

        # Detect date tokens (at end of slug)
        date_indices = _detect_date_tokens(tokens)
        skip.update(date_indices)

        # Detect provider
        provider = _detect_provider(tokens, raw)

        # Build family from remaining tokens
        family = _build_family(tokens, provider, skip)

    else:
        # ── HUMAN-READABLE PATH ────────────────────────────────────────────

        # Apply brand protection before any splitting
        protected = cleaned
        for pat, replacement in _BRAND_PROTECTIONS:
            protected = pat.sub(replacement, protected)

        # Extract parenthesized content
        main_text, paren_variant, reasoning_effort, paren_date = _extract_parenthesized(protected)

        if paren_date and not date:
            date = paren_date

        # Convert main text to slug
        main_slug = _to_slug(main_text)

        # Strip provider prefix
        main_slug = _strip_provider_prefix(main_slug)

        # Append explicit variant from parentheses if not already in slug
        if paren_variant and not main_slug.endswith(f"-{paren_variant}"):
            # Also check multi-word variants
            if not any(
                main_slug.endswith(f"-{v}") for v in [paren_variant] + paren_variant.split("-")
            ):
                main_slug = f"{main_slug}-{paren_variant}"

        # Set variant from parenthesized or detect from slug tokens
        tokens = main_slug.split("-")

        # Detect date tokens early so variant detection sees the right trailing token
        date_idx = _detect_date_tokens(tokens)

        if paren_variant:
            variant = paren_variant
            # Remove variant tokens from the end of the token list for family extraction
            variant_indices = []
            variant_parts = paren_variant.split("-")
            if tokens[-len(variant_parts) :] == variant_parts:
                variant_indices = list(range(len(tokens) - len(variant_parts), len(tokens)))
        else:
            # Strip date tokens from the end before variant detection
            non_date_tokens = [t for i, t in enumerate(tokens) if i not in date_idx]
            paren_effort = reasoning_effort
            variant, reasoning_effort, vi = _detect_variant_in_slug(non_date_tokens)
            if paren_effort is not None:
                reasoning_effort = paren_effort
            # Map indices back to original token list
            non_date_indices = [i for i in range(len(tokens)) if i not in date_idx]
            variant_indices = [non_date_indices[i] for i in vi]

        canonical_id = main_slug
        # Only append date if not already present in the slug
        if date and not main_slug.endswith(date):
            canonical_id = f"{main_slug}-{date}"
        base_id = main_slug

        # Build skip set for family extraction
        skip = set(variant_indices)

        # Detect size
        size, size_indices = _detect_size_in_tokens(tokens, skip)
        skip.update(size_indices)

        # Detect version
        version, version_indices = _detect_version_in_tokens(tokens, skip)
        skip.update(version_indices)

        # Detect date tokens
        date_idx = _detect_date_tokens(tokens)
        skip.update(date_idx)

        # Detect provider
        provider = _detect_provider(tokens, raw)

        # Build family
        family = _build_family(tokens, provider, skip)

    # Determine is_latest_alias
    is_latest = date is None and not re.search(r"\d{4}", raw)

    return {
        "raw": raw,
        "canonical_id": canonical_id,
        "base_id": base_id,
        "provider": provider,
        "family": family,
        "version": version,
        "size": size,
        "variant": variant,
        "reasoning_effort": reasoning_effort,
        "date": date,
        "is_latest_alias": is_latest,
    }
