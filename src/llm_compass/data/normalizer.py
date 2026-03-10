"""
Model Name Normalizer  v2
=========================
Normalizes raw model name strings into structured canonical records
ready for PostgreSQL insertion via the ModelNormalized schema.

Output fields per record:
    raw · canonical_id · base_id · provider · family · version · size · variant · date · is_latest_alias

Design principles:
  1. Zero static model name lists — all rules are shape-based
  2. Provider detected via PROVIDER_PATTERNS (regex on token shape/brand slug)
     — slug fallback for unknown brands; new brands auto-handled
  3. Smart CamelCase tokenizer — no brand name whitelists
  4. nonthinking → variant=standard
  5. (high/medium/low reasoning) is inference config — stripped, no variant
  6. (YYMM) date format e.g. (2512) → 2025-12
  7. o-series family preserved: o4-mini stays o4-mini, not "mini"
  8. T=N and other sampling annotations stripped before canonicalisation
  9. fast+reasoning compound → variant=reasoning
 10. Collision-free: Thinking vs Nonthinking produce distinct canonical IDs

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
# PART 1 — TOKEN SHAPE CLASSIFIERS  (purely structural, zero word lists)
# =============================================================================

_DATE_SHAPES = [
    re.compile(r'^20\d{6}$'),               # 20250514
    re.compile(r'^20\d{2}-\d{2}-\d{2}$'),   # 2025-05-06
    re.compile(r'^\d{1,2}/\d{2,4}$'),       # 11/25   4/2025
    re.compile(r'^\d{4}$'),                  # any 4-digit: 2507 0324
    re.compile(r'^\d{2}$'),                  # lone 2-digit fragments
    re.compile(r'^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$', re.I),
]


def _is_date_token(tok: str) -> bool:
    return any(p.match(tok) for p in _DATE_SHAPES)


_SIZE_SHAPE_RE = re.compile(
    r'^(?:\d+x)?\d+(?:\.\d+)?[bmk]'
    r'(?:[x/\-]a?\d+(?:\.\d+)?[bmk]'
    r'|a\d+(?:\.\d+)?[bmk]'
    r')?$', re.I
)
_MOE_FRAG_RE  = re.compile(r'^a\d+(?:\.\d+)?[bmk]$', re.I)
_TIER_SIZE_RE = re.compile(r'^(mini|nano|small|lite|tiny)$', re.I)


def _is_size_token(tok: str) -> bool:
    return (bool(_SIZE_SHAPE_RE.match(tok))
            or bool(_MOE_FRAG_RE.match(tok))
            or bool(_TIER_SIZE_RE.match(tok)))


_VER_PREFIX_RE = re.compile(r'^v(\d+(?:\.\d+){0,2})$', re.I)
_VER_PLAIN_RE  = re.compile(r'^(\d+(?:\.\d+)*)$')
_NOISE_TOKENS  = frozenset({'t=1', 'chatgpt', 'chatgpt4'})


def _version_value(tok: str) -> Optional[str]:
    if _is_date_token(tok) or _is_size_token(tok):
        return None
    m = _VER_PREFIX_RE.match(tok)
    if m:
        return m.group(1)
    m = _VER_PLAIN_RE.match(tok)
    if m:
        return m.group(1)
    return None


# =============================================================================
# PART 2 — TOKENIZATION
# =============================================================================

_CAMEL_PASSES = [
    (re.compile(r'([A-Z]{2,})([A-Z][a-z])'),      r'\1 \2'),
    (re.compile(r'([a-z])([A-Z])'),                r'\1 \2'),
    (re.compile(r'(\d)([A-Z][a-z])'),              r'\1 \2'),
    (re.compile(r'(?:(?<=\s)|^)([a-np-z])(\d)'),   r'\1 \2'),
    (re.compile(r'([a-zA-Z])(\d+\.\d+)'),          r'\1 \2'),
    (re.compile(r'\s{2,}'),                         ' '),
]

_DATE_PAREN_RE = re.compile(
    r'^\s*(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{2,4}|\d{4}-\d{2}|\d{2,8})\s*$'
)
_SAMPLING_RE = re.compile(r'\(T\s*=\s*[\d.]+\)|\bT\s*=\s*[\d.]+\b', re.I)


def _split_camel(s: str) -> str:
    for pat, repl in _CAMEL_PASSES:
        s = pat.sub(repl, s)
    return s.strip()


def _tokenize(raw: str) -> list[str]:
    s = _SAMPLING_RE.sub(' ', raw.strip().strip('"'))

    variant_texts: list[str] = []
    for pc in re.findall(r'\(([^)]*)\)', s):
        if not _DATE_PAREN_RE.match(pc.strip()):
            variant_texts.append(pc)

    main = re.sub(r'\([^)]*\)', ' ', s)

    result: list[str] = []
    for raw_tok in re.split(r'[\s\-_/:\\+]+', main):
        if not raw_tok:
            continue
        sub = [t for t in re.split(r'\s+', _split_camel(raw_tok)) if t]
        _is_ver = lambda t: bool(re.match(r'^\d+(?:\.\d+)*$', t))
        _is_txt = lambda t: bool(re.match(r'^[a-zA-Z]+$', t))
        has_version_boundary = any(
            (_is_ver(sub[i]) and _is_txt(sub[i + 1])) or
            (_is_txt(sub[i]) and _is_ver(sub[i + 1]))
            for i in range(len(sub) - 1)
        )
        has_camel_split = (
            len(sub) > 1
            and all(_is_txt(t) for t in sub)
            and len(raw_tok) > 4
        )
        if has_version_boundary or has_camel_split:
            result.extend(t.lower() for t in sub)
        else:
            result.append(''.join(sub).lower())

    _SERIES_FUSE_RE = re.compile(r'^([rk]\d+)([a-z].+)$')
    split_result = []
    for tok in result:
        m = _SERIES_FUSE_RE.match(tok)
        if m:
            split_result.extend([m.group(1), m.group(2)])
        else:
            split_result.append(tok)
    result = split_result

    for pc in variant_texts:
        for t in re.split(r'[\s\-_/:\\+]+', _split_camel(pc)):
            if t:
                result.append(t.lower())

    return result


# =============================================================================
# PART 3 — DATE EXTRACTION
# =============================================================================

_DATE_PATTERNS = [
    (r'\((\d{4})(\d{2})(\d{2})\)',
     lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    (r'\(?(\d{4}-\d{2}-\d{2})\)?',
     lambda m: m.group(1)),
    (r'[Pp]review\s+(\d{2})-(\d{2})\b',
     lambda m: f"2025-{m.group(1)}-{m.group(2)}"),
    (r'[\(\-](\d{2})(\d{2})\)?(?:\b|$)',
     lambda m: f"20{m.group(1)}-{m.group(2)}" if int(m.group(1)) >= 20 else None),
    (r'\((\d{1,2})/(\d{2})\)',
     lambda m: f"20{m.group(2)}-{m.group(1).zfill(2)}"),
    (r'\((\d{1,2})/(\d{4})\)',
     lambda m: f"{m.group(2)}-{m.group(1).zfill(2)}"),
    (r'\b(\d{4})-(\d{2})\b',
     lambda m: f"{m.group(1)}-{m.group(2)}"),
]


def _extract_date(raw: str) -> Optional[str]:
    for pat, fmt in _DATE_PATTERNS:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            try:
                date = fmt(m)
                if date and re.match(r'\d{4}-\d{2}', date):
                    return date
            except Exception:
                pass
    return None


# =============================================================================
# PART 4 — VARIANT EXTRACTION
# =============================================================================

_VARIANT_TRIGGERS = [
    (re.compile(r'^non[-_]?thinking$', re.I), 'standard'),
    (re.compile(r'^nonthinking$',       re.I), 'standard'),
    (re.compile(r'^thinking$',          re.I), 'thinking'),
    (re.compile(r'^think$',             re.I), 'thinking'),
    (re.compile(r'^codex$',             re.I), 'codex'),
    (re.compile(r'^instruct$',          re.I), 'instruct'),
    (re.compile(r'^preview$',           re.I), 'preview'),
    (re.compile(r'^exp$',               re.I), 'preview'),
    (re.compile(r'^experimental$',      re.I), 'preview'),
    (re.compile(r'^reasoner$',          re.I), 'reasoning'),
    (re.compile(r'^reasoning$',         re.I), 'reasoning'),
]

_INFERENCE_NOISE_RE = re.compile(r'^(?:high|medium|low|default)$', re.I)


def _extract_variant(tokens: list[str]) -> tuple[str, list[str]]:
    consumed: set[int] = set()

    noise_idx     = {i for i, t in enumerate(tokens) if _INFERENCE_NOISE_RE.match(t)}
    reasoning_idx = {i for i, t in enumerate(tokens) if re.match(r'^reasoning$', t, re.I)}
    fast_idx      = {i for i, t in enumerate(tokens) if re.match(r'^fast$', t, re.I)}

    if noise_idx and reasoning_idx:
        consumed |= noise_idx | reasoning_idx

    variant = 'standard'
    remaining_reasoning = reasoning_idx - consumed
    if fast_idx and remaining_reasoning:
        consumed |= fast_idx | remaining_reasoning
        variant = 'reasoning'

    if variant == 'standard':
        for pat, label in _VARIANT_TRIGGERS:
            for i, tok in enumerate(tokens):
                if i not in consumed and pat.match(tok):
                    variant = label
                    consumed.add(i)
                    break
            if variant != 'standard':
                break

    if noise_idx and reasoning_idx:
        consumed |= noise_idx
    return variant, [t for i, t in enumerate(tokens) if i not in consumed]


# =============================================================================
# PART 5 — SIZE EXTRACTION
# =============================================================================

_SUBFAMILY_WORDS = frozenset({'flash', 'pro', 'ultra', 'max', 'plus', 'base', 'next', 'turbo'})


def _extract_size(tokens: list[str]) -> tuple[Optional[str], list[str]]:
    first_tier_idx = next(
        (i for i, t in enumerate(tokens) if _TIER_SIZE_RE.match(t)), None
    )
    tier_is_subfamily = (
        first_tier_idx is not None
        and first_tier_idx > 0
        and tokens[first_tier_idx - 1].lower() in _SUBFAMILY_WORDS
    )

    for i, tok in enumerate(tokens):
        if not _is_size_token(tok):
            continue
        if _TIER_SIZE_RE.match(tok) and tier_is_subfamily:
            continue
        _FUSED_MOE_RE = re.compile(r'^(\d+(?:\.\d+)?[bmk])(a\d+(?:\.\d+)?[bmk])$', re.I)
        fused = _FUSED_MOE_RE.match(tok)
        if fused:
            return f"{fused.group(1)}-{fused.group(2)}", tokens[:i] + tokens[i+1:]
        if (i + 1 < len(tokens)
                and _MOE_FRAG_RE.match(tokens[i + 1])
                and not _TIER_SIZE_RE.match(tok)):
            return f"{tok}-{tokens[i+1]}", tokens[:i] + tokens[i+2:]
        return tok, tokens[:i] + tokens[i+1:]
    return None, tokens


# =============================================================================
# PART 6 — VERSION EXTRACTION
# =============================================================================

_VERSION_FAMILY_RE = re.compile(r'^[rk]\d+(?:\.\d+)?$', re.I)


def _extract_version(tokens: list[str]) -> tuple[Optional[str], list[str]]:
    for i, tok in enumerate(tokens):
        if _VERSION_FAMILY_RE.match(tok):
            continue
        ver = _version_value(tok)
        if ver is not None:
            return ver, tokens[:i] + tokens[i+1:]
    return None, tokens


# =============================================================================
# PART 7 — PROVIDER DETECTION
# =============================================================================

PROVIDER_PATTERNS = [
    (re.compile(r'^claude$|^anthropic$',            re.I), 'anthropic'),
    (re.compile(r'^gptoss$',                        re.I), 'openai'),
    (re.compile(r'^gpt$|^openai$',                  re.I), 'openai'),
    (re.compile(r'^gemini$|^google$|^gemma$',        re.I), 'google'),
    (re.compile(r'^llama$|^meta$',                  re.I), 'meta'),
    (re.compile(r'^mistral$|^mixtral$|^devstral$',   re.I), 'mistral'),
    (re.compile(r'^qwen\d*$|^qwq$',                re.I), 'alibaba'),
    (re.compile(r'^deepseek$',                      re.I), 'deepseek'),
    (re.compile(r'^grok$|^xai$',                    re.I), 'xai'),
    (re.compile(r'^phi$|^microsoft$',               re.I), 'microsoft'),
    (re.compile(r'^command$|^cohere$',              re.I), 'cohere'),
    (re.compile(r'^kimi$|^moonshot$',               re.I), 'moonshot'),
    (re.compile(r'^minimax$',                       re.I), 'minimax'),
    (re.compile(r'^glm$|^chatglm$|^zhipu$',         re.I), 'zhipu'),
    (re.compile(r'^yi$',                            re.I), '01-ai'),
    (re.compile(r'^baichuan$',                      re.I), 'baichuan'),
    (re.compile(r'^arctic$|^snowflake$',            re.I), 'snowflake'),
    (re.compile(r'^dbrx$|^databricks$',             re.I), 'databricks'),
    (re.compile(r'^solar$|^upstage$',               re.I), 'upstage'),
    (re.compile(r'^granite$|^ibm$',                 re.I), 'ibm'),
    (re.compile(r'^falcon$|^tii$',                  re.I), 'tii'),
    (re.compile(r'^cerebras$',                      re.I), 'cerebras'),
    (re.compile(r'^jamba$',                         re.I), 'ai21'),
    (re.compile(r'^starcoder$|^bigcode$',           re.I), 'bigcode'),
]


def _detect_provider(tokens: list[str], raw: str) -> str:
    for pat, provider in PROVIDER_PATTERNS:
        for tok in tokens:
            if pat.match(tok):
                return provider
    first = re.split(r'[\s\-_]', raw.strip())[0]
    return re.sub(r'[^a-z0-9]', '', first.lower()) or 'unknown'


# =============================================================================
# PART 8 — O-SERIES DETECTION
# =============================================================================

_O_SERIES_RE = re.compile(r'\bo(\d+)(?:[- ]?(mini|nano|small))?\b', re.IGNORECASE)


def _detect_o_series(raw: str) -> tuple[bool, dict]:
    m = _O_SERIES_RE.search(raw)
    if not m:
        return False, {}
    num    = m.group(1)
    tier   = m.group(2).lower() if m.group(2) else None
    family = f"o{num}" + (f"-{tier}" if tier else "")
    return True, {'provider': 'openai', 'family': family, 'variant': 'reasoning'}


# =============================================================================
# PART 9 — FAMILY CONSTRUCTION
# =============================================================================

_FAMILY_DISCARD = frozenset({
    'model', 'ai', 'the', 'by', 'from', 'new', 'latest', 'updated',
    'instruct', 'thinking', 'reasoning', 'nonthinking', 'non',
    'preview', 'exp', 'experimental', 'fast', 'codex',
    'high', 'low', 'default', 'v',
    'mini', 'nano', 'small', 'lite', 'tiny',
    'adaptive',
})


def _build_family(tokens: list[str], provider: str) -> str:
    seen, parts = set(), []
    for tok in tokens:
        clean = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', tok)
        if not clean or clean in _FAMILY_DISCARD or _is_date_token(clean):
            continue
        if clean not in seen:
            seen.add(clean)
            parts.append(clean)
    return ('-'.join(parts) if parts else provider).lower()


# =============================================================================
# PART 9b — TOKEN MERGE MAP
# =============================================================================

_TOKEN_MERGES: list[tuple[tuple[str, ...], str]] = [
    (('deep', 'seek'), 'deepseek'),
    (('chat', 'gpt'),  'chatgpt'),
    (('alibaba',),     'qwen'),
]


def _merge_tokens(tokens: list[str]) -> list[str]:
    for seq, merged in _TOKEN_MERGES:
        n = len(seq)
        i = 0
        out: list[str] = []
        while i < len(tokens):
            if tuple(tokens[i:i + n]) == seq:
                out.append(merged)
                i += n
            else:
                out.append(tokens[i])
                i += 1
        tokens = out

    _VERSIONED_BRANDS = frozenset({'qwen'})
    i = 0
    out = []
    while i < len(tokens):
        if (tokens[i] in _VERSIONED_BRANDS
                and i + 1 < len(tokens)
                and re.match(r'^\d+$', tokens[i + 1])):
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


# =============================================================================
# PART 10 — MASTER normalize() FUNCTION
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
            version, size, variant, date, is_latest_alias

    Example:
        >>> normalize("Claude 3.7 Sonnet (Thinking)")
        {
            'raw': 'Claude 3.7 Sonnet (Thinking)',
            'canonical_id': 'claude-sonnet-3.7-thinking',
            'base_id': 'claude-sonnet-3.7-thinking',
            'provider': 'anthropic',
            'family': 'claude-sonnet',
            'version': '3.7',
            'size': None,
            'variant': 'thinking',
            'date': None,
            'is_latest_alias': True,
        }
    """
    is_o, o_fields = _detect_o_series(raw)
    date           = _extract_date(raw)
    tokens         = _tokenize(raw)
    tokens         = [t for t in tokens
                      if not _is_date_token(t) and t not in _NOISE_TOKENS]
    tokens         = _merge_tokens(tokens)

    variant, tokens = _extract_variant(tokens)

    if is_o:
        size   = None
        tokens = [t for t in tokens
                  if not _TIER_SIZE_RE.match(t) and not _MOE_FRAG_RE.match(t)]
    else:
        size, tokens = _extract_size(tokens)

    version, tokens = _extract_version(tokens)
    provider        = _detect_provider(tokens, raw)
    family          = _build_family(tokens, provider)

    if is_o:
        provider = o_fields['provider']
        family   = o_fields['family']
        if variant == 'standard':
            variant = o_fields['variant']

    if re.search(r'\bdeep[- ]?seek[- ]?r1\b', raw, re.IGNORECASE) and variant == 'standard':
        variant = 'reasoning'

    is_latest = (date is None) and (not re.search(r'\d{4}', raw))

    parts = [family]
    if version: parts.append(version)
    if size:    parts.append(size)
    parts.append(variant)
    if date:    parts.append(date)

    def _clean(p): return re.sub(r'-{2,}', '-', '-'.join(p).lower()).strip('-')

    cid     = _clean(parts)
    base_id = _clean([p for p in parts if p != date])

    return {
        'raw':             raw,
        'canonical_id':    cid,
        'base_id':         base_id,
        'provider':        provider,
        'family':          family,
        'version':         version,
        'size':            size,
        'variant':         variant,
        'date':            date,
        'is_latest_alias': is_latest,
    }