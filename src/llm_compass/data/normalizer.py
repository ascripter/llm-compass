"""
Model Name Normalizer  v2
=========================
Reads all rows from the ingestion template Google Sheet, normalizes every
model name into a structured canonical record, and writes results to an
Excel file (.xlsx).

Output fields per row:
    provider · family · version · size · variant · date · canonical_id

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
    python normalizer.py                        # fetch live Google Sheet
    python normalizer.py --csv local.csv        # skip network fetch
    python normalizer.py --output results.xlsx
    python normalizer.py --no-save              # print summary only
"""

import argparse
import csv
import io
import re
import sys
import urllib.request
from collections import defaultdict
from typing import Optional

from openpyxl import Workbook

# ── Configuration ──────────────────────────────────────────────────────────────
SHEET_ID       = "1a_4LmfIuKhhHzzZ3-xcO6jStHCIxbhgF60rRxoPfiMU"
GID            = "984276769"
DEFAULT_OUTPUT = "normalized_model_names.xlsx"


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


# Covers: 70b  1.5b  235b  8x7b  480b/a35b  235b-a22b  270m  180k
_SIZE_SHAPE_RE = re.compile(
    r'^(?:\d+x)?\d+(?:\.\d+)?[bmk]'
    r'(?:[x/\-]a?\d+(?:\.\d+)?[bmk]'   # delimited:  30b-a3b  235b-a22b
    r'|a\d+(?:\.\d+)?[bmk]'              # fused:      30ba3b   235ba22b
    r')?$', re.I
)
_MOE_FRAG_RE  = re.compile(r'^a\d+(?:\.\d+)?[bmk]$', re.I)   # a22b  a3b
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

# Smart CamelCase splitting — four passes (from reference code):
#   1. Acronym-end:  GPTSonnet → GPT Sonnet
#   2. CamelCase:    proPreview → pro Preview
#   3. Digit→Upper:  4Sonnet   → 4 Sonnet
#   4. Word-start lower→digit: flash2 → flash 2
#      (lookbehind preserves Qwen3/GLM4 mid-word)
_CAMEL_PASSES = [
    (re.compile(r'([A-Z]{2,})([A-Z][a-z])'),      r'\1 \2'),
    (re.compile(r'([a-z])([A-Z])'),                r'\1 \2'),
    (re.compile(r'(\d)([A-Z][a-z])'),              r'\1 \2'),
    (re.compile(r'(?:(?<=\s)|^)([a-np-z])(\d)'),   r'\1 \2'),
    # Split letter→decimal: Opus4.6 → Opus 4.6  (decimal only, not plain suffix digits)
    (re.compile(r'([a-zA-Z])(\d+\.\d+)'),         r'\1 \2'),
    (re.compile(r'\s{2,}'),                         ' '),
]

# Parens whose entire content is a date/build-ID are fully suppressed
_DATE_PAREN_RE = re.compile(
    r'^\s*(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{2,4}|\d{4}-\d{2}|\d{2,8})\s*$'
)

# Sampling annotations stripped before tokenisation
_SAMPLING_RE = re.compile(r'\(T\s*=\s*[\d.]+\)|\bT\s*=\s*[\d.]+\b', re.I)


def _split_camel(s: str) -> str:
    for pat, repl in _CAMEL_PASSES:
        s = pat.sub(repl, s)
    return s.strip()


def _tokenize(raw: str) -> list[str]:
    """
    Lowercase token list from a raw model name.

    - Strips sampling annotations (T=1) first
    - Fully suppresses date parens: (9/25) (2512) (20250514)
    - Extracts semantic parens: (Thinking) (high reasoning) → appended tokens
    - CamelCase-splits each delimiter-separated piece, then re-joins its
      sub-tokens as one compound to preserve brand names:
        DeepSeek → 'deepseek'   Qwen3 → 'qwen3'   BrandNew → 'brandnew'
      Real-delimiter pieces (spaces) produce separate tokens as normal.
    """
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
        # Rejoin sub-tokens as one compound UNLESS a boundary is present:
        #   version ↔ text:  ["3.7","Sonnet"] or ["opus","4.6"] — keep separate
        #   camel pure-text: ["Magistral","Medium"] — keep separate
        #     (known brand compounds are re-merged downstream by _merge_tokens)
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

    # Split series-id+word fusions that arrive with no camel boundary:
    # k2thinking → k2, thinking    r1distill → r1, distill
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
# PART 3 — DATE EXTRACTION  (on raw string, before tokenization)
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
#
# Priority order:
#   nonthinking > thinking > codex > fast+reasoning(compound) > instruct >
#   preview > reasoner > reasoning
#
# Inference-config rule:
#   noise-qualifier (high/medium/low/default) + "reasoning" → consumed, no variant
#
# Compound rule:
#   "fast" + "reasoning" together → variant=reasoning (both consumed)

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

    # Inference-config: noise-qualifier + reasoning → suppress both, no variant
    if noise_idx and reasoning_idx:
        consumed |= noise_idx | reasoning_idx

    # Compound: fast + reasoning → reasoning (when reasoning not already suppressed)
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

    # Only consume noise tokens when they were part of an inference-config pair.
    # A standalone 'medium' (e.g. "Magistral Medium") is a legitimate tier word.
    if noise_idx and reasoning_idx:
        consumed |= noise_idx
    return variant, [t for i, t in enumerate(tokens) if i not in consumed]


# =============================================================================
# PART 5 — SIZE EXTRACTION
# =============================================================================

# Tier label (mini/nano/small/lite) is a subfamily qualifier when immediately
# preceded by one of these words — it belongs in the family, not size field.
# e.g. "Flash Lite" → lite stays in family; "Devstral small" → size=small
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
        # Normalise fused MoE token: 30ba3b → 30b-a3b
        _FUSED_MOE_RE = re.compile(r'^(\d+(?:\.\d+)?[bmk])(a\d+(?:\.\d+)?[bmk])$', re.I)
        fused = _FUSED_MOE_RE.match(tok)
        if fused:
            return f"{fused.group(1)}-{fused.group(2)}", tokens[:i] + tokens[i+1:]
        # Merge adjacent MoE fragment: 235b + a22b → 235b-a22b
        if (i + 1 < len(tokens)
                and _MOE_FRAG_RE.match(tokens[i + 1])
                and not _TIER_SIZE_RE.match(tok)):
            return f"{tok}-{tokens[i+1]}", tokens[:i] + tokens[i+2:]
        return tok, tokens[:i] + tokens[i+1:]
    return None, tokens


# =============================================================================
# PART 6 — VERSION EXTRACTION
# =============================================================================

_VERSION_FAMILY_RE = re.compile(r'^[rk]\d+(?:\.\d+)?$', re.I)  # r1 r2 k2 k2.5


def _extract_version(tokens: list[str]) -> tuple[Optional[str], list[str]]:
    for i, tok in enumerate(tokens):
        if _VERSION_FAMILY_RE.match(tok):
            continue
        ver = _version_value(tok)
        if ver is not None:
            return ver, tokens[:i] + tokens[i+1:]
    return None, tokens


# =============================================================================
# PART 7 — PROVIDER DETECTION  (PROVIDER_PATTERNS + slug fallback)
# =============================================================================
#
# Patterns match against individual tokens from the tokenized model name.
# Each pattern is a structural brand-slug regex, not a model name list.
# Order matters: most-specific patterns first (gpt-oss before gpt).
# Unknown brands get a slug of the first word automatically.

PROVIDER_PATTERNS = [
    (re.compile(r'^claude$|^anthropic$',            re.I), 'anthropic'),
    (re.compile(r'^gptoss$',                        re.I), 'openai'),   # gpt-oss compound
    (re.compile(r'^gpt$|^openai$',                  re.I), 'openai'),
    (re.compile(r'^gemini$|^google$|^gemma$',        re.I), 'google'),
    (re.compile(r'^llama$|^meta$',                  re.I), 'meta'),
    (re.compile(r'^mistral$|^mixtral$|^devstral$',   re.I), 'mistral'),
    (re.compile(r'^qwen\d*$|^qwq$',                re.I), 'alibaba'),  # qwen qwen3 qwen2.5
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
    # Slug fallback: first word of raw name
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
# Brand names that are sometimes written with a delimiter (space or dash) between
# their parts are listed here as consecutive token sequences → single merged token.
# This is NOT a model list — it's a brand-slug alias list. New entries can be
# added without touching any other part of the pipeline.

_TOKEN_MERGES: list[tuple[tuple[str, ...], str]] = [
    (('deep', 'seek'), 'deepseek'),
    (('chat', 'gpt'),  'chatgpt'),
    # Company-name aliases → canonical product family token
    (('alibaba',),     'qwen'),
]


def _merge_tokens(tokens: list[str]) -> list[str]:
    """Replace consecutive token sequences with their merged form."""
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

    # Versioned brand merges: qwen + plain-integer → qwen{N}
    # e.g. ['qwen', '2', 'vl'] → ['qwen2', 'vl']  (matches Qwen2-VL tokenization)
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
    Normalize a single raw model name into a structured canonical record.

    Pipeline:
        raw string
          ├─ o-series detection      (needs original form)
          ├─ date extraction         (needs original form)
          ├─ tokenize                (paren-aware, CamelCase-split, lowercase)
          │    token list
          │      ├─ drop date-shaped + explicit noise tokens
          │      ├─ variant extraction   (priority-ordered; inference config suppressed)
          │      ├─ size extraction      (shape-based; MoE merge; tier-subfamily rule)
          │      ├─ version extraction   (shape-based; r/k family tokens skipped)
          │      ├─ provider detection   (PROVIDER_PATTERNS → slug fallback)
          │      └─ family construction  (remaining tokens; unknowns kept)
          └─ canonical ID:  family[-version][-size]-variant[-date]
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

    # DeepSeek R1 architecture always → reasoning
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


# =============================================================================
# PART 11 — DATA LOADING
# =============================================================================

def _fetch_sheet_csv(sheet_id: str, gid: str) -> list[dict]:
    url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}"
           f"/export?format=csv&gid={gid}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return list(csv.DictReader(io.StringIO(resp.read().decode("utf-8"))))


def _load_csv_file(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _filter_rows(rows: list[dict]) -> list[dict]:
    """Keep all rows — blank model rows included (written as empty in xlsx)."""
    return rows


# =============================================================================
# PART 12 — XLSX OUTPUT
# =============================================================================

def build_xlsx(rows: list[dict], norm_map: dict, output_path: str) -> None:
    """
    Single sheet, no styling.
    Columns: model | canonical_id | provider | family | version | size | variant | date
    One output row per input row, in original sheet order.
    Blank model rows get empty normalization fields.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Model Normalization"
    ws.freeze_panes = "A2"

    ws.append(["model", "canonical_id", "base_id", "provider", "family", "version", "size", "variant", "date"])

    for row in rows:
        model = row.get('model', '').strip()
        if model:
            n = norm_map.get(model, normalize(model))
            ws.append([
                model,
                n['canonical_id'],
                n['base_id'],
                n['provider'],
                n['family'],
                n['version'] or '',
                n['size'] or '',
                n['variant'],
                n['date'] or '',
            ])
        else:
            ws.append(['', '', '', '', '', '', '', '', ''])

    wb.save(output_path)
    print(f"✅  Saved → {output_path}  ({len(rows)} rows)")


# =============================================================================
# PART 13 — CONSOLE SUMMARY
# =============================================================================

def _print_summary(rows: list[dict], norm_map: dict) -> None:
    by_provider: dict[str, int] = defaultdict(int)
    by_variant:  dict[str, int] = defaultdict(int)
    for n in norm_map.values():
        by_provider[n['provider']] += 1
        by_variant[n['variant']]   += 1

    unique_cids = len({n['canonical_id'] for n in norm_map.values()})
    print()
    print("=" * 52)
    print("  NORMALIZATION SUMMARY")
    print("=" * 52)
    print(f"  Total sheet rows      : {len(rows)}")
    print(f"  Unique raw names      : {len(norm_map)}")
    print(f"  Unique canonical IDs  : {unique_cids}")
    print()
    print("  By provider:")
    for p, c in sorted(by_provider.items(), key=lambda x: -x[1]):
        print(f"    {p:<22} {c:>4}")
    print()
    print("  By variant:")
    for v, c in sorted(by_variant.items(), key=lambda x: -x[1]):
        print(f"    {v:<22} {c:>4}")
    print("=" * 52)


# =============================================================================
# PART 14 — MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Normalize AI model names and write to Excel'
    )
    parser.add_argument('--sheet',   default=SHEET_ID,       help='Google Sheet ID')
    parser.add_argument('--gid',     default=GID,            help='Sheet tab GID')
    parser.add_argument('--csv',     default=None,           help='Local CSV (skips fetch)')
    parser.add_argument('--output',  default=DEFAULT_OUTPUT, help='Output .xlsx path')
    parser.add_argument('--no-save', action='store_true',    help='Print summary only')
    args = parser.parse_args()

    if args.csv:
        print(f"📂  Loading local CSV: {args.csv}")
        rows = _load_csv_file(args.csv)
    else:
        print(f"📥  Fetching Google Sheet  (id={args.sheet}  gid={args.gid})...")
        try:
            rows = _fetch_sheet_csv(args.sheet, args.gid)
            print(f"    ✓ Fetched {len(rows)} rows")
        except Exception as e:
            print(f"\n❌  Failed to fetch sheet: {e}")
            print("    Make sure the sheet is publicly shared (Anyone with link → Viewer).")
            print("    Or pass a local export with --csv path/to/export.csv")
            sys.exit(1)

    rows = _filter_rows(rows)
    print(f"    ✓ {len(rows)} total rows loaded")

    unique = list({r['model'].strip() for r in rows if r.get('model', '').strip()})
    print(f"\n🔄  Normalizing {len(unique)} unique model names...")
    norm_map = {name: normalize(name) for name in unique}

    _print_summary(rows, norm_map)

    if not args.no_save:
        print(f"\n📊  Building Excel workbook ({len(rows)} rows)...")
        build_xlsx(rows, norm_map, args.output)


if __name__ == '__main__':
    main()
