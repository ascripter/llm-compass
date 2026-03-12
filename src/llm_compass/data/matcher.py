"""
Model Name Matcher
==================
Resolves arbitrary raw model-name strings (from benchmark sources) to
canonical LLMMetadata entries using a cascade of progressively relaxed
field-tuple lookups.

The matcher normalizes BOTH sides — reference names/aliases AND query
strings — via ``normalizer.normalize()``, then walks a cascade of match
levels from most specific (exact canonical_id) to least specific
(family-only).  The first level that yields ≥1 candidate wins.

Usage:
    from .matcher import ModelMatcher

    matcher = ModelMatcher()
    matcher.build_index([
        {"id": 1, "name_normalized": "claude-3.5-sonnet",
         "name_aliases": ["claude-3-5-sonnet-20241022"]},
        ...
    ])

    candidates = matcher.match("Claude 3.5 Sonnet (Thinking)")
    model_id   = matcher.resolve("Claude 3.5 Sonnet (Thinking)")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .normalizer import normalize


# ── Cascade definition ────────────────────────────────────────────────────────
# Each entry: (tier_label, tuple_of_field_names)
# Levels are tried top-to-bottom; first hit wins.

MATCH_CASCADE: list[tuple[str, tuple[str, ...]]] = [
    ("exact", ("canonical_id",)),
    ("base_id", ("base_id",)),
    ("full", ("provider", "family", "version", "size", "variant", "reasoning_effort", "date")),
    ("full_no_effort", ("provider", "family", "version", "size", "variant", "date")),
    ("full_no_provider", ("family", "version", "size", "variant", "reasoning_effort", "date")),
    ("core+date", ("family", "version", "size", "variant", "date")),
    ("core+effort", ("family", "version", "size", "variant", "reasoning_effort")),
    ("core", ("family", "version", "size", "variant")),
    ("no_size", ("family", "version", "variant")),
    ("no_variant", ("family", "version", "date")),
    ("family_version", ("family", "version")),
]

# Ordered tier labels for comparison (lower index = higher confidence).
_TIER_ORDER: dict[str, int] = {label: i for i, (label, _) in enumerate(MATCH_CASCADE)}


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _NormalizedRef:
    """Internal: one normalized reference entry (from name or alias)."""

    model_id: int
    name_normalized: str  # the LLMMetadata.name_normalized (shared across aliases)
    source_string: str  # the specific string that was normalized
    fields: dict  # output of normalize(source_string)


@dataclass(frozen=True)
class MatchCandidate:
    """A single match candidate returned to the caller."""

    model_id: int
    name_normalized: str  # LLMMetadata.name_normalized
    matched_via: str  # which reference string triggered the match
    tier: str  # cascade level label


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_key(fields: dict, field_names: tuple[str, ...]) -> tuple:
    """Build a hashable lookup key from selected fields."""
    return tuple(fields.get(f) for f in field_names)


# ── ModelMatcher ──────────────────────────────────────────────────────────────


class ModelMatcher:
    """
    Builds an index of reference models (from LLMMetadata rows) and matches
    arbitrary query strings against it using a cascade of field-tuple lookups.
    """

    def __init__(self) -> None:
        # maps:  tier_label → { key_tuple → list[_NormalizedRef] }
        self._maps: dict[str, dict[tuple, list[_NormalizedRef]]] = {}

    # ── Index building ────────────────────────────────────────────────────

    def build_index(self, models: list[dict]) -> None:
        """
        Populate the matcher index from LLMMetadata-shaped dicts.

        Args:
            models: list of dicts, each with keys:
                ``id``  (int), ``name_normalized`` (str),
                ``name_aliases`` (list[str] | None).
        """
        # Initialise empty maps for every cascade level
        self._maps = {label: defaultdict(list) for label, _ in MATCH_CASCADE}

        for model in models:
            model_id: int = model["id"]
            name_normalized: str = model["name_normalized"]
            aliases: list[str] = model.get("name_aliases") or []

            all_strings = [name_normalized] + [a for a in aliases if a]

            for s in all_strings:
                fields = normalize(s)
                ref = _NormalizedRef(
                    model_id=model_id,
                    name_normalized=name_normalized,
                    source_string=s,
                    fields=fields,
                )
                for label, field_names in MATCH_CASCADE:
                    key = _make_key(fields, field_names)
                    self._maps[label][key].append(ref)

    # ── Matching ──────────────────────────────────────────────────────────

    def match(self, query: str) -> list[MatchCandidate]:
        """
        Match a single query string against the reference index.

        Walks the cascade top-to-bottom and returns candidates from the
        *first* level that produces ≥1 hit.  Results are deduplicated by
        ``model_id`` (keeps the first ref encountered per model).

        Returns:
            List of ``MatchCandidate`` (may be empty).
        """
        q_fields = normalize(query)

        for label, field_names in MATCH_CASCADE:
            key = _make_key(q_fields, field_names)
            refs = self._maps.get(label, {}).get(key)
            if refs:
                return self._dedup(refs, label)

        return []

    def match_batch(self, queries: list[str]) -> dict[str, list[MatchCandidate]]:
        """Match multiple queries. Returns ``{raw_query: [candidates]}``."""
        return {q: self.match(q) for q in queries}

    # ── Resolution (convenience for FK lookup) ────────────────────────────

    def resolve(
        self,
        query: str,
        min_tier: str = "family_only",
    ) -> Optional[int]:
        """
        Return the ``model_id`` of the single best candidate, or ``None``.

        Returns ``None`` when:
        - no candidates found, or
        - best tier is less confident than *min_tier*, or
        - multiple distinct ``model_id`` values at the winning tier
          (ambiguous match).
        """
        candidates = self.match(query)
        if not candidates:
            return None

        tier = candidates[0].tier
        if _TIER_ORDER.get(tier, 999) > _TIER_ORDER.get(min_tier, 999):
            return None

        model_ids = {c.model_id for c in candidates}
        if len(model_ids) > 1:
            return None

        return candidates[0].model_id

    def resolve_batch(
        self,
        queries: list[str],
        min_tier: str = "family_only",
    ) -> dict[str, Optional[int]]:
        """Batch version of :meth:`resolve`."""
        return {q: self.resolve(q, min_tier=min_tier) for q in queries}

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _dedup(refs: list[_NormalizedRef], tier: str) -> list[MatchCandidate]:
        """Deduplicate refs by model_id, keeping first occurrence."""
        seen: set[int] = set()
        out: list[MatchCandidate] = []
        for ref in refs:
            if ref.model_id not in seen:
                seen.add(ref.model_id)
                out.append(
                    MatchCandidate(
                        model_id=ref.model_id,
                        name_normalized=ref.name_normalized,
                        matched_via=ref.source_string,
                        tier=tier,
                    )
                )
        return out
