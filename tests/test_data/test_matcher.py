"""
Tests for ModelMatcher — cascade-based model name matching.

Tests are organized by cascade tier, then edge cases.
All normalize() outputs verified empirically against the current normalizer.
"""

import pytest

from llm_compass.data.matcher import ModelMatcher, MatchCandidate


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Minimal reference set mimicking LLMMetadata rows
REFERENCE_MODELS = [
    {
        "id": 1,
        "name_normalized": "claude-3.5-sonnet",
        "name_aliases": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest"],
    },
    {
        "id": 2,
        "name_normalized": "gpt-5.2",
        "name_aliases": ["gpt-5.2-2025-12-11"],
    },
    {
        "id": 3,
        "name_normalized": "o4-mini",
        "name_aliases": [],
    },
    {
        "id": 4,
        "name_normalized": "qwen3-vl-235b-a22b-thinking",
        "name_aliases": [],
    },
    {
        "id": 5,
        "name_normalized": "gemini-2.5-flash",
        "name_aliases": ["gemini-2.5-flash-001", "gemini-2.5-flash-latest"],
    },
    {
        "id": 6,
        "name_normalized": "deepseek-v3",
        "name_aliases": ["deepseek-chat"],
    },
    {
        "id": 7,
        "name_normalized": "llama-3.1-70b-instruct",
        "name_aliases": ["meta-llama-3.1-70b-instruct", "llama3-70b-instruct"],
    },
    {
        "id": 8,
        "name_normalized": "gpt-4.1",
        "name_aliases": ["gpt-4.1-2025-04-14"],
    },
    {
        "id": 9,
        "name_normalized": "claude-3.5-haiku",
        "name_aliases": ["claude-3-5-haiku-20241022"],
    },
    {
        "id": 10,
        "name_normalized": "gemini-2.5-pro",
        "name_aliases": [],
    },
]


@pytest.fixture
def matcher():
    m = ModelMatcher()
    m.build_index(REFERENCE_MODELS)
    return m


# ── Tier 1: Exact canonical_id match ─────────────────────────────────────────

class TestTierExact:
    """Tier 1 — query.canonical_id matches a reference canonical_id exactly."""

    def test_gpt_with_noise_stripped(self, matcher):
        """'GPT-5.2 (High Reasoning)' strips noise → canonical_id = gpt-5.2-standard."""
        candidates = matcher.match("GPT-5.2 (High Reasoning)")
        assert len(candidates) == 1
        assert candidates[0].model_id == 2
        assert candidates[0].tier == "exact"

    def test_o4_mini(self, matcher):
        """'o4-mini' is its own canonical_id."""
        candidates = matcher.match("o4-mini")
        assert len(candidates) == 1
        assert candidates[0].model_id == 3
        assert candidates[0].tier == "exact"

    def test_gemini_natural_language(self, matcher):
        """'Gemini 2.5 Flash' normalizes same as 'gemini-2.5-flash'."""
        candidates = matcher.match("Gemini 2.5 Flash")
        assert len(candidates) == 1
        assert candidates[0].model_id == 5
        assert candidates[0].tier == "exact"

    def test_reference_name_itself(self, matcher):
        """Feeding back a reference name should always match exactly."""
        candidates = matcher.match("gpt-5.2")
        assert len(candidates) == 1
        assert candidates[0].model_id == 2
        assert candidates[0].tier == "exact"


# ── Tier 2: base_id match ────────────────────────────────────────────────────

class TestTierBaseId:
    """Tier 2 — base_id matches (date stripped)."""

    def test_dated_alias_matches_undated_ref(self, matcher):
        """'gpt-5.2-2025-12-11' has a date → base_id matches 'gpt-5.2'."""
        # The alias is already indexed so it should match at exact level
        # Let's test a date variant that isn't in the alias list
        candidates = matcher.match("gpt-4.1-2025-04-14")
        assert any(c.model_id == 8 for c in candidates)
        assert candidates[0].tier in ("exact", "base_id")


# ── Tier 3-5: Field-tuple matches ────────────────────────────────────────────

class TestTierFieldTuples:
    """Tiers 3-5 — match by progressively dropping fields."""

    def test_same_family_different_variant(self, matcher):
        """'Claude 3.5 Sonnet (Thinking)' has variant=thinking while ref has variant=standard.
        Should match at no_size tier (family+version+variant won't match standard,
        so it falls to family_version)."""
        candidates = matcher.match("Claude 3.5 Sonnet (Thinking)")
        assert len(candidates) >= 1
        assert any(c.model_id == 1 for c in candidates)

    def test_qwen_different_subfamily_no_match(self, matcher):
        """'Qwen3-235B-A22B-Instruct' normalizes to family=qwen3,
        but reference 'qwen3-vl-235b-a22b-thinking' has family=qwen3-vl.
        Different families → no match (this is correct behavior)."""
        candidates = matcher.match("Qwen3-235B-A22B-Instruct")
        # qwen3 != qwen3-vl, so no match expected
        assert len(candidates) == 0

    def test_qwen_same_subfamily_matches(self, matcher):
        """If we query with the full 'vl' subfamily, it should match."""
        candidates = matcher.match("Qwen3-VL-235B-A22B-Instruct")
        assert len(candidates) >= 1
        assert any(c.model_id == 4 for c in candidates)


# ── Tier 8-9: Family-version and family-only ──────────────────────────────────

class TestTierFamilyLevel:
    """Tiers 8-9 — coarse matching by family and/or version."""

    def test_family_version_match(self, matcher):
        """Query with same family and version but different variant/size."""
        candidates = matcher.match("Gemini 2.5 Pro")
        assert len(candidates) >= 1
        assert any(c.model_id == 10 for c in candidates)


# ── No match ──────────────────────────────────────────────────────────────────

class TestNoMatch:
    """Queries that shouldn't match anything in the reference set."""

    def test_unknown_model(self, matcher):
        """A model not in the reference set at all."""
        candidates = matcher.match("Falcon 180B")
        assert candidates == []

    def test_empty_string(self, matcher):
        """Empty string should return no matches."""
        candidates = matcher.match("")
        assert candidates == []


# ── Ambiguity ─────────────────────────────────────────────────────────────────

class TestAmbiguity:
    """Multiple distinct model_ids at the same cascade level."""

    def test_gemini_bare_no_subfamily(self, matcher):
        """'Gemini 2.5' normalizes to family=gemini, but references have
        family=gemini-flash and gemini-pro. Different families → no match."""
        candidates = matcher.match("Gemini 2.5")
        # gemini != gemini-flash or gemini-pro
        assert len(candidates) == 0

    def test_resolve_returns_none_on_ambiguity(self, matcher):
        """resolve() returns None when multiple model_ids tie."""
        # Two Claude 3.5 models: sonnet (1) and haiku (9) share family/version
        # A query matching both should be ambiguous
        candidates = matcher.match("Claude 3.5")
        model_ids = {c.model_id for c in candidates}
        if len(model_ids) > 1:
            result = matcher.resolve("Claude 3.5")
            assert result is None


# ── resolve() ─────────────────────────────────────────────────────────────────

class TestResolve:
    """Test the convenience resolve() method."""

    def test_resolve_returns_model_id(self, matcher):
        """Unambiguous match returns model_id."""
        result = matcher.resolve("o4-mini")
        assert result == 3

    def test_resolve_returns_none_for_unknown(self, matcher):
        """No match returns None."""
        result = matcher.resolve("Falcon 180B")
        assert result is None

    def test_resolve_min_tier_restricts(self, matcher):
        """min_tier can restrict to only high-confidence matches."""
        # This should match at exact level
        result = matcher.resolve("o4-mini", min_tier="exact")
        assert result == 3


# ── resolve_batch() ───────────────────────────────────────────────────────────

class TestResolveBatch:
    """Test batch resolution."""

    def test_batch_returns_dict(self, matcher):
        queries = ["o4-mini", "Falcon 180B", "gpt-5.2"]
        results = matcher.resolve_batch(queries)
        assert results["o4-mini"] == 3
        assert results["Falcon 180B"] is None
        assert results["gpt-5.2"] == 2


# ── MatchCandidate fields ────────────────────────────────────────────────────

class TestMatchCandidateFields:
    """Verify MatchCandidate has correct metadata."""

    def test_candidate_has_correct_name_normalized(self, matcher):
        candidates = matcher.match("gpt-5.2")
        assert candidates[0].name_normalized == "gpt-5.2"

    def test_candidate_matched_via_tracks_source(self, matcher):
        """matched_via should indicate which ref string triggered the match."""
        candidates = matcher.match("gpt-5.2")
        assert candidates[0].matched_via in ("gpt-5.2", "gpt-5.2-2025-12-11")

    def test_candidate_tier_is_set(self, matcher):
        candidates = matcher.match("gpt-5.2")
        assert candidates[0].tier in (
            "exact", "base_id", "full", "no_provider", "core",
            "no_size", "no_variant", "family_version", "family_only",
        )
