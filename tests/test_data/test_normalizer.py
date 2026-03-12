"""
Tests for normalize() — the model name normalizer.

Organized by:
  1. Roundtrip tests: slug inputs should produce canonical_id == input
  2. Human-readable equivalence: natural language -> expected slug
  3. Bug regression tests: all 6 original bugs
  4. Variant handling
  5. Inference/reasoning effort levels
  6. Field extraction (provider, family, version, size)
"""

import pytest

from llm_compass.data.normalizer import normalize


# ── 1. Roundtrip: slug in = slug out ────────────────────────────────────────

class TestRoundtrip:
    """Slug-format reference names must produce canonical_id == input."""

    @pytest.mark.parametrize("slug", [
        "claude-3.5-sonnet",
        "claude-3.7-sonnet-thinking",
        "claude-4.5-opus-thinking",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5-codex",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "deepseek-v3",
        "deepseek-v3.1",
        "deepseek-r1",
        "deepseek-v3.2-exp-thinking",
        "llama-3.1-70b-instruct",
        "llama-3.3-70b",
        "qwen3-235b-a22b-thinking",
        "qwen3-vl-235b-a22b-thinking",
        "minimax-m2",
        "minimax-m2.1",
        "mimo-v2-flash",
        "mistral-large-3",
        "mixtral-8x22b-instruct",
        "phi-4-mini-reasoning",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
        "sonar-reasoning-pro",
    ])
    def test_slug_roundtrip(self, slug):
        result = normalize(slug)
        assert result["canonical_id"] == slug, (
            f"Expected canonical_id={slug!r}, got {result['canonical_id']!r}"
        )


# ── 2. Human-readable equivalence ───────────────────────────────────────────

class TestHumanReadable:
    """Natural language model names should produce correct slugs."""

    @pytest.mark.parametrize("raw, expected_canonical", [
        ("Claude 3.5 Sonnet", "claude-3.5-sonnet"),
        ("Gemini 2.5 Flash", "gemini-2.5-flash"),
        ("Gemini 2.5 Flash Lite", "gemini-2.5-flash-lite"),
        ("Gemini 2.5 Pro", "gemini-2.5-pro"),
        ("DeepSeek V3.1", "deepseek-v3.1"),
        ("DeepSeek V3", "deepseek-v3"),
        ("mimo V2 Flash", "mimo-v2-flash"),
        ("MiniMax M2", "minimax-m2"),
        ("GPT 5.2", "gpt-5.2"),
        ("GPT-4o Mini", "gpt-4o-mini"),
    ])
    def test_human_to_slug(self, raw, expected_canonical):
        result = normalize(raw)
        assert result["canonical_id"] == expected_canonical, (
            f"normalize({raw!r}): expected canonical_id={expected_canonical!r}, "
            f"got {result['canonical_id']!r}"
        )


# ── 3. Bug regression tests ─────────────────────────────────────────────────

class TestBugRegression:
    """Regression tests for the 6 original bugs."""

    def test_meta_llama_equivalence(self):
        """Bug 1: Meta Llama / Llama variants should produce same base_id."""
        r1 = normalize("Meta Llama 3 70b")
        r2 = normalize("Llama 3 70b")
        r3 = normalize("Meta Llama3 70b")
        # All should have same base_id
        assert r1["base_id"] == r2["base_id"], (
            f"'Meta Llama 3 70b' base_id={r1['base_id']!r} != "
            f"'Llama 3 70b' base_id={r2['base_id']!r}"
        )
        assert r2["base_id"] == r3["base_id"], (
            f"'Llama 3 70b' base_id={r2['base_id']!r} != "
            f"'Meta Llama3 70b' base_id={r3['base_id']!r}"
        )

    def test_no_standard_in_ids(self):
        """Bug 2: 'standard' should never appear in canonical_id or base_id."""
        for raw in ["GPT-5.2", "Claude 3.5 Sonnet", "Gemini 2.5 Flash", "o4-mini"]:
            result = normalize(raw)
            assert "standard" not in result["canonical_id"], (
                f"{raw!r}: 'standard' found in canonical_id={result['canonical_id']!r}"
            )
            assert "standard" not in result["base_id"], (
                f"{raw!r}: 'standard' found in base_id={result['base_id']!r}"
            )

    def test_v_preserved_in_mimo(self):
        """Bug 3: 'V' should be preserved in 'mimo V2 Flash'."""
        result = normalize("mimo V2 Flash")
        assert "v2" in result["canonical_id"], (
            f"'v2' not found in canonical_id={result['canonical_id']!r}"
        )

    def test_v_preserved_in_deepseek(self):
        """Bug 3: 'V' should be preserved in 'DeepSeekV3.1'."""
        result = normalize("DeepSeekV3.1")
        assert "v3.1" in result["canonical_id"], (
            f"'v3.1' not found in canonical_id={result['canonical_id']!r}"
        )

    def test_lite_preserved(self):
        """Bug 4: 'Lite' should be preserved in 'Gemini 2.5 Flash Lite'."""
        result = normalize("Gemini 2.5 Flash Lite")
        assert "lite" in result["canonical_id"], (
            f"'lite' not found in canonical_id={result['canonical_id']!r}"
        )

    def test_minimax_not_split(self):
        """Bug 5: MiniMax should not be split by size regex."""
        result = normalize("MiniMax M1 80k")
        assert result["canonical_id"].startswith("minimax"), (
            f"Expected canonical_id to start with 'minimax', got {result['canonical_id']!r}"
        )
        assert "mini" != result.get("size"), (
            f"'mini' should not be detected as size"
        )

    def test_minimax_case_consistency(self):
        """Bug 6: Minimax-M2, Minimax-m2, MiniMax M2 should all be same."""
        r1 = normalize("Minimax-M2")
        r2 = normalize("Minimax-m2")
        r3 = normalize("MiniMax M2")
        assert r1["canonical_id"] == r2["canonical_id"] == r3["canonical_id"], (
            f"Inconsistent canonical_ids: {r1['canonical_id']!r}, "
            f"{r2['canonical_id']!r}, {r3['canonical_id']!r}"
        )


# ── 4. Variant handling ─────────────────────────────────────────────────────

class TestVariant:
    """Variant detection and ID construction."""

    def test_thinking_variant(self):
        result = normalize("Claude 3.7 Sonnet (Thinking)")
        assert result["variant"] == "thinking"
        assert result["canonical_id"] == "claude-3.7-sonnet-thinking"

    def test_no_variant_means_empty(self):
        """No explicit variant -> variant='' (not 'standard')."""
        result = normalize("GPT-5.2")
        assert result["variant"] == ""

    def test_instruct_variant_in_slug(self):
        result = normalize("llama-3.1-70b-instruct")
        assert result["variant"] == "instruct"

    def test_codex_variant(self):
        result = normalize("gpt-5-codex")
        assert result["variant"] == "codex"

    def test_preview_variant(self):
        result = normalize("gemini-3-flash-preview")
        assert result["variant"] == "preview"

    def test_exp_variant(self):
        result = normalize("deepseek-v3.2-exp-thinking")
        # "exp" is part of slug, "thinking" is the variant
        assert result["variant"] == "thinking"

    def test_non_thinking_variant(self):
        result = normalize("deepseek-v3.1-non-thinking")
        assert result["variant"] == "non-thinking"

    def test_fast_reasoning_compound_variant(self):
        result = normalize("grok-4-fast-reasoning")
        assert result["variant"] == "fast-reasoning"

    def test_fast_non_reasoning_compound_variant(self):
        result = normalize("grok-4-fast-non-reasoning")
        assert result["variant"] == "fast-non-reasoning"

    def test_o_series_no_forced_variant(self):
        """O-series models should not have forced reasoning variant."""
        result = normalize("o4-mini")
        assert result["variant"] == ""
        assert result["provider"] == "openai"


# ── 5. Inference / reasoning effort ─────────────────────────────────────────

class TestReasoningEffort:
    """Reasoning effort level extraction."""

    def test_high_reasoning(self):
        result = normalize("GPT-5.2 (High Reasoning)")
        assert result["variant"] == "reasoning"
        assert result["reasoning_effort"] == "high"

    def test_low_reasoning(self):
        result = normalize("GPT-5.2 (Low Reasoning)")
        assert result["variant"] == "reasoning"
        assert result["reasoning_effort"] == "low"

    def test_xhigh_reasoning(self):
        result = normalize("Model X (xhigh reasoning)")
        assert result["reasoning_effort"] == "xhigh"

    def test_default_reasoning_effort(self):
        """'(Reasoning)' with no level -> effort='medium'."""
        result = normalize("GPT-5.2 (Reasoning)")
        assert result["variant"] == "reasoning"
        assert result["reasoning_effort"] == "medium"

    def test_no_reasoning_effort_when_no_variant(self):
        result = normalize("GPT-5.2")
        assert result["reasoning_effort"] is None

    def test_reasoning_not_in_canonical_id(self):
        """Effort level should NOT appear in canonical_id."""
        result = normalize("GPT-5.2 (High Reasoning)")
        assert "high" not in result["canonical_id"]

    def test_reasoning_in_canonical_id_as_variant(self):
        """'reasoning' variant SHOULD appear in canonical_id."""
        result = normalize("GPT-5.2 (High Reasoning)")
        assert result["canonical_id"] == "gpt-5.2-reasoning"


# ── 6. Field extraction ─────────────────────────────────────────────────────

class TestFieldExtraction:
    """Verify individual field extraction."""

    def test_provider_anthropic(self):
        assert normalize("claude-3.5-sonnet")["provider"] == "anthropic"

    def test_provider_openai(self):
        assert normalize("gpt-5.2")["provider"] == "openai"

    def test_provider_openai_o_series(self):
        assert normalize("o3-mini")["provider"] == "openai"

    def test_provider_google(self):
        assert normalize("gemini-2.5-flash")["provider"] == "google"

    def test_provider_meta(self):
        assert normalize("llama-3.3-70b")["provider"] == "meta"

    def test_provider_deepseek(self):
        assert normalize("deepseek-v3")["provider"] == "deepseek"

    def test_provider_minimax(self):
        assert normalize("minimax-m2")["provider"] == "minimax"

    def test_family_claude_sonnet(self):
        result = normalize("claude-3.5-sonnet")
        assert "sonnet" in result["family"]
        assert "claude" in result["family"]

    def test_family_gemini_flash(self):
        result = normalize("gemini-2.5-flash")
        assert "flash" in result["family"]
        assert "gemini" in result["family"]

    def test_family_gemini_flash_lite(self):
        result = normalize("gemini-2.5-flash-lite")
        assert "lite" in result["family"]

    def test_size_70b(self):
        result = normalize("llama-3.3-70b")
        assert result["size"] == "70b"

    def test_size_moe(self):
        result = normalize("qwen3-235b-a22b-thinking")
        assert result["size"] is not None
        assert "235b" in result["size"]
        assert "a22b" in result["size"]

    def test_version_dotted(self):
        result = normalize("claude-3.5-sonnet")
        assert result["version"] == "3.5"

    def test_version_with_v_prefix(self):
        result = normalize("deepseek-v3.1")
        assert result["version"] == "v3.1"

    def test_date_extraction(self):
        result = normalize("claude-3-5-sonnet-20241022")
        assert result["date"] is not None

    def test_empty_input(self):
        result = normalize("")
        assert result["canonical_id"] == ""
        assert result["provider"] == "unknown"
