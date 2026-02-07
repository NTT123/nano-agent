"""Tests for cost tracking module."""

import pytest

from nano_agent.data_structures import Usage
from nano_agent.providers.cost import (
    CostBreakdown,
    ModelPricing,
    calculate_cost,
    format_cost,
    get_pricing,
    get_provider_for_model,
)


class TestGetProviderForModel:
    def test_anthropic_models(self) -> None:
        assert get_provider_for_model("claude-sonnet-4-20250514") == "anthropic"
        assert get_provider_for_model("claude-opus-4") == "anthropic"
        assert get_provider_for_model("claude-haiku-3.5") == "anthropic"

    def test_openai_models(self) -> None:
        assert get_provider_for_model("gpt-4.1") == "openai"
        assert get_provider_for_model("gpt-4o") == "openai"
        assert get_provider_for_model("o3") == "openai"
        assert get_provider_for_model("o4-mini") == "openai"
        assert get_provider_for_model("gpt-5.2-codex") == "openai"

    def test_gemini_models(self) -> None:
        assert get_provider_for_model("gemini-2.5-flash") == "gemini"
        assert get_provider_for_model("gemini-3-pro-preview") == "gemini"

    def test_fireworks_models(self) -> None:
        assert (
            get_provider_for_model("accounts/fireworks/models/kimi-k2p5") == "fireworks"
        )
        assert get_provider_for_model("kimi-k2p5") == "fireworks"

    def test_unknown_model(self) -> None:
        assert get_provider_for_model("unknown-model-123") is None


class TestGetPricing:
    def test_exact_match(self) -> None:
        pricing = get_pricing("claude-sonnet-4", provider="anthropic")
        assert pricing is not None
        assert pricing.input_per_mtok == 3.0
        assert pricing.output_per_mtok == 15.0

    def test_prefix_match(self) -> None:
        pricing = get_pricing("claude-sonnet-4-20250514")
        assert pricing is not None
        assert pricing.input_per_mtok == 3.0

    def test_gemini_prefix_match(self) -> None:
        pricing = get_pricing("gemini-3-pro-preview")
        assert pricing is not None
        assert pricing.input_per_mtok == 1.25

    def test_fireworks_suffix_match(self) -> None:
        pricing = get_pricing("accounts/fireworks/models/kimi-k2p5")
        assert pricing is not None
        assert pricing.input_per_mtok == 0.40

    def test_unknown_model_returns_none(self) -> None:
        pricing = get_pricing("totally-unknown-model")
        assert pricing is None

    def test_unknown_provider_returns_none(self) -> None:
        pricing = get_pricing("some-model", provider="nonexistent")
        assert pricing is None


class TestCalculateCostAnthropic:
    def test_basic_input_output(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        result = calculate_cost(usage, "claude-sonnet-4")
        assert result.input_cost == pytest.approx(0.003)
        assert result.output_cost == pytest.approx(0.0075)
        assert result.total_cost == pytest.approx(0.0105)

    def test_cache_write_and_read(self) -> None:
        usage = Usage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=2000,
            cache_read_input_tokens=3000,
        )
        result = calculate_cost(usage, "claude-sonnet-4")
        # cache_write: 2000 * 3.75 / 1M = 0.0075
        assert result.cache_write_cost == pytest.approx(0.0075)
        # cache_read: 3000 * 0.30 / 1M = 0.0009
        assert result.cache_read_cost == pytest.approx(0.0009)
        expected_total = 0.003 + 0.0075 + 0.0075 + 0.0009
        assert result.total_cost == pytest.approx(expected_total)


class TestCalculateCostOpenAI:
    def test_with_cached_tokens(self) -> None:
        usage = Usage(
            input_tokens=10000,
            output_tokens=5000,
            cached_tokens=4000,
        )
        result = calculate_cost(usage, "gpt-4.1")
        # Uncached: (10000 - 4000) * 2.0 / 1M = 0.012
        # Cached: 4000 * 0.50 / 1M = 0.002
        assert result.input_cost == pytest.approx(0.014)
        # Output: 5000 * 8.0 / 1M = 0.040
        assert result.output_cost == pytest.approx(0.040)
        assert result.total_cost == pytest.approx(0.054)

    def test_with_reasoning_tokens(self) -> None:
        usage = Usage(
            input_tokens=5000,
            output_tokens=2000,
            reasoning_tokens=1000,
        )
        result = calculate_cost(usage, "o3")
        # OpenAI output_tokens INCLUDES reasoning_tokens
        # Non-reasoning output: (2000 - 1000) * 8.0 / 1M = 0.008
        assert result.output_cost == pytest.approx(0.008)
        # Reasoning: 1000 * 8.0 / 1M = 0.008
        assert result.reasoning_cost == pytest.approx(0.008)
        # Input: 5000 * 2.0 / 1M = 0.010
        assert result.input_cost == pytest.approx(0.010)
        # Total: 0.010 + 0.008 + 0.008 = 0.026
        assert result.total_cost == pytest.approx(0.026)


class TestCalculateCostGemini:
    def test_with_thinking_tokens(self) -> None:
        usage = Usage(
            input_tokens=5000,
            output_tokens=2000,
            reasoning_tokens=3000,
        )
        result = calculate_cost(usage, "gemini-2.5-flash")
        # Input: 5000 * 0.15 / 1M = 0.00075
        assert result.input_cost == pytest.approx(0.00075)
        # Output: 2000 * 0.60 / 1M = 0.0012
        assert result.output_cost == pytest.approx(0.0012)
        # Reasoning: 3000 * 3.50 / 1M = 0.0105
        assert result.reasoning_cost == pytest.approx(0.0105)
        assert result.total_cost == pytest.approx(0.01245)


class TestCalculateCostUnknown:
    def test_unknown_model_returns_zero(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        result = calculate_cost(usage, "unknown-model-xyz")
        assert result.total_cost == 0.0
        assert result.input_cost == 0.0
        assert result.output_cost == 0.0


class TestFormatCost:
    def test_small_cost(self) -> None:
        assert format_cost(0.0001) == "$0.0001"

    def test_medium_cost(self) -> None:
        assert format_cost(0.042) == "$0.042"

    def test_large_cost(self) -> None:
        assert format_cost(1.23) == "$1.23"

    def test_zero_cost(self) -> None:
        assert format_cost(0.0) == "$0.0000"
