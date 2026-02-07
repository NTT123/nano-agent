"""Cost tracking for LLM API usage.

This module provides pricing tables and cost calculation functions
for various LLM providers (Anthropic, OpenAI, Gemini, Fireworks).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data_structures import Usage

__all__ = [
    "ModelPricing",
    "CostBreakdown",
    "get_provider_for_model",
    "get_pricing",
    "calculate_cost",
    "format_cost",
]


@dataclass(frozen=True)
class ModelPricing:
    """Pricing per million tokens for a model."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0
    cache_read_per_mtok: float = 0.0
    cached_input_per_mtok: float = 0.0
    reasoning_per_mtok: float = 0.0


@dataclass(frozen=True)
class CostBreakdown:
    """Breakdown of costs by category."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_write_cost: float = 0.0
    cache_read_cost: float = 0.0
    reasoning_cost: float = 0.0
    total_cost: float = 0.0


# =============================================================================
# Pricing Tables (per million tokens)
# =============================================================================

ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4": ModelPricing(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_write_per_mtok=3.75,  # 1.25x input
        cache_read_per_mtok=0.30,  # 0.1x input
    ),
    "claude-opus-4": ModelPricing(
        input_per_mtok=15.0,
        output_per_mtok=75.0,
        cache_write_per_mtok=18.75,  # 1.25x input
        cache_read_per_mtok=1.50,  # 0.1x input
    ),
    "claude-haiku-3.5": ModelPricing(
        input_per_mtok=0.80,
        output_per_mtok=4.0,
        cache_write_per_mtok=1.0,  # 1.25x input
        cache_read_per_mtok=0.08,  # 0.1x input
    ),
}

OPENAI_PRICING: dict[str, ModelPricing] = {
    "gpt-4.1": ModelPricing(
        input_per_mtok=2.0,
        output_per_mtok=8.0,
        cached_input_per_mtok=0.50,
    ),
    "gpt-4o": ModelPricing(
        input_per_mtok=2.50,
        output_per_mtok=10.0,
        cached_input_per_mtok=1.25,
    ),
    "o3": ModelPricing(
        input_per_mtok=2.0,
        output_per_mtok=8.0,
        cached_input_per_mtok=0.50,
        reasoning_per_mtok=8.0,
    ),
    "o4-mini": ModelPricing(
        input_per_mtok=1.10,
        output_per_mtok=4.40,
        cached_input_per_mtok=0.275,
        reasoning_per_mtok=4.40,
    ),
    "gpt-5.2-codex": ModelPricing(
        input_per_mtok=2.0,
        output_per_mtok=8.0,
        cached_input_per_mtok=0.50,
        reasoning_per_mtok=8.0,
    ),
}

GEMINI_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-flash": ModelPricing(
        input_per_mtok=0.15,
        output_per_mtok=0.60,
        reasoning_per_mtok=3.50,
    ),
    "gemini-2.5-pro": ModelPricing(
        input_per_mtok=1.25,
        output_per_mtok=10.0,
        reasoning_per_mtok=10.0,
    ),
    "gemini-3-flash": ModelPricing(
        input_per_mtok=0.15,
        output_per_mtok=0.60,
        reasoning_per_mtok=3.50,
    ),
    "gemini-3-pro": ModelPricing(
        input_per_mtok=1.25,
        output_per_mtok=10.0,
        reasoning_per_mtok=10.0,
    ),
}

FIREWORKS_PRICING: dict[str, ModelPricing] = {
    "kimi-k2p5": ModelPricing(
        input_per_mtok=0.40,
        output_per_mtok=1.60,
    ),
}

# Combined table mapping provider -> pricing dict
_ALL_PRICING: dict[str, dict[str, ModelPricing]] = {
    "anthropic": ANTHROPIC_PRICING,
    "openai": OPENAI_PRICING,
    "gemini": GEMINI_PRICING,
    "fireworks": FIREWORKS_PRICING,
}


def get_provider_for_model(model: str) -> str | None:
    """Detect provider from model name prefix.

    Args:
        model: Model name (e.g., "claude-sonnet-4-20250514")

    Returns:
        Provider string or None if unknown
    """
    lower = model.lower()
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith(("gpt", "o3", "o4", "o1")):
        return "openai"
    if lower.startswith("gemini"):
        return "gemini"
    if "fireworks" in lower or "kimi" in lower:
        return "fireworks"
    return None


def get_pricing(model: str, provider: str | None = None) -> ModelPricing | None:
    """Look up pricing for a model, using prefix matching for versioned names.

    Args:
        model: Model name (e.g., "claude-sonnet-4-20250514")
        provider: Provider name (auto-detected if None)

    Returns:
        ModelPricing or None if not found
    """
    if provider is None:
        provider = get_provider_for_model(model)
    if provider is None:
        return None

    pricing_table = _ALL_PRICING.get(provider)
    if pricing_table is None:
        return None

    lower = model.lower()

    # Exact match first
    if lower in pricing_table:
        return pricing_table[lower]

    # Prefix match (for versioned model names like "claude-sonnet-4-20250514")
    # Also checks if any key is a suffix of the model name (for path-prefixed
    # models like "accounts/fireworks/models/kimi-k2p5" matching "kimi-k2p5")
    best_match: str | None = None
    best_len = 0
    for key in pricing_table:
        if (lower.startswith(key) or lower.endswith(key)) and len(key) > best_len:
            best_match = key
            best_len = len(key)

    if best_match is not None:
        return pricing_table[best_match]

    return None


def calculate_cost(
    usage: Usage, model: str, provider: str | None = None
) -> CostBreakdown:
    """Calculate cost breakdown for a given usage and model.

    Provider-aware cost formulas:
    - Anthropic: input_tokens excludes cached; cache_write at 1.25x; cache_read at 0.1x
    - OpenAI: input_tokens includes cached; subtract cached_tokens for uncached portion
    - Gemini: reasoning_tokens billed at separate (higher) rate
    - Generic fallback: simple input/output pricing

    Args:
        usage: Token usage from API response
        model: Model name
        provider: Provider name (auto-detected if None)

    Returns:
        CostBreakdown with per-category costs
    """
    if provider is None:
        provider = get_provider_for_model(model)

    pricing = get_pricing(model, provider)
    if pricing is None:
        return CostBreakdown()

    input_cost = 0.0
    output_cost = 0.0
    cache_write_cost = 0.0
    cache_read_cost = 0.0
    reasoning_cost = 0.0

    if provider == "anthropic":
        # Anthropic: input_tokens does NOT include cached tokens
        input_cost = usage.input_tokens * pricing.input_per_mtok / 1_000_000
        output_cost = usage.output_tokens * pricing.output_per_mtok / 1_000_000
        cache_write_cost = (
            usage.cache_creation_input_tokens * pricing.cache_write_per_mtok / 1_000_000
        )
        cache_read_cost = (
            usage.cache_read_input_tokens * pricing.cache_read_per_mtok / 1_000_000
        )
    elif provider == "openai":
        # OpenAI: input_tokens INCLUDES cached tokens, subtract for uncached
        uncached_input = max(0, usage.input_tokens - usage.cached_tokens)
        input_cost = uncached_input * pricing.input_per_mtok / 1_000_000
        input_cost += usage.cached_tokens * pricing.cached_input_per_mtok / 1_000_000
        # OpenAI: output_tokens INCLUDES reasoning tokens, subtract for non-reasoning
        if pricing.reasoning_per_mtok and usage.reasoning_tokens:
            non_reasoning_output = max(0, usage.output_tokens - usage.reasoning_tokens)
            output_cost = non_reasoning_output * pricing.output_per_mtok / 1_000_000
            reasoning_cost = (
                usage.reasoning_tokens * pricing.reasoning_per_mtok / 1_000_000
            )
        else:
            output_cost = usage.output_tokens * pricing.output_per_mtok / 1_000_000
    elif provider == "gemini":
        # Gemini: reasoning_tokens billed at separate (higher) rate
        input_cost = usage.input_tokens * pricing.input_per_mtok / 1_000_000
        output_cost = usage.output_tokens * pricing.output_per_mtok / 1_000_000
        if pricing.reasoning_per_mtok and usage.reasoning_tokens:
            reasoning_cost = (
                usage.reasoning_tokens * pricing.reasoning_per_mtok / 1_000_000
            )
    else:
        # Generic fallback
        input_cost = usage.input_tokens * pricing.input_per_mtok / 1_000_000
        output_cost = usage.output_tokens * pricing.output_per_mtok / 1_000_000

    total_cost = (
        input_cost + output_cost + cache_write_cost + cache_read_cost + reasoning_cost
    )

    return CostBreakdown(
        input_cost=input_cost,
        output_cost=output_cost,
        cache_write_cost=cache_write_cost,
        cache_read_cost=cache_read_cost,
        reasoning_cost=reasoning_cost,
        total_cost=total_cost,
    )


def format_cost(cost: float) -> str:
    """Format a cost value as a dollar string.

    Args:
        cost: Cost in dollars

    Returns:
        Formatted string like "$0.042" or "$1.23"
    """
    if cost < 0.001:
        return f"${cost:.4f}"
    if cost < 0.1:
        return f"${cost:.3f}"
    return f"${cost:.2f}"
