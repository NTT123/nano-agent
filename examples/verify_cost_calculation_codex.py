"""Verify cost calculation against real Codex (OpenAI) API usage.

Makes real API calls to verify input/output token costs and reasoning tokens
using the OpenAI cost formula.

OpenAI billing:
  - input_tokens INCLUDES cached_tokens
  - Uncached input: (input_tokens - cached_tokens) at input rate
  - Cached input: cached_tokens at reduced cached_input rate
  - output_tokens INCLUDES reasoning_tokens
  - Non-reasoning output: (output_tokens - reasoning_tokens) at output rate
  - Reasoning: reasoning_tokens at reasoning rate

Also uses synthetic Usage objects to verify formulas for edge cases.

Note on caching: The Codex endpoint (chatgpt.com) requires the `instructions`
parameter, which does NOT participate in OpenAI prefix caching.  cached_tokens
will always be 0 from this endpoint.  To get real cached_tokens, use OpenAIAPI
with the standard API (api.openai.com), which passes the system prompt as a
developer message in the input array.  The cached_tokens formula is verified
via synthetic scenarios instead.

Scenarios:
  Part 1 — Real API calls (reasoning tokens)
    Turn 1: First call  → expect reasoning_tokens for algorithmic question
    Turn 2: Follow-up   → reasoning for optimization question
    Turn 3: Follow-up   → reasoning for test-writing question

  Part 2 — Synthetic scenarios
    Scenario A: With cached_tokens only
    Scenario B: With reasoning_tokens only
    Scenario C: Mixed cached + reasoning
    Scenario D: No caching, no reasoning (e.g., gpt-4.1)

Run:
    uv run python examples/verify_cost_calculation_codex.py
"""

import asyncio

from nano_agent import CodexAPI, DAG, Usage
from nano_agent.providers.cost import (
    CostBreakdown,
    calculate_cost,
    format_cost,
    get_pricing,
    get_provider_for_model,
)

SEPARATOR = "─" * 70

# A detailed system prompt.  Includes "think step by step" instruction so
# harder questions are more likely to produce reasoning_tokens.
LARGE_SYSTEM_PROMPT = """\
You are an expert software engineering assistant. Think step by step for
complex problems.  Be concise but thorough.

# Code Review Guidelines

When reviewing code, follow these principles:

## General Principles
- Readability over cleverness: Code should be easy to understand at a glance
- Single responsibility: Each function/class should do one thing well
- DRY (Don't Repeat Yourself): Extract common patterns into reusable components
- YAGNI (You Aren't Gonna Need It): Don't add functionality until it's needed
- KISS (Keep It Simple, Stupid): Choose the simplest solution that works

## Python-Specific Guidelines
- Follow PEP 8 for style conventions
- Use type hints for function signatures and class attributes
- Prefer dataclasses or named tuples over plain dictionaries for structured data
- Use context managers for resource management
- Prefer list/dict/set comprehensions over explicit loops where readable
- Use pathlib.Path instead of os.path for file operations
- Prefer f-strings over .format() or % formatting
- Use enum.Enum for fixed sets of constants
- Always handle exceptions at the appropriate level
- Use logging instead of print for production code

## Architecture Patterns
- Dependency injection over global state
- Composition over inheritance
- Program to interfaces, not implementations
- Fail fast: validate inputs early and raise clear errors
- Immutability by default: prefer frozen dataclasses and tuples
- Separate pure functions from side effects
- Use the repository pattern for data access
- Apply CQRS for complex read/write operations

## Testing Guidelines
- Write tests before fixing bugs (test-driven bug fixing)
- Test behavior, not implementation details
- Use parametrized tests for testing multiple inputs
- Mock external dependencies, not internal implementation
- Aim for high coverage but don't chase 100%
- Integration tests for critical paths
- Property-based testing for algorithmic code

## Security Considerations
- Never log sensitive data (passwords, tokens, PII)
- Validate and sanitize all user input
- Use parameterized queries for database operations
- Apply principle of least privilege
- Keep dependencies updated and audit regularly
- Use HTTPS everywhere, validate certificates
- Implement rate limiting for public APIs
- Use secure random number generation for tokens

## Performance Guidelines
- Profile before optimizing
- Use appropriate data structures (sets for membership testing, deques for FIFO)
- Batch database operations where possible
- Use async I/O for network-bound operations
- Cache expensive computations with appropriate invalidation
- Prefer generators for large data processing pipelines
- Use connection pooling for database and HTTP connections

## Documentation
- Write docstrings for public APIs
- Document "why" not "what" in code comments
- Keep README updated with setup instructions
- Use type hints as living documentation
- Include examples in docstrings for complex functions

## Git Workflow
- Write descriptive commit messages (imperative mood, ~50 chars subject)
- One logical change per commit
- Rebase feature branches before merging
- Use conventional commits format
- Never commit secrets or credentials
- Review your own PR before requesting review

## Error Handling
- Use domain-specific exception hierarchies
- Include context in error messages
- Log the full stack trace at appropriate level
- Provide actionable error messages to users
- Implement circuit breakers for external services
- Use retry with exponential backoff for transient failures
- Never swallow exceptions silently

## Concurrency
- Prefer asyncio for I/O-bound operations
- Use multiprocessing for CPU-bound operations
- Protect shared state with locks or use immutable data
- Avoid deadlocks by consistent lock ordering
- Use thread-safe data structures from queue module
- Implement graceful shutdown for long-running tasks
- Use semaphores to limit concurrent resource usage

Remember: Think step by step for complex problems, but keep answers concise.
"""


def manual_openai_cost(usage: Usage, model: str) -> CostBreakdown:
    """Calculate cost manually from published OpenAI pricing.

    Intentionally separate from calculate_cost() so we can compare.

    OpenAI billing:
      - input_tokens INCLUDES cached_tokens
      - output_tokens INCLUDES reasoning_tokens
    """
    pricing = get_pricing(model)
    if pricing is None:
        return CostBreakdown()

    # Input: split into uncached + cached
    uncached_input = max(0, usage.input_tokens - usage.cached_tokens)
    inp = uncached_input * pricing.input_per_mtok / 1_000_000
    inp += usage.cached_tokens * pricing.cached_input_per_mtok / 1_000_000

    # Output: split into non-reasoning + reasoning
    if pricing.reasoning_per_mtok and usage.reasoning_tokens:
        non_reasoning = max(0, usage.output_tokens - usage.reasoning_tokens)
        out = non_reasoning * pricing.output_per_mtok / 1_000_000
        reas = usage.reasoning_tokens * pricing.reasoning_per_mtok / 1_000_000
    else:
        out = usage.output_tokens * pricing.output_per_mtok / 1_000_000
        reas = 0.0

    return CostBreakdown(
        input_cost=inp,
        output_cost=out,
        reasoning_cost=reas,
        total_cost=inp + out + reas,
    )


def print_usage(usage: Usage) -> None:
    """Print raw token counts."""
    print("  Raw token counts:")
    print(f"    input_tokens:       {usage.input_tokens:>8,}")
    print(f"    output_tokens:      {usage.output_tokens:>8,}")
    print(f"    cached_tokens:      {usage.cached_tokens:>8,}")
    print(f"    reasoning_tokens:   {usage.reasoning_tokens:>8,}")
    print(f"    total_tokens:       {usage.total_tokens:>8,}")


def print_cost_comparison(
    computed: CostBreakdown, manual: CostBreakdown, model: str
) -> bool:
    """Print side-by-side comparison. Returns True if all match."""
    pricing = get_pricing(model)
    if pricing:
        print(f"  Pricing ({model}):")
        rates = [f"input=${pricing.input_per_mtok}", f"output=${pricing.output_per_mtok}"]
        if pricing.cached_input_per_mtok:
            rates.append(f"cached_input=${pricing.cached_input_per_mtok}")
        if pricing.reasoning_per_mtok:
            rates.append(f"reasoning=${pricing.reasoning_per_mtok}")
        print(f"    {', '.join(rates)}  (per MTok)")

    print()
    header = f"  {'Category':<20} {'calculate_cost()':>16} {'manual_calc':>16} {'match':>6}"
    print(header)
    print(f"  {'─' * 20} {'─' * 16} {'─' * 16} {'─' * 6}")

    rows = [
        ("input_cost", computed.input_cost, manual.input_cost),
        ("output_cost", computed.output_cost, manual.output_cost),
        ("reasoning_cost", computed.reasoning_cost, manual.reasoning_cost),
        ("total_cost", computed.total_cost, manual.total_cost),
    ]
    all_match = True
    for label, c_val, m_val in rows:
        if c_val == 0 and m_val == 0 and label != "total_cost":
            continue
        match = abs(c_val - m_val) < 1e-10
        if not match:
            all_match = False
        mark = "✓" if match else "✗"
        print(f"  {label:<20} {format_cost(c_val):>16} {format_cost(m_val):>16} {mark:>6}")

    print()
    if all_match:
        print("  ✓ All costs match!")
    else:
        print("  ✗ MISMATCH — investigate!")
    return all_match


def print_manual_formula(usage: Usage, model: str) -> None:
    """Print the exact arithmetic so user can verify by hand."""
    pricing = get_pricing(model)
    if not pricing:
        return
    print("  Manual arithmetic:")
    uncached = max(0, usage.input_tokens - usage.cached_tokens)
    if uncached:
        val = uncached * pricing.input_per_mtok / 1_000_000
        print(f"    uncached input: {uncached:,} × ${pricing.input_per_mtok}/MTok = {format_cost(val)}")
    if usage.cached_tokens:
        val = usage.cached_tokens * pricing.cached_input_per_mtok / 1_000_000
        print(f"    cached input:   {usage.cached_tokens:,} × ${pricing.cached_input_per_mtok}/MTok = {format_cost(val)}")
    if pricing.reasoning_per_mtok and usage.reasoning_tokens:
        non_reasoning = max(0, usage.output_tokens - usage.reasoning_tokens)
        if non_reasoning:
            val = non_reasoning * pricing.output_per_mtok / 1_000_000
            print(f"    output:         {non_reasoning:,} × ${pricing.output_per_mtok}/MTok = {format_cost(val)}")
        val = usage.reasoning_tokens * pricing.reasoning_per_mtok / 1_000_000
        print(f"    reasoning:      {usage.reasoning_tokens:,} × ${pricing.reasoning_per_mtok}/MTok = {format_cost(val)}")
    elif usage.output_tokens:
        val = usage.output_tokens * pricing.output_per_mtok / 1_000_000
        print(f"    output:         {usage.output_tokens:,} × ${pricing.output_per_mtok}/MTok = {format_cost(val)}")


def verify_turn(
    label: str,
    response_text: str,
    usage: Usage,
    model: str,
) -> tuple[bool, float]:
    """Print and verify a single turn. Returns (ok, cost)."""
    print(f"  Response: {response_text[:120]}{'...' if len(response_text) > 120 else ''}")
    print_usage(usage)

    print()
    print_manual_formula(usage, model)

    computed = calculate_cost(usage, model=model)
    manual = manual_openai_cost(usage, model)
    print()
    ok = print_cost_comparison(computed, manual, model)
    return ok, computed.total_cost


async def main() -> None:
    api = CodexAPI()
    model = api.model
    provider = get_provider_for_model(model)
    print(f"Model: {model}")
    print(f"Provider: {provider}")

    all_ok = True
    cumulative_cost = 0.0

    # =====================================================================
    # Part 1: Real API calls
    # =====================================================================
    print(f"\n{'═' * 70}")
    print("  PART 1: Real API Calls (with reasoning)")
    print(f"{'═' * 70}")
    print("  Note: Codex endpoint requires `instructions` (no cached_tokens).")
    print("  Harder questions to trigger reasoning_tokens.")

    # ── Turn 1 ──
    print(f"\n[Turn 1] First call (expect reasoning for algorithmic question)")
    print(SEPARATOR)
    dag = DAG().system(LARGE_SYSTEM_PROMPT)
    dag = dag.user(
        "Write a Python function to find the longest increasing subsequence "
        "in a list of integers. What is the time complexity? Keep it under "
        "20 lines."
    )
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    ok, cost = verify_turn("Turn 1", response.get_text(), response.usage, model)
    if not ok:
        all_ok = False
    cumulative_cost += cost

    # ── Turn 2 ──
    print(f"\n[Turn 2] Follow-up (expect reasoning for optimization question)")
    print(SEPARATOR)
    dag = dag.user(
        "Now optimize it to O(n log n) using binary search. Show the code "
        "and explain the key insight."
    )
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    ok, cost = verify_turn("Turn 2", response.get_text(), response.usage, model)
    if not ok:
        all_ok = False
    cumulative_cost += cost

    # ── Turn 3 ──
    print(f"\n[Turn 3] Follow-up (expect reasoning for test-writing question)")
    print(SEPARATOR)
    dag = dag.user(
        "Write comprehensive pytest tests for both implementations. Include "
        "edge cases: empty list, single element, all decreasing, all equal, "
        "and a large random input."
    )
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    ok, cost = verify_turn("Turn 3", response.get_text(), response.usage, model)
    if not ok:
        all_ok = False
    cumulative_cost += cost

    # =====================================================================
    # Part 2: Synthetic scenarios
    # =====================================================================
    print(f"\n{'═' * 70}")
    print("  PART 2: Synthetic Scenarios")
    print(f"{'═' * 70}")
    print("  (Using synthetic Usage to verify formulas)")

    # ── Scenario A: Cached tokens ──
    print(f"\n[Scenario A] Cached tokens: 10k input (4k cached), 2k output, no reasoning")
    print(SEPARATOR)
    usage_a = Usage(
        input_tokens=10_000,
        output_tokens=2_000,
        cached_tokens=4_000,
    )
    print_usage(usage_a)
    print()
    print_manual_formula(usage_a, model)

    computed = calculate_cost(usage_a, model=model)
    manual = manual_openai_cost(usage_a, model)
    print()
    if not print_cost_comparison(computed, manual, model):
        all_ok = False

    # ── Scenario B: Reasoning tokens ──
    print(f"\n[Scenario B] Reasoning tokens: 5k input, 3k output (1k reasoning)")
    print(SEPARATOR)
    usage_b = Usage(
        input_tokens=5_000,
        output_tokens=3_000,
        reasoning_tokens=1_000,
    )
    print_usage(usage_b)
    print()
    print_manual_formula(usage_b, model)

    computed = calculate_cost(usage_b, model=model)
    manual = manual_openai_cost(usage_b, model)
    print()
    if not print_cost_comparison(computed, manual, model):
        all_ok = False

    # ── Scenario C: Mixed cached + reasoning ──
    print(f"\n[Scenario C] Mixed: 20k input (8k cached), 5k output (2k reasoning)")
    print(SEPARATOR)
    usage_c = Usage(
        input_tokens=20_000,
        output_tokens=5_000,
        cached_tokens=8_000,
        reasoning_tokens=2_000,
    )
    print_usage(usage_c)
    print()
    print_manual_formula(usage_c, model)

    computed = calculate_cost(usage_c, model=model)
    manual = manual_openai_cost(usage_c, model)
    print()
    if not print_cost_comparison(computed, manual, model):
        all_ok = False

    # ── Scenario D: No caching, no reasoning (like gpt-4.1) ──
    gpt41_model = "gpt-4.1"
    print(f"\n[Scenario D] No caching, no reasoning ({gpt41_model}): 10k input, 3k output")
    print(SEPARATOR)
    usage_d = Usage(
        input_tokens=10_000,
        output_tokens=3_000,
    )
    print_usage(usage_d)
    print()
    print_manual_formula(usage_d, gpt41_model)

    computed = calculate_cost(usage_d, model=gpt41_model)
    manual = manual_openai_cost(usage_d, gpt41_model)
    print()
    if not print_cost_comparison(computed, manual, gpt41_model):
        all_ok = False

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'═' * 70}")
    print(f"  Cumulative real API cost (3 turns): {format_cost(cumulative_cost)}")
    if all_ok:
        print("  ✓ All 7 scenarios passed!")
    else:
        print("  ✗ Some scenarios failed — check above for details")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
