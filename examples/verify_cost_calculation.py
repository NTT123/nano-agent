"""Verify cost calculation against real Claude Code API usage.

Makes real API calls to verify input/output token costs AND prompt caching.
Uses a large system prompt (>4096 tokens) to meet the caching minimum for
Opus 4.6, so Turn 1 triggers cache_creation and Turns 2-3 trigger cache_read.

Also uses synthetic Usage objects to verify formulas for edge cases.

Scenarios:
  Part 1 — Real API calls (with prompt caching)
    Turn 1: First call  → cache_creation (system prompt written to cache)
    Turn 2: Follow-up   → cache_read (system prompt + Turn 1 read from cache)
    Turn 3: Follow-up   → cache_read (system prompt + Turns 1-2 read from cache)

  Part 2 — Synthetic scenarios (cache_write + cache_read)
    Scenario A: Large cache write
    Scenario B: Cache read
    Scenario C: Mixed cache_write + cache_read + normal input

Run:
    uv run python examples/verify_cost_calculation.py
"""

import asyncio

from nano_agent import DAG, ClaudeCodeAPI, Usage
from nano_agent.cost import (
    CostBreakdown,
    calculate_cost,
    format_cost,
    get_pricing,
    get_provider_for_model,
)

SEPARATOR = "─" * 70

# A large system prompt (~5000 tokens) to exceed the 4096-token caching
# minimum for Opus 4.6.  This is a realistic coding assistant prompt.
LARGE_SYSTEM_PROMPT = """\
You are an expert software engineering assistant. Be concise. Reply in 1-2 sentences.

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

## API Design
- Use consistent naming conventions
- Version your APIs from the start
- Return appropriate HTTP status codes
- Implement pagination for list endpoints
- Use standard error response format
- Document with OpenAPI/Swagger
- Implement idempotency for write operations
- Use ETags for caching

## Database Design
- Normalize to 3NF, denormalize for performance
- Use migrations for schema changes
- Index columns used in WHERE and JOIN clauses
- Use transactions for multi-step operations
- Implement soft deletes for audit trails
- Use UUIDs for public-facing IDs
- Partition large tables by time or category

## Monitoring and Observability
- Structured logging with correlation IDs
- Track key business metrics
- Set up alerts for error rate spikes
- Distributed tracing for microservices
- Health check endpoints for all services
- Monitor resource usage (CPU, memory, disk, network)

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

## Deployment
- Use environment variables for configuration
- Implement health checks and readiness probes
- Use rolling deployments for zero downtime
- Implement feature flags for gradual rollouts
- Containerize applications with minimal base images
- Use multi-stage builds to reduce image size
- Implement graceful shutdown handling
- Log to stdout/stderr, let the platform handle aggregation

## Code Organization
- Group by feature, not by type
- Keep modules small and focused
- Use __init__.py to define public API
- Separate configuration from code
- Use abstract base classes for interfaces
- Keep import graphs acyclic
- Put constants near their usage, not in global files

Remember: Always be concise. Reply in 1-2 sentences maximum.
"""


def manual_anthropic_cost(usage: Usage, model: str) -> CostBreakdown:
    """Calculate cost manually from published Anthropic pricing.

    Intentionally separate from calculate_cost() so we can compare.

    Anthropic billing:
      - input_tokens does NOT include cached tokens
      - cache_creation at 1.25x input rate
      - cache_read at 0.1x input rate
    """
    pricing = get_pricing(model)
    if pricing is None:
        return CostBreakdown()

    inp = usage.input_tokens * pricing.input_per_mtok / 1_000_000
    out = usage.output_tokens * pricing.output_per_mtok / 1_000_000
    cw = usage.cache_creation_input_tokens * pricing.cache_write_per_mtok / 1_000_000
    cr = usage.cache_read_input_tokens * pricing.cache_read_per_mtok / 1_000_000

    return CostBreakdown(
        input_cost=inp,
        output_cost=out,
        cache_write_cost=cw,
        cache_read_cost=cr,
        total_cost=inp + out + cw + cr,
    )


def print_usage(usage: Usage) -> None:
    """Print raw token counts."""
    print("  Raw token counts:")
    print(f"    input_tokens:                {usage.input_tokens:>8,}")
    print(f"    output_tokens:               {usage.output_tokens:>8,}")
    print(f"    cache_creation_input_tokens: {usage.cache_creation_input_tokens:>8,}")
    print(f"    cache_read_input_tokens:     {usage.cache_read_input_tokens:>8,}")


def print_cost_comparison(
    computed: CostBreakdown, manual: CostBreakdown, model: str
) -> bool:
    """Print side-by-side comparison. Returns True if all match."""
    pricing = get_pricing(model)
    if pricing:
        print(f"  Pricing ({model}):")
        rates = [f"input=${pricing.input_per_mtok}", f"output=${pricing.output_per_mtok}"]
        if pricing.cache_write_per_mtok:
            rates.append(f"cache_write=${pricing.cache_write_per_mtok}")
        if pricing.cache_read_per_mtok:
            rates.append(f"cache_read=${pricing.cache_read_per_mtok}")
        print(f"    {', '.join(rates)}  (per MTok)")

    print()
    header = f"  {'Category':<20} {'calculate_cost()':>16} {'manual_calc':>16} {'match':>6}"
    print(header)
    print(f"  {'─' * 20} {'─' * 16} {'─' * 16} {'─' * 6}")

    rows = [
        ("input_cost", computed.input_cost, manual.input_cost),
        ("output_cost", computed.output_cost, manual.output_cost),
        ("cache_write_cost", computed.cache_write_cost, manual.cache_write_cost),
        ("cache_read_cost", computed.cache_read_cost, manual.cache_read_cost),
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
    if usage.input_tokens:
        val = usage.input_tokens * pricing.input_per_mtok / 1_000_000
        print(f"    input:       {usage.input_tokens:,} × ${pricing.input_per_mtok}/MTok = {format_cost(val)}")
    if usage.output_tokens:
        val = usage.output_tokens * pricing.output_per_mtok / 1_000_000
        print(f"    output:      {usage.output_tokens:,} × ${pricing.output_per_mtok}/MTok = {format_cost(val)}")
    if usage.cache_creation_input_tokens:
        val = usage.cache_creation_input_tokens * pricing.cache_write_per_mtok / 1_000_000
        print(f"    cache_write: {usage.cache_creation_input_tokens:,} × ${pricing.cache_write_per_mtok}/MTok = {format_cost(val)}")
    if usage.cache_read_input_tokens:
        val = usage.cache_read_input_tokens * pricing.cache_read_per_mtok / 1_000_000
        print(f"    cache_read:  {usage.cache_read_input_tokens:,} × ${pricing.cache_read_per_mtok}/MTok = {format_cost(val)}")


def verify_turn(
    label: str,
    response_text: str,
    usage: Usage,
    model: str,
    expect_cache_write: bool = False,
    expect_cache_read: bool = False,
) -> tuple[bool, float]:
    """Print and verify a single turn. Returns (ok, cost)."""
    print(f"  Response: {response_text[:120]}{'...' if len(response_text) > 120 else ''}")
    print_usage(usage)

    # Check caching expectations
    if expect_cache_write and usage.cache_creation_input_tokens == 0:
        print("  ⚠ Expected cache_creation_input_tokens > 0 (prompt may be below 4096-token minimum)")
    if expect_cache_read and usage.cache_read_input_tokens == 0:
        print("  ⚠ Expected cache_read_input_tokens > 0 (cache may have expired or not been created)")

    print()
    print_manual_formula(usage, model)

    computed = calculate_cost(usage, model=model)
    manual = manual_anthropic_cost(usage, model)
    print()
    ok = print_cost_comparison(computed, manual, model)
    return ok, computed.total_cost


async def main() -> None:
    api = ClaudeCodeAPI()
    model = api.model
    provider = get_provider_for_model(model)
    print(f"Model: {model}")
    print(f"Provider: {provider}")

    all_ok = True
    cumulative_cost = 0.0

    # =====================================================================
    # Part 1: Real API calls with prompt caching
    # =====================================================================
    print(f"\n{'═' * 70}")
    print("  PART 1: Real API Calls (with prompt caching)")
    print(f"{'═' * 70}")
    print(f"  System prompt is ~5000 tokens to exceed the 4096-token caching minimum.")

    # ── Turn 1: should trigger cache_creation ──
    print(f"\n[Turn 1] First call (expect cache_creation for system prompt)")
    print(SEPARATOR)
    dag = DAG().system(LARGE_SYSTEM_PROMPT)
    dag = dag.user("What is the capital of France?")
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    ok, cost = verify_turn(
        "Turn 1", response.get_text(), response.usage, model,
        expect_cache_write=True,
    )
    if not ok:
        all_ok = False
    cumulative_cost += cost

    # ── Turn 2: should trigger cache_read ──
    print(f"\n[Turn 2] Follow-up (expect cache_read for system prompt + Turn 1)")
    print(SEPARATOR)
    dag = dag.user("And what is the population of that city?")
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    ok, cost = verify_turn(
        "Turn 2", response.get_text(), response.usage, model,
        expect_cache_read=True,
    )
    if not ok:
        all_ok = False
    cumulative_cost += cost

    # ── Turn 3: should trigger cache_read ──
    print(f"\n[Turn 3] Another follow-up (expect cache_read for system + Turns 1-2)")
    print(SEPARATOR)
    dag = dag.user(
        "Compare Paris vs London in detail: population, area, GDP, landmarks, "
        "and public transport. Use 3-4 sentences."
    )
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    ok, cost = verify_turn(
        "Turn 3", response.get_text(), response.usage, model,
        expect_cache_read=True,
    )
    if not ok:
        all_ok = False
    cumulative_cost += cost

    # =====================================================================
    # Part 2: Synthetic scenarios (cache_write + cache_read)
    # =====================================================================
    print(f"\n{'═' * 70}")
    print("  PART 2: Synthetic Scenarios (cache_write + cache_read)")
    print(f"{'═' * 70}")
    print("  (Using synthetic Usage to verify formulas)")

    # ── Scenario A: Large cache write ──
    print(f"\n[Scenario A] Cache write: 50k input, 10k cache_creation, 1k output")
    print(SEPARATOR)
    usage_a = Usage(
        input_tokens=50_000,
        output_tokens=1_000,
        cache_creation_input_tokens=10_000,
    )
    print_usage(usage_a)
    print()
    print_manual_formula(usage_a, model)

    computed = calculate_cost(usage_a, model=model)
    manual = manual_anthropic_cost(usage_a, model)
    print()
    if not print_cost_comparison(computed, manual, model):
        all_ok = False

    # ── Scenario B: Cache read ──
    print(f"\n[Scenario B] Cache read: 5k input, 45k cache_read, 2k output")
    print(SEPARATOR)
    usage_b = Usage(
        input_tokens=5_000,
        output_tokens=2_000,
        cache_read_input_tokens=45_000,
    )
    print_usage(usage_b)
    print()
    print_manual_formula(usage_b, model)

    computed = calculate_cost(usage_b, model=model)
    manual = manual_anthropic_cost(usage_b, model)
    print()
    if not print_cost_comparison(computed, manual, model):
        all_ok = False

    # ── Scenario C: Mixed ──
    print(f"\n[Scenario C] Mixed: 20k input, 5k cache_write, 30k cache_read, 3k output")
    print(SEPARATOR)
    usage_c = Usage(
        input_tokens=20_000,
        output_tokens=3_000,
        cache_creation_input_tokens=5_000,
        cache_read_input_tokens=30_000,
    )
    print_usage(usage_c)
    print()
    print_manual_formula(usage_c, model)

    computed = calculate_cost(usage_c, model=model)
    manual = manual_anthropic_cost(usage_c, model)
    print()
    if not print_cost_comparison(computed, manual, model):
        all_ok = False

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'═' * 70}")
    print(f"  Cumulative real API cost (3 turns): {format_cost(cumulative_cost)}")
    if all_ok:
        print("  ✓ All 6 scenarios passed!")
    else:
        print("  ✗ Some scenarios failed — check above for details")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
