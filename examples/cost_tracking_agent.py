"""Cost Tracking: Demonstrates per-turn cost breakdown and cumulative total.

Uses calculate_cost() and format_cost() to show cost information after
each API call in a multi-turn conversation.
"""

import asyncio

from nano_agent import DAG, ClaudeAPI
from nano_agent.providers.cost import calculate_cost, format_cost


async def main() -> None:
    api = ClaudeAPI()
    dag = DAG().system("You are a helpful assistant. Be concise.")
    total_cost = 0.0

    # Turn 1
    dag = dag.user("What is the capital of France?")
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    cost = calculate_cost(response.usage, model=response.model)
    total_cost += cost.total_cost

    print(f"Turn 1: {response.get_text()}")
    print(f"  Input cost:  {format_cost(cost.input_cost)}")
    print(f"  Output cost: {format_cost(cost.output_cost)}")
    if cost.cache_write_cost:
        print(f"  Cache write: {format_cost(cost.cache_write_cost)}")
    if cost.cache_read_cost:
        print(f"  Cache read:  {format_cost(cost.cache_read_cost)}")
    print(f"  Turn total:  {format_cost(cost.total_cost)}")
    print()

    # Turn 2
    dag = dag.user("And what is the population of that city?")
    response = await api.send(dag)
    dag = dag.assistant(response.content)

    cost = calculate_cost(response.usage, model=response.model)
    total_cost += cost.total_cost

    print(f"Turn 2: {response.get_text()}")
    print(f"  Input cost:  {format_cost(cost.input_cost)}")
    print(f"  Output cost: {format_cost(cost.output_cost)}")
    if cost.cache_write_cost:
        print(f"  Cache write: {format_cost(cost.cache_write_cost)}")
    if cost.cache_read_cost:
        print(f"  Cache read:  {format_cost(cost.cache_read_cost)}")
    print(f"  Turn total:  {format_cost(cost.total_cost)}")
    print()

    print(f"Cumulative total: {format_cost(total_cost)}")


if __name__ == "__main__":
    asyncio.run(main())
