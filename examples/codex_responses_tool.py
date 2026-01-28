"""Demo: call Codex Responses endpoint with Codex OAuth token and tool usage.

Features:
- Reasoning enabled (model thinks before responding)
- Tool usage with BashTool
- Multi-turn conversation with tool results

Note: Reasoning summary text requires organization verification to access.
The encrypted_content is returned for multi-turn continuation but summary may be empty.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from nano_agent import BashTool, OpenAIAPI, get_codex_access_token

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


def _codex_tool_schema(api: OpenAIAPI, tool: BashTool) -> dict[str, object]:
    schema = api._convert_tool_to_openai(tool)
    params = schema.get("parameters")
    if isinstance(params, dict):
        props = params.get("properties")
        if isinstance(props, dict):
            params["required"] = list(props.keys())
    return schema


async def _stream_response(
    client: httpx.AsyncClient,
    token: str,
    request_body: dict[str, object],
) -> dict[str, Any] | None:
    last_response: dict[str, Any] | None = None
    async with client.stream(
        "POST",
        CODEX_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=request_body,
    ) as resp:
        print("Status:", resp.status_code)
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    print(line)
                    continue
                if event.get("type") == "response.completed":
                    last_response = event.get("response")
            else:
                print(line)
    return last_response


def _extract_function_call(response: dict[str, Any]) -> dict[str, Any] | None:
    for item in response.get("output", []):
        if item.get("type") == "function_call":
            return item
    return None


def _extract_assistant_text(response: dict[str, Any]) -> str | None:
    for item in response.get("output", []):
        if item.get("type") == "message":
            parts = []
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    parts.append(block.get("text", ""))
            if parts:
                return "".join(parts).strip()
    return None


def _extract_reasoning(response: dict[str, Any]) -> str | None:
    """Extract reasoning summary text from response."""
    for item in response.get("output", []):
        if item.get("type") == "reasoning":
            summary = item.get("summary", [])
            parts = []
            for block in summary:
                if block.get("type") == "summary_text":
                    parts.append(block.get("text", ""))
            if parts:
                return "".join(parts).strip()
    return None


async def main() -> None:
    token = get_codex_access_token()
    if not token:
        raise SystemExit(
            "Codex token not found. Ensure Codex is configured to store auth in file mode."
        )

    api = OpenAIAPI(api_key=token, base_url=CODEX_RESPONSES_URL, reasoning=True)

    tool = BashTool()
    tools = [tool]

    user_text = "Use the Bash tool to echo 'codex ok', then reply with the output."
    context: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }
    ]

    request_body: dict[str, object] = {
        "model": api.model,
        "instructions": "You are a helpful coding assistant.",
        "input": list(context),
        "store": False,
        "stream": True,
        "tools": [_codex_tool_schema(api, t) for t in tools],
        "parallel_tool_calls": api.parallel_tool_calls,
        "reasoning": {"effort": "xhigh", "summary": "detailed"},
        "include": ["reasoning.encrypted_content"],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await _stream_response(client, token, request_body)
        if not response:
            raise SystemExit("No response received.")

        # Print usage statistics
        usage = response.get("usage", {})
        input_details = usage.get("input_tokens_details", {})
        output_details = usage.get("output_tokens_details", {})
        print("\n[Usage Statistics]")
        print(f"  Input tokens: {usage.get('input_tokens', 0)}")
        print(f"    - Cached: {input_details.get('cached_tokens', 0)}")
        print(f"  Output tokens: {usage.get('output_tokens', 0)}")
        print(f"    - Reasoning: {output_details.get('reasoning_tokens', 0)}")
        print(f"  Total tokens: {usage.get('total_tokens', 0)}")

        # Print reasoning if present
        reasoning = _extract_reasoning(response)
        if reasoning:
            print("\n[Reasoning]")
            print(reasoning)
            print()
        
        function_call = _extract_function_call(response)
        if not function_call:
            print("No tool call found.")
            print(json.dumps(response, indent=2))
            return

        call_id = function_call.get("call_id")
        arguments = function_call.get("arguments", "{}")
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = {}

        # Execute tool locally
        result = await tool.execute(args)
        if isinstance(result, list):
            tool_output = "".join(tc.text for tc in result)
        else:
            tool_output = result.text

        # Build follow-up context: prior context + model output items + tool output
        followup_context = list(context)
        followup_context.extend(response.get("output", []))
        followup_context.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_output,
            }
        )

        followup_body: dict[str, object] = {
            "model": api.model,
            "instructions": "You are a helpful coding assistant.",
            "input": followup_context,
            "store": False,
            "stream": True,
            "tools": [_codex_tool_schema(api, t) for t in tools],
            "parallel_tool_calls": api.parallel_tool_calls,
            "reasoning": {"effort": "xhigh", "summary": "detailed"},
            "include": ["reasoning.encrypted_content"],
        }

        final_response = await _stream_response(client, token, followup_body)
        if not final_response:
            raise SystemExit("No final response received.")

        # Print reasoning if present
        final_reasoning = _extract_reasoning(final_response)
        if final_reasoning:
            print("\n[Reasoning]")
            print(final_reasoning)
            print()

        assistant_text = _extract_assistant_text(final_response)
        print("Assistant:")
        print(assistant_text or "(no text)")


if __name__ == "__main__":
    asyncio.run(main())
