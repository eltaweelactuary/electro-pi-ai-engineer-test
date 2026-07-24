"""
run_simulation.py — Text-based demo of the QuickBite agent.

Purpose:
  Exercises the Agent's LLM and @function_tool methods end-to-end without
  requiring a LiveKit server, microphone, or STT/TTS provider keys.
  The spec explicitly allows mocking STT/TTS with text I/O provided
  the LLM + tool-calling logic is real, which it is here.

What runs:
  - QuickBiteAgent is instantiated (same class the production worker uses).
  - Its @function_tool methods are collected and passed to the LLM.
  - A scripted sequence of user turns is fed in; the LLM decides when to
    invoke tools; results are fed back to the LLM.
  - Every user message, tool call, tool result, and agent reply is printed.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys

# Make voice_agent importable when running from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from livekit.agents.llm import ChatContext
from voice_agent import QuickBiteAgent, SYSTEM_PROMPT


async def run_turn(agent: QuickBiteAgent, chat_ctx: ChatContext, user_text: str) -> None:
    """Send one user turn through the LLM, execute any tool calls, print output."""
    print(f'\n  [user]   "{user_text}"')
    chat_ctx.add_message(role="user", content=user_text)

    # Ask the LLM. `tools` is the list of @function_tool methods the SDK
    # collected off the agent automatically.
    stream = agent.llm.chat(chat_ctx=chat_ctx, tools=agent.tools)

    # Collect the full response and any tool calls emitted mid-stream.
    text_parts: list[str] = []
    tool_calls: list = []

    async for chunk in stream:
        delta = chunk.delta if hasattr(chunk, "delta") else None
        if delta is None:
            continue
        if getattr(delta, "content", None):
            text_parts.append(delta.content)
        if getattr(delta, "tool_calls", None):
            tool_calls.extend(delta.tool_calls)

    # If the LLM decided to call tools, invoke them and feed results back.
    while tool_calls:
        # Record the assistant's tool-call message
        chat_ctx.add_message(role="assistant", content="", tool_calls=tool_calls)

        for tc in tool_calls:
            fn_name = tc.name if hasattr(tc, "name") else tc.function.name
            raw_args = tc.arguments if hasattr(tc, "arguments") else tc.function.arguments
            args = raw_args if isinstance(raw_args, dict) else _parse_args(raw_args)

            print(f"  [tool]   {fn_name}({args})")

            fn = getattr(agent, fn_name, None)
            if fn is None:
                result = f"Error: no tool named {fn_name!r}"
            else:
                try:
                    # @function_tool wraps async methods; call directly with kwargs.
                    result = await fn(context=None, **args)
                except Exception as exc:  # noqa: BLE001
                    # Keeping the session alive on tool errors — see NOTES.md.
                    result = f"Tool error: {exc}"

            print(f"  [result] {result}")
            chat_ctx.add_message(
                role="tool",
                content=result,
                tool_call_id=getattr(tc, "id", None) or getattr(tc, "call_id", None),
            )

        # Ask the LLM again with the tool results in context.
        tool_calls = []
        text_parts = []
        stream = agent.llm.chat(chat_ctx=chat_ctx, tools=agent.tools)
        async for chunk in stream:
            delta = chunk.delta if hasattr(chunk, "delta") else None
            if delta is None:
                continue
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
            if getattr(delta, "tool_calls", None):
                tool_calls.extend(delta.tool_calls)

    final_reply = "".join(text_parts).strip()
    if final_reply:
        chat_ctx.add_message(role="assistant", content=final_reply)
        print(f'  [agent]  "{final_reply}"')


def _parse_args(raw):
    import json
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw or {}


async def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    print("=" * 60)
    print("  QuickBite Voice Agent — Simulated Session")
    print("  (LiveKit SDK, Gemini LLM, STT/TTS mocked with text I/O)")
    print("=" * 60)

    agent = QuickBiteAgent()
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content=SYSTEM_PROMPT)

    turns = [
        "Hey, can you check order ORD-002?",
        "When will it arrive?",
        "Also check ORD-003 for me.",
        "Actually, please cancel ORD-003. I changed my mind about sushi.",
        "One more thing — what's the status of ORD-999?",
    ]

    for i, text in enumerate(turns, 1):
        print(f"\n--- turn {i} ---")
        await run_turn(agent, chat_ctx, text)

    print("\n" + "=" * 60)
    print("  Session complete.")
    print("  Tool invocations demonstrated above via the [tool] / [result] log.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
