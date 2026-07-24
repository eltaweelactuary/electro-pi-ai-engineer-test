"""
voice_agent.py — QuickBite food-delivery voice agent

Uses the livekit-agents Python SDK:
  - Agent subclass with a system persona (instructions=...)
  - Two @function_tool-decorated async methods (get_order_status, cancel_order)
  - An entrypoint() showing the AgentSession pipeline (STT -> LLM -> TTS)

Run modes:
  1. Real LiveKit worker (needs a LiveKit server + STT/TTS provider keys):
         python voice_agent.py dev
  2. Demo mode (no LiveKit server needed, drives the LLM + tools directly):
         python run_simulation.py
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv

# Load .env from the repo root (one level above this file)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import google as lk_google

logger = logging.getLogger("quickbite-agent")


# --------------------------------------------------------------------------- #
# Mocked "database" — replace with a real service call in production.
# --------------------------------------------------------------------------- #

ORDER_DB: dict[str, dict] = {
    "ORD-001": {
        "status": "delivered",
        "eta": None,
        "restaurant": "Pizza Palace",
        "items": ["Margherita Pizza", "Garlic Bread"],
    },
    "ORD-002": {
        "status": "in_transit",
        "eta": "15 minutes",
        "restaurant": "Burger Barn",
        "items": ["Double Cheeseburger", "Fries"],
    },
    "ORD-003": {
        "status": "preparing",
        "eta": "30 minutes",
        "restaurant": "Sushi Spot",
        "items": ["California Roll", "Miso Soup"],
    },
    "ORD-004": {
        "status": "cancelled",
        "eta": None,
        "restaurant": "Taco Town",
        "items": ["Burrito Bowl"],
    },
}


SYSTEM_PROMPT = (
    "You are a friendly customer-support assistant for QuickBite, a food delivery app. "
    "You help customers check order status and cancel orders. "
    "Always use the provided tools to look up order information — never guess. "
    "Keep replies short, conversational, and easy to say out loud."
)


# --------------------------------------------------------------------------- #
# Agent definition
# --------------------------------------------------------------------------- #


class QuickBiteAgent(Agent):
    """Voice agent for QuickBite customer support.

    The @function_tool-decorated methods below are automatically registered
    with the LLM and exposed as callable tools during a session.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
            # Gemini works well for tool-calling and is inexpensive.
            # Swap this out for openai.LLM / anthropic.LLM etc. as needed.
            llm=lk_google.LLM(model="gemini-2.0-flash"),
        )

    @function_tool
    async def get_order_status(self, context: RunContext, order_id: str) -> str:
        """Look up the current status of a food delivery order.

        Args:
            order_id: The order identifier (e.g. "ORD-001").
        """
        logger.info("get_order_status(order_id=%s)", order_id)
        oid = order_id.strip().upper()
        order = ORDER_DB.get(oid)
        if order is None:
            return (
                f"I couldn't find an order with the ID {oid}. "
                "Could you double-check the number?"
            )

        status = order["status"]
        restaurant = order["restaurant"]
        items = ", ".join(order["items"])

        if status == "in_transit":
            return (
                f"Order {oid} from {restaurant} is on the way. "
                f"ETA is about {order['eta']}. Items: {items}."
            )
        if status == "preparing":
            return (
                f"Order {oid} from {restaurant} is being prepared. "
                f"It should be ready in about {order['eta']}. Items: {items}."
            )
        if status == "delivered":
            return f"Order {oid} from {restaurant} was already delivered. Items: {items}."
        if status == "cancelled":
            return f"Order {oid} has been cancelled."
        return f"Order {oid} has status: {status}."

    @function_tool
    async def cancel_order(
        self, context: RunContext, order_id: str, reason: str
    ) -> str:
        """Cancel a pending food delivery order.

        Only orders that have not yet been delivered can be cancelled.

        Args:
            order_id: The order identifier (e.g. "ORD-001").
            reason: The customer's reason for cancelling.
        """
        logger.info("cancel_order(order_id=%s, reason=%s)", order_id, reason)
        oid = order_id.strip().upper()
        order = ORDER_DB.get(oid)
        if order is None:
            return f"I couldn't find order {oid}, so there's nothing to cancel."
        if order["status"] == "delivered":
            return (
                f"Order {oid} has already been delivered, so it can't be cancelled. "
                "If there's a problem with the order, I can help with a refund instead."
            )
        if order["status"] == "cancelled":
            return f"Order {oid} was already cancelled."

        ORDER_DB[oid]["status"] = "cancelled"
        return (
            f"Order {oid} is now cancelled. Reason: {reason}. "
            "You should see the refund in 3-5 business days."
        )


# --------------------------------------------------------------------------- #
# Production entrypoint (real LiveKit worker)
# --------------------------------------------------------------------------- #


async def entrypoint(ctx: JobContext) -> None:
    """Standard LiveKit worker entrypoint.

    In production, replace the STT/TTS below with real provider plugins
    (deepgram, elevenlabs, openai, google, cartesia, etc.). The AgentSession
    wires them together as an STT -> LLM -> TTS pipeline.

    Provider swaps require ONLY changing the two lines below — the Agent
    subclass, its tools, and the system prompt stay identical.
    """
    from livekit.plugins import deepgram, elevenlabs, silero

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        # LLM is set on the Agent itself, but you can override here too.
        tts=elevenlabs.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(agent=QuickBiteAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    # Running this file directly starts a real LiveKit worker.
    # For a demo without a LiveKit room, run: python run_simulation.py
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
