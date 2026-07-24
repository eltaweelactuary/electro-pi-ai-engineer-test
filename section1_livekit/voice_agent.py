"""
voice_agent.py — QuickBite support agent (Gemini)

Uses Google Gemini for LLM + function calling.
STT/TTS mocked with text I/O.
"""

import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


@dataclass
class TranscriptionEvent:
    text: str

class MockSTT:
    async def transcribe(self, text):
        return TranscriptionEvent(text=text)

class MockTTS:
    async def synthesize(self, text):
        return text


# --- order db ---
ORDER_DB = {
    "ORD-001": {"status": "delivered", "eta": None, "restaurant": "Pizza Palace", "items": ["Margherita Pizza", "Garlic Bread"]},
    "ORD-002": {"status": "in_transit", "eta": "15 minutes", "restaurant": "Burger Barn", "items": ["Double Cheeseburger", "Fries"]},
    "ORD-003": {"status": "preparing", "eta": "30 minutes", "restaurant": "Sushi Spot", "items": ["California Roll", "Miso Soup"]},
    "ORD-004": {"status": "cancelled", "eta": None, "restaurant": "Taco Town", "items": ["Burrito Bowl"]},
}


def get_order_status(order_id: str) -> str:
    """Look up order status."""
    order_id = order_id.strip().upper()
    order = ORDER_DB.get(order_id)
    if not order:
        return f"No order found with ID '{order_id}'."
    s = order["status"]
    r = order["restaurant"]
    items = ", ".join(order["items"])
    if s == "in_transit":
        return f"{order_id} from {r} is on its way — ETA {order['eta']}. Items: {items}"
    elif s == "preparing":
        return f"{order_id} from {r} is being prepared. ETA: {order['eta']}. Items: {items}"
    elif s == "delivered":
        return f"{order_id} from {r} was delivered. Items: {items}"
    elif s == "cancelled":
        return f"{order_id} has been cancelled."
    return f"{order_id}: status '{s}'"


def cancel_order(order_id: str, reason: str) -> str:
    """Cancel an order."""
    order_id = order_id.strip().upper()
    order = ORDER_DB.get(order_id)
    if not order:
        return f"Can't find order '{order_id}'."
    if order["status"] == "delivered":
        return f"{order_id} already delivered, can't cancel."
    if order["status"] == "cancelled":
        return f"{order_id} is already cancelled."
    ORDER_DB[order_id]["status"] = "cancelled"
    return f"{order_id} cancelled. Reason: {reason}. Refund in 3-5 days."


TOOL_FNS = {"get_order_status": get_order_status, "cancel_order": cancel_order}

SYSTEM_PROMPT = (
    "You are a friendly support agent for QuickBite food delivery. "
    "Help customers check order status and cancel orders. "
    "Use the tools provided. Keep responses short."
)


class FoodDeliveryAgent:
    """Agent using Gemini function calling."""

    def __init__(self):
        self.stt = MockSTT()
        self.tts = MockTTS()

        from google import genai
        from google.genai import types

        self.types = types
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

        tool_decls = types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="get_order_status",
                description="Look up current status of a food delivery order",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"order_id": types.Schema(type=types.Type.STRING, description="e.g. ORD-001")},
                    required=["order_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="cancel_order",
                description="Cancel a pending food delivery order",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "order_id": types.Schema(type=types.Type.STRING),
                        "reason": types.Schema(type=types.Type.STRING, description="Why cancelling"),
                    },
                    required=["order_id", "reason"],
                ),
            ),
        ])

        self.chat = self.client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[tool_decls],
            ),
        )
        print("[init] Gemini connected")

    async def process_user_input(self, user_text: str) -> str:
        transcript = await self.stt.transcribe(user_text)
        print(f'\n  [user] "{transcript.text}"')

        reply = self._gemini_turn(transcript.text)

        await self.tts.synthesize(reply)
        print(f'  [agent] "{reply}"')
        return reply

    def _gemini_turn(self, text: str) -> str:
        types = self.types
        response = self.chat.send_message(text)

        for _ in range(5):  # safety limit
            fc_found = False
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc_found = True
                    name = part.function_call.name
                    args = dict(part.function_call.args)
                    print(f"  [tool] {name}({args})")

                    fn = TOOL_FNS.get(name)
                    result = fn(**args) if fn else f"Unknown tool: {name}"
                    print(f"  [result] {result}")

                    response = self.chat.send_message(
                        types.Content(parts=[
                            types.Part(function_response=types.FunctionResponse(
                                name=name, response={"result": result}
                            ))
                        ])
                    )
                    break
            if not fc_found:
                break

        return response.text


if __name__ == "__main__":
    print("Run `python run_simulation.py` for the demo.")
