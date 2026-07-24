"""
Simulated voice session — demonstrates tool calls.
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from voice_agent import FoodDeliveryAgent

async def main():
    print("=" * 55)
    print("  QuickBite Voice Agent — Simulated Session")
    print("=" * 55)
    agent = FoodDeliveryAgent()
    turns = [
        "Hey, can you check order ORD-002?",
        "When will it arrive?",
        "Check ORD-003 too",
        "Cancel ORD-003, I changed my mind about sushi.",
        "What about ORD-999?",
    ]
    for i, t in enumerate(turns, 1):
        print(f"\n--- turn {i} ---")
        await agent.process_user_input(t)
    print("\n" + "=" * 55)
    print("  Done.")
    print("=" * 55)

if __name__ == "__main__":
    asyncio.run(main())
