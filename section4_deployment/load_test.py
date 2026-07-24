"""
Basic load test — hits the inference API with 10 concurrent requests
and measures latency + TTFT.

Usage:
  1. Start the server: python app.py
  2. Run this: python load_test.py

Nothing fancy — just httpx async requests. Could use locust for
something more serious but this gets the job done for the test.
"""

import asyncio
import time
import sys

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    import httpx

SERVER = "http://localhost:8000"
N_REQUESTS = 10
PROMPT = "What is deep learning? Explain briefly."


async def check_health(client):
    """Make sure server is up before hammering it."""
    try:
        r = await client.get(f"{SERVER}/health", timeout=5)
        info = r.json()
        print(f"Server healthy — model: {info['model']}, device: {info['device']}")
        return True
    except Exception as e:
        print(f"Server not reachable: {e}")
        return False


async def single_request(client, idx):
    """Fire one generate request, measure latency."""
    payload = {"prompt": PROMPT, "max_tokens": 80, "temperature": 0.7}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{SERVER}/generate", json=payload, timeout=60)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        if r.status_code == 200:
            data = r.json()
            return {"ok": True, "ms": elapsed, "tokens": data["tokens_generated"],
                    "tok_s": data["tokens_per_second"]}
        return {"ok": False, "ms": elapsed, "error": r.status_code}
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"ok": False, "ms": elapsed, "error": str(e)}


async def streaming_ttft(client):
    """Measure time-to-first-token on the streaming endpoint."""
    payload = {"prompt": PROMPT, "max_tokens": 80}
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0

    try:
        async with client.stream("POST", f"{SERVER}/generate/stream",
                                  json=payload, timeout=60) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk.startswith("[DONE]"):
                        break
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    n_tokens += 1

        total = (time.perf_counter() - t0) * 1000
        return {"ttft_ms": ttft or 0, "total_ms": total, "tokens": n_tokens}
    except Exception as e:
        return {"error": str(e)}


async def main():
    print("=" * 50)
    print("  Load Test — 10 concurrent requests")
    print("=" * 50)
    print()

    async with httpx.AsyncClient() as client:
        if not await check_health(client):
            print("\nStart the server first: python app.py")
            return

        # streaming TTFT test
        print("\n[1] Streaming TTFT test...")
        result = await streaming_ttft(client)
        if "error" not in result:
            print(f"    TTFT: {result['ttft_ms']:.0f} ms")
            print(f"    Total: {result['total_ms']:.0f} ms ({result['tokens']} tokens)")
        else:
            print(f"    Failed: {result['error']}")

        # concurrent requests
        print(f"\n[2] Firing {N_REQUESTS} concurrent requests...")
        wall_t0 = time.perf_counter()
        tasks = [single_request(client, i) for i in range(N_REQUESTS)]
        results = await asyncio.gather(*tasks)
        wall_time = (time.perf_counter() - wall_t0) * 1000

        ok = [r for r in results if r["ok"]]
        fail = [r for r in results if not r["ok"]]

        print(f"    Success: {len(ok)}/{N_REQUESTS}")
        if fail:
            print(f"    Failed: {len(fail)}")

        if ok:
            lats = sorted(r["ms"] for r in ok)
            avg_tps = sum(r["tok_s"] for r in ok) / len(ok)
            print(f"    Latency — min: {lats[0]:.0f}ms, "
                  f"p50: {lats[len(lats)//2]:.0f}ms, "
                  f"max: {lats[-1]:.0f}ms, "
                  f"avg: {sum(lats)/len(lats):.0f}ms")
            print(f"    Throughput: {avg_tps:.1f} tok/s avg per request")
            print(f"    Wall clock (all {N_REQUESTS}): {wall_time:.0f} ms")

    print("\n" + "=" * 50)
    print("  Done.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
