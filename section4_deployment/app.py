"""
app.py — FastAPI inference server for Qwen2.5-1.5B-Instruct

Endpoints:
  GET  /health           -> server status
  POST /generate         -> full response (JSON)
  POST /generate/stream  -> streaming response (SSE)

Why FastAPI over vLLM/TGI:
  - Full control over the streaming logic
  - Easier to show in a Dockerfile for this test
  - In production I'd definitely use vLLM (see NOTES.md)

Loads the model in 4-bit to fit on T4 comfortably.
"""

import os
import time
import torch
from threading import Thread
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TextIteratorStreamer,
)
import uvicorn


# --- config ---
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
USE_4BIT = os.environ.get("USE_4BIT", "true").lower() == "true"
MAX_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
PORT = int(os.environ.get("PORT", "8000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# globals — loaded at startup
model = None
tokenizer = None


def load_model():
    global model, tokenizer
    print(f"loading {MODEL_ID} (4bit={USE_4BIT}, device={DEVICE})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if USE_4BIT and DEVICE == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
    else:
        # fallback: fp16 on gpu or fp32 on cpu
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=dtype,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )

    model.eval()
    mem = torch.cuda.memory_allocated() / 1024**2 if DEVICE == "cuda" else 0
    print(f"ready. VRAM: {mem:.0f} MB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # cleanup on shutdown
    global model, tokenizer
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- app ---
app = FastAPI(title="LLM Inference API", lifespan=lifespan)


# --- schemas ---
class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenResponse(BaseModel):
    text: str
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float


# --- routes ---

@app.get("/health")
def health():
    vram = torch.cuda.memory_allocated() / 1024**2 if DEVICE == "cuda" else None
    return {
        "status": "healthy" if model else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "quantized": USE_4BIT,
        "vram_mb": vram,
    }


@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    if model is None:
        raise HTTPException(503, "model still loading")

    msgs = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=req.max_tokens,
            temperature=max(req.temperature, 0.01),  # avoid div by zero
            do_sample=req.temperature > 0,
        )
    elapsed = time.perf_counter() - t0

    new_ids = ids[0][inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(new_ids, skip_special_tokens=True)
    n = len(new_ids)

    return GenResponse(
        text=output,
        tokens_generated=n,
        total_time_ms=elapsed * 1000,
        tokens_per_second=n / elapsed if elapsed > 0 else 0,
    )


@app.post("/generate/stream")
def generate_stream(req: GenRequest):
    """Token-by-token streaming via Server-Sent Events."""
    if model is None:
        raise HTTPException(503, "model still loading")

    msgs = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # TextIteratorStreamer yields tokens as they're generated
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "max_new_tokens": req.max_tokens,
        "temperature": max(req.temperature, 0.01),
        "do_sample": req.temperature > 0,
        "streamer": streamer,
    }

    # run generation in background thread so we can stream from main
    thread = Thread(target=lambda: model.generate(**gen_kwargs))

    def event_stream():
        thread.start()
        for token_text in streamer:
            if token_text:
                yield f"data: {token_text}\n\n"
        thread.join()
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                           headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
