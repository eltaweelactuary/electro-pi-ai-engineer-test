# Write-ups

## Section 1 — LiveKit

### How I'd handle barge-in (interruption)

So the main issue is: what happens when the user starts talking while the agent is still speaking?
In LiveKit's AgentSession, there's VAD (voice activity detection) baked in. You'd set something like `interrupt_on_speech=True` in the session config — once the user speaks, TTS playback stops immediately and whatever the LLM was generating gets flushed.

The tricky part is state management. When interrupted, you don't want to keep the half-spoken response in context — it confuses the LLM on the next turn. My approach would be:
- Track what was actually "committed" (fully spoken to the user) vs what was cut off
- Only add committed text to conversation history
- Tune `min_speech_duration_ms` to maybe 250-300ms so random background noise doesn't trigger false interrupts

I experimented a bit with this during development and found that without the duration threshold, even keyboard typing could trigger barge-in. Annoying.

### Adding a second tool safely

I already added `cancel_order` as a second tool in my implementation, so here's what I learned:

1. Make the tool names and descriptions clearly different — Gemini gets confused if two tools sound similar
2. Always wrap tool execution in try/except. Don't let an exception crash the whole session. Return something like `"Sorry, couldn't process that — please try again"` so the LLM can relay it naturally
3. Validate inputs inside the tool function before doing anything. I check order_id format with a simple regex
4. If the LLM sends parallel tool calls (hasn't happened to me with Gemini yet, but it can), handle them sequentially — don't let one failure block the others

---

## Section 2 — RAG Pipeline

### What I'd change for longer documents

My current setup uses 500-char chunks with 50 overlap. Works fine for the 3 short docs I have, but would definitely break down on 50-page PDFs. Here's what I'd do differently:

**Chunking**: Switch to semantic chunking — split on actual section headers / paragraph boundaries instead of fixed character count. LangChain has `MarkdownHeaderTextSplitter` which I'd use for structured docs. For unstructured stuff (scanned PDFs etc), I'd try the `SemanticChunker` that groups by embedding similarity.

**Retrieval**: Two things —
1. Add BM25 alongside the dense embeddings (hybrid search). Dense search misses exact keyword matches sometimes — like if someone asks about "ORD-123" and the embedding doesn't capture that well. BM25 catches it.
2. Re-ranking. Pull top-20 candidates with the cheap retrieval, then run them through a cross-encoder (I've used `cross-encoder/ms-marco-MiniLM-L-6-v2` before) to re-score. Much better precision.

**Other ideas**: Metadata filtering (filter by doc section before retrieval), and maybe compressing retrieved chunks with an LLM before stuffing them into context. Haven't tried the compression thing in prod though, only read about it.

---

## Section 3 — Quantization

### GPTQ/AWQ vs bitsandbytes vs GGUF — when to use what

From my experience playing around with these:

**bitsandbytes (NF4)**: My go-to for quick prototyping. Zero setup — just add `load_in_4bit=True` and you're done. But it's slow for actual serving because it dequantizes weights on every forward pass. Also great for QLoRA fine-tuning.

**GPTQ**: Better for production GPU serving. You need a calibration dataset (~128 samples) upfront which is annoying, but the resulting model is *statically* quantized — no runtime overhead. Pairs well with vLLM for batched inference.

**AWQ**: Similar to GPTQ but claims to preserve "salient" weights at higher precision. In my testing the quality difference vs GPTQ was marginal (maybe 1-2% on perplexity) but it's there. I'd pick AWQ over GPTQ if I'm deploying something customer-facing where quality matters.

**GGUF (llama.cpp)**: For when you don't have a GPU at all, or you're running on a Mac. The Q4_K_M variant is the sweet spot — keeps attention layers at higher precision. I use this on my laptop for local testing.

**My rule of thumb**:
- Just experimenting → bitsandbytes
- Shipping a GPU API → AWQ + vLLM
- Need it to run anywhere (CPU, phones, edge) → GGUF
- Budget cloud deployment → GPTQ + TGI (battle-tested)

The key difference people miss: bitsandbytes is *runtime* quantization (slow), GPTQ/AWQ/GGUF are *pre-computed* (fast at inference).

---

## Section 4 — Deployment

### Scaling to 50 concurrent users

Right now my FastAPI server handles maybe 3-5 requests before latency goes through the roof. Here's what I'd change:

**Step 1 — Switch inference engine**: Drop raw `transformers` for vLLM. It does continuous batching (PagedAttention) which means it processes multiple requests at the token level instead of one-at-a-time. Easily 5-10x better throughput.

**Step 2 — Add a queue**: Put Redis or RabbitMQ between the API gateway and the inference workers. When 50 people hit the endpoint at once, you don't want them all fighting for GPU memory. Queue them, process in batches.

**Step 3 — Horizontal scaling**: Run 2-3 replicas behind a load balancer. Kubernetes with HPA (scaling on GPU utilization or queue depth) works well. Each pod gets one T4/A10 GPU.

**Step 4 — Caching**: For common queries, cache at the semantic level — embed the query, check if something with >0.95 similarity was answered before, return cached response. Saves a ton of compute.

**Step 5 — Streaming**: Already implemented this, but it matters more at scale. Users see tokens immediately → perceived latency drops, and you free GPU memory earlier since partial responses flush.

Honestly for 50 users with a 1.5B model on a T4, I think 2 replicas with vLLM would handle it comfortably. If it were a 7B model, you'd probably need 3-4 replicas or switch to A10 GPUs.

The stack I'd actually deploy: **K8s + vLLM (AWQ model) + Nginx + Redis + Prometheus/Grafana**. Maybe Triton inference server if you need multi-model serving.
