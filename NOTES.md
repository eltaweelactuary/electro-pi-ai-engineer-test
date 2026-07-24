# Write-ups

## Section 1 — LiveKit

### Barge-in / interruption handling

LiveKit's `AgentSession` runs Voice Activity Detection (VAD, typically via
`silero.VAD`) alongside a turn-detection model. Barge-in falls out of that
plumbing rather than being something the Agent class handles directly:

- When VAD detects user speech while the agent is still speaking, the session
  cancels the current TTS synthesis and any in-flight LLM generation, then
  reopens the STT stream.
- The `AgentSession` fires `speech_committed` / `speech_interrupted` events;
  only text that was actually spoken out loud gets appended to the chat
  context. This prevents the model from later "remembering" a half-spoken reply
  that the user never heard.
- Tuning to avoid false interrupts: raise `min_speech_duration` so short noises
  (typing, coughs, background TV) don't trigger a cut-off, and set a short
  post-speech silence threshold before letting the agent resume.
- On the LLM plugin, `preemptive_generation=True` on the session lets the
  agent start drafting a reply while the user is still finishing their turn,
  which reduces perceived latency but must be balanced against the cost of
  cancelled generations when the user extends their turn.

### Adding a second tool safely

The submitted agent already carries two tools (`get_order_status` and
`cancel_order`) — the same shape applies to any additional tool:

1. **Schema clarity.** Give each tool a specific `name` and a docstring that
   distinguishes it clearly from every other tool. Ambiguous descriptions are
   the main cause of the LLM routing to the wrong tool.
2. **Typed, minimal arguments.** LiveKit builds the JSON schema from the
   method signature and docstring, so use narrow types (`str`, `int`, an
   `enum`-shaped `Literal[...]`) and mark truly optional parameters as
   optional. Anything left ambiguous will be filled in by the LLM's guess.
3. **Never raise; always return.** A tool that raises kills the current
   session turn. Wrap the tool body in `try / except` and return a plain
   English error string (e.g. `"That order ID doesn't look right — could you
   read it again?"`). The LLM will relay it to the user naturally.
4. **Validate inputs before side effects.** For destructive actions
   (`cancel_order`, `issue_refund`, `charge_card`), validate the ID format,
   confirm the entity exists, and check any preconditions before mutating
   state. In practice a lot of LLM-issued cancellations come with almost-right
   IDs — `ORD 001` vs `ORD-001` etc.
5. **Timeouts and idempotency.** External API calls inside a tool should have
   a short client-side timeout so a slow backend doesn't stall the whole voice
   turn. If the action can be retried, make it idempotent (client-generated
   idempotency key) so a re-invocation from the LLM doesn't create a duplicate.

---

## Section 2 — LangChain RAG

### Improving retrieval on longer documents

The current pipeline uses a `RecursiveCharacterTextSplitter` at 500 characters
with 50-character overlap plus a similarity-score threshold on the FAISS
retriever. That is fine for the three short markdown files here but would
degrade on 50-page PDFs, contracts, or manuals. The main levers:

- **Structure-aware chunking.** Fixed character sizes split mid-sentence and
  mid-clause. On markdown / HTML, `MarkdownHeaderTextSplitter` (or an HTML
  equivalent) preserves the section boundary, and attaches the heading path
  as metadata for citation. On raw PDFs, chunking on paragraphs (double
  newline) or sentences with a target token budget usually beats fixed size.
- **Semantic chunking.** For unstructured text, split on embedding-similarity
  drops — `SemanticChunker` groups adjacent sentences until the topic shifts.
  Slower to ingest, but keeps coherent ideas together.
- **Hybrid retrieval.** Combine dense (embedding) retrieval with a sparse
  keyword retriever (BM25 or `bm25s`). Reciprocal-rank fusion merges the two
  ranked lists. Dense catches paraphrases; BM25 catches exact IDs, product
  names, statute numbers — the things embeddings routinely miss.
- **Cross-encoder re-ranking.** Retrieve ~20 candidates cheaply, then re-score
  with a cross-encoder such as `cross-encoder/ms-marco-MiniLM-L-6-v2` or a
  Cohere `rerank` call. Precision at top-3 improves substantially, which is
  what actually feeds the LLM.
- **Metadata filters and hierarchical retrieval.** For long docs, tag chunks
  with section / chapter / date metadata and let the retriever pre-filter.
  For very large corpora, a two-stage design — coarse retrieval at doc level,
  fine retrieval at chunk level — keeps latency and context length in check.
- **Contextual compression.** After retrieval, run each chunk through a small
  LLM that keeps only the sentences directly relevant to the query. This lets
  more distinct chunks fit inside the context window without losing signal.

---

## Section 3 — Quantization

### GPTQ / AWQ vs bitsandbytes vs GGUF

All three cut memory by roughly 4× at 4-bit, but they solve different
problems:

**bitsandbytes (NF4).** Runtime quantization inside `transformers` — weights
are stored 4-bit and dequantized on the fly during each forward pass.
Advantages: zero calibration data, no separate build step, works for QLoRA
fine-tuning. Disadvantage: slower inference than pre-quantized formats
because of the dequant overhead per token.

**GPTQ.** Post-training quantization with a small calibration set
(typically 128–256 samples) that minimizes reconstruction error layer by
layer. Produces a static 4-bit model with no runtime overhead. Works
excellently with vLLM / TGI for batched serving. The main cost is the
calibration step and the sensitivity of quality to how representative the
calibration set is.

**AWQ (Activation-aware Weight Quantization).** Same shape as GPTQ, but
identifies "salient" weights based on activation magnitudes and keeps those
at higher precision. Typically edges out GPTQ by a small perplexity margin
at the same bit-width. Same caveats about calibration.

**GGUF (llama.cpp).** A container format plus a family of quantization
schemes (Q4_K_M, Q5_K_M, Q8_0, …). Optimized for CPU / Metal / mixed
CPU-GPU. Q4_K_M keeps attention and value layers at a higher precision than
the FFN, which preserves quality noticeably. Best fit for laptops, phones,
edge devices, or servers that need to run models without a GPU. Not the
right choice for high-throughput GPU batch serving.

**Rules of thumb:**

- Prototyping or fine-tuning → bitsandbytes (zero setup)
- Production GPU serving with high throughput → AWQ (or GPTQ) + vLLM
- Cost-sensitive cloud with batching → GPTQ + TGI
- CPU / on-device / heterogeneous hardware → GGUF Q4_K_M via llama.cpp

The core distinction people miss: **bitsandbytes dequantizes at runtime,
GPTQ/AWQ/GGUF store already-quantized kernels.** That's why the three
pre-quantized formats are always faster at pure inference than bitsandbytes
for the same bit-width.

---

## Section 4 — Deployment

### Scaling to 50 concurrent users

The submitted setup — a single FastAPI process wrapping raw `transformers` —
serializes generations on the GPU and starts queuing (or OOMing) somewhere
around 3–5 concurrent long generations on a T4. To handle 50 concurrent
users, roughly the following stack:

**Continuous batching first.** Replace `transformers.generate` with vLLM (or
TGI). vLLM's PagedAttention batches requests at the *token* level rather than
the request level — a new request can join an in-flight batch on the next
step instead of waiting for the previous batch to finish. On the same T4,
that alone is typically 5–10× throughput.

**Streaming as a first-class citizen.** Keep the SSE endpoint. Streaming
cuts perceived latency (users see tokens immediately) and lets the server
release memory as soon as a client disconnects, which matters under load.

**Prefix caching.** If every request shares a long system prompt (RAG
context, few-shot examples), vLLM's prefix cache avoids re-computing the KV
for that prefix across requests. Big win for RAG-heavy workloads.

**Horizontal scaling.** Behind a load balancer (Kubernetes with HPA on GPU
utilization + p95 latency, or ECS + ALB). For 50 users on a 1.5B model,
2–3 replicas on T4 / A10G is typically comfortable; for a 7B model, 3–4
replicas or a step up to A10G / L4.

**Queue between edge and inference.** A Redis / RabbitMQ queue absorbs spikes
and lets you cap concurrent in-flight requests per replica, which prevents
OOM under bursty load.

**Semantic response cache.** Embed the incoming query, look up an existing
answer with cosine ≥ 0.95, return that instead of regenerating. In practice
this covers a meaningful fraction of duplicate customer-support-style
queries and saves both cost and latency.

**Observability.** Expose Prometheus metrics for TTFT, tokens/sec,
concurrent-request count, and queue depth; alert on p95 latency regression.
Without this you can't tell whether you actually need more replicas or the
current ones are just poorly configured.

Concrete production stack for this workload: **Kubernetes + vLLM (AWQ 4-bit
Qwen or similar) + NGINX ingress + Redis queue + Prometheus / Grafana +
horizontal-pod-autoscaler on request-queue depth.**
