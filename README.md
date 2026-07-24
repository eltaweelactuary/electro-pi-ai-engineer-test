# Electro Pi — AI Engineer Technical Test

Submission for the mid-level AI Engineer practical assessment.

Every section runs end-to-end. Sections 1 & 2 need only a Gemini API key (free);
Sections 3 & 4 run on Google Colab's free T4 GPU.

---

## Repo layout

```
.
├── .env.example                     # template — copy to .env and fill in key
├── .gitignore                       # blocks .env, __pycache__, model weights
├── README.md
├── NOTES.md                         # half-page write-ups per section
├── section1_livekit/
│   ├── voice_agent.py               # Agent subclass + @function_tool methods
│   ├── run_simulation.py            # text-based demo driver (STT/TTS mocked)
│   └── requirements.txt
├── section2_langchain_rag/
│   ├── rag_pipeline.py              # chunk / embed / retrieve / answer
│   ├── run_examples.py              # runs 3 example questions
│   ├── documents/                   # 3 sample markdown docs
│   └── requirements.txt
├── section3_quantization/
│   ├── quantize_and_benchmark.py    # FP16 vs 4-bit NF4 comparison
│   └── requirements.txt
├── section4_deployment/
│   ├── app.py                       # FastAPI server + streaming endpoint
│   ├── Dockerfile
│   ├── load_test.py                 # 10 concurrent requests, TTFT + latency
│   └── requirements.txt
└── colab_notebooks/                 # single-notebook version per section
    ├── Section1_LiveKit.ipynb
    ├── Section2_RAG.ipynb
    ├── Section3_Quantization.ipynb
    └── Section4_Deployment.ipynb
```

---

## Quick start — Google Colab (recommended)

1. Open a notebook from `colab_notebooks/` in Colab.
2. For Sections 3 & 4: Runtime → Change runtime type → **T4 GPU**.
3. Add `GOOGLE_API_KEY` to Colab Secrets (key icon on the left sidebar).
4. Run all cells.

## Quick start — local

```bash
# 1. Set up the key
cp .env.example .env
# then edit .env and paste your GOOGLE_API_KEY

# 2. Run any section
cd section1_livekit
pip install -r requirements.txt
python run_simulation.py
```

---

## API key

Sections 1 & 2 use **Google Gemini** (free tier, no credit card required).
Get a key at https://aistudio.google.com/apikey and put it in `.env`:

```
GOOGLE_API_KEY=your_key_here
```

`.env` is gitignored, so the key won't be pushed.

Sections 3 & 4 don't need any API key — they run a local open-weight model
(Qwen2.5-1.5B-Instruct).

---

## Assumptions and limitations

- **Section 1** — no LiveKit server is available for the demo, so STT/TTS are
  mocked with text I/O. The spec explicitly permits this. The Agent subclass,
  `@function_tool` methods, and LLM tool-calling are the real `livekit-agents`
  SDK. `voice_agent.py` also contains a full `entrypoint()` showing the
  production `AgentSession` pipeline for when real STT/TTS providers are wired
  up.
- **Section 2** — chunking is a straightforward `RecursiveCharacterTextSplitter`
  at 500 chars with 50 overlap. That is sufficient for the 3 short docs shipped
  here; alternatives for longer / noisier corpora are discussed in NOTES.md.
- **Section 3** — Qwen2.5-1.5B-Instruct was chosen because it fits in T4 VRAM
  at both FP16 and 4-bit, keeping the comparison apples-to-apples on free-tier
  compute. Larger models would OOM at FP16.
- **Section 4** — the Dockerfile downloads the model on first startup rather
  than baking it into the image. This keeps the image small (~4 GB) at the
  cost of a longer first-run cold start.

---

## Time / scope

Roughly 6 hours across all four sections. If given more time the biggest wins
would be: a real LiveKit room hookup for Section 1, hybrid BM25 + dense
retrieval for Section 2, an AWQ comparison alongside bitsandbytes for
Section 3, and proper Kubernetes manifests + vLLM for Section 4.
