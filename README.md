# Electro Pi — AI Engineer Technical Test

## What's this

My submission for the mid-level AI engineer assessment. Everything is organized by section and tested on Google Colab (free tier T4 GPU).

I focused on making things actually work end-to-end rather than over-engineering. Each section has a Colab notebook you can just upload and run.

## Repo structure

```
.
├── .env                          # your API key goes here (gitignored)
├── .gitignore
├── README.md
├── NOTES.md                      # write-ups for all sections
├── section1_livekit/
│   ├── voice_agent.py            # agent + tool definitions
│   ├── run_simulation.py         # demo script
│   └── requirements.txt
├── section2_langchain_rag/
│   ├── rag_pipeline.py           # the actual RAG chain
│   ├── run_examples.py           # runs 3 example queries
│   ├── documents/                # sample docs (food delivery domain)
│   └── requirements.txt
├── section3_quantization/
│   ├── quantize_and_benchmark.py # fp16 vs 4bit comparison
│   └── requirements.txt
├── section4_deployment/
│   ├── app.py                    # FastAPI + streaming
│   ├── Dockerfile
│   ├── load_test.py              # 10 concurrent requests test
│   └── requirements.txt
└── colab_notebooks/              # ← easiest way to run everything
    ├── Section1_LiveKit.ipynb
    ├── Section2_RAG.ipynb
    ├── Section3_Quantization.ipynb
    └── Section4_Deployment.ipynb
```

## How to run

### Option 1: Google Colab (recommended)

1. Upload any notebook from `colab_notebooks/` to colab
2. For sections 3 & 4: switch runtime to T4 GPU
3. Add your `XAI_API_KEY` to Colab Secrets (the key icon on the left sidebar)
4. Run all cells

### Option 2: Locally

```bash
# setup
pip install python-dotenv
cp .env.example .env
# edit .env → put your Gemini key

# then for any section:
cd section1_livekit
pip install -r requirements.txt
python run_simulation.py
```

## API key

I'm using **xAI Grok** (OpenAI-compatible) for sections 1 & 2.

Put it in `.env`:
```
XAI_API_KEY=xai-...
XAI_BASE_URL=https://api.x.ai/v1
```

The `.gitignore` blocks `.env` so it won't get pushed to GitHub.

## Limitations / honest notes

- **Section 1**: No actual LiveKit server running — I mocked STT/TTS with text I/O. The LLM tool-calling part is 100% real though (Gemini function calling). In prod you'd just swap MockSTT/MockTTS with deepgram/elevenlabs plugins.

- **Section 2**: My chunking strategy is basic (fixed 500 char). Works for these short docs but I discuss what I'd change for longer documents in NOTES.md.

- **Section 3**: Ran on Colab T4. Chose Qwen2.5-1.5B because it actually fits in VRAM for both fp16 AND quantized versions — larger models would OOM on fp16. Numbers are real measurements, not estimates.

- **Section 4**: Dockerfile works but downloads the model at runtime (not baked in) so first startup takes ~2 min. Trade-off: smaller image size vs cold start.

## What I'd do differently with more time

- Hook up a real LiveKit cloud instance for section 1 (was going to but the free trial signup took too long)
- Add BM25 hybrid search to section 2
- Try AWQ quantization alongside bitsandbytes for a three-way comparison
- Set up proper K8s manifests for section 4 instead of just a Dockerfile
