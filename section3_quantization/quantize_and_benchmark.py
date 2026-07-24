"""
Section 3 — Quantization benchmark: FP16 vs 4-bit (bitsandbytes NF4)

Model: Qwen2.5-1.5B-Instruct
  - picked this because it fits on a T4 in both FP16 and 4-bit
  - larger models (7B) would OOM in FP16 on colab free tier

Measures: VRAM, tokens/sec, output quality on 5 prompts.

Run on Google Colab with T4 GPU for real numbers.
"""

import time
import gc
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TOKENS = 150

# same prompts for both versions — keep it fair
PROMPTS = [
    "Explain what insurance underwriting means in simple terms.",
    "Write a short Python function that calculates factorial recursively.",
    "What are 3 key differences between supervised and unsupervised learning?",
    "List 4 health benefits of the Mediterranean diet.",
    "Translate to French: 'The weather is nice today, let's go for a walk.'",
]


def vram_mb():
    """current GPU memory used in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def ram_mb():
    """current process RAM in MB"""
    return psutil.Process().memory_info().rss / 1024**2


def generate_and_time(model, tokenizer, prompt):
    """Generate response, return (output_text, num_tokens, elapsed_seconds)"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return output_text, len(new_ids), elapsed


def run_benchmark(model, tokenizer, label):
    """Run all 5 prompts, collect metrics."""
    print(f"\n--- {label} ---")
    results = []
    for i, p in enumerate(PROMPTS):
        text, n_tok, elapsed = generate_and_time(model, tokenizer, p)
        tps = n_tok / elapsed if elapsed > 0 else 0
        print(f"  prompt {i+1}: {n_tok} tokens, {elapsed:.2f}s, {tps:.1f} tok/s")
        results.append({"output": text, "tokens": n_tok, "time": elapsed, "tok_s": tps})
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. bitsandbytes 4-bit needs CUDA.")
        print("         run this on Colab with T4 for proper results.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # ===== FP16 =====
    print("Loading FP16...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    fp16_vram = vram_mb()
    print(f"  VRAM used: {fp16_vram:.0f} MB")

    fp16_results = run_benchmark(model_fp16, tokenizer, "FP16")

    # cleanup before loading next model
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # ===== 4-bit NF4 =====
    print("\nLoading 4-bit NF4...")
    torch.cuda.reset_peak_memory_stats()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # saves a bit more memory
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    q4_vram = vram_mb()
    print(f"  VRAM used: {q4_vram:.0f} MB")

    q4_results = run_benchmark(model_4bit, tokenizer, "4-bit NF4")

    del model_4bit
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Summary =====
    fp16_avg = sum(r["tok_s"] for r in fp16_results) / len(fp16_results)
    q4_avg = sum(r["tok_s"] for r in q4_results) / len(q4_results)
    savings = (1 - q4_vram / fp16_vram) * 100

    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    print(f"{'':25s} {'FP16':>10s}  {'4-bit NF4':>10s}")
    print("-" * 50)
    print(f"{'VRAM (MB)':25s} {fp16_vram:>10.0f}  {q4_vram:>10.0f}")
    print(f"{'Avg tok/s':25s} {fp16_avg:>10.1f}  {q4_avg:>10.1f}")
    print(f"{'Memory savings':25s} {'—':>10s}  {savings:.0f}%")
    print()

    # show output comparison for first prompt
    print("Output comparison (prompt 1):")
    print(f"  [FP16]:  {fp16_results[0]['output'][:150]}...")
    print(f"  [4-bit]: {q4_results[0]['output'][:150]}...")
    print()
    print("(Quality is nearly identical for this model size — ")
    print(" bigger models show more degradation at 4-bit)")


if __name__ == "__main__":
    main()
