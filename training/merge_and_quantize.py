"""
octo-llm Merge & Quantize
Merges LoRA adapter with base model and exports to GGUF Q4_K_M.

Usage:
    python training/merge_and_quantize.py [--adapter PATH] [--skip-gguf]

Requires: llama.cpp binaries (llama-quantize, convert_hf_to_gguf.py)
"""

import argparse
import os
import sys
import shutil
import subprocess
import torch
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_ADAPTER = "./training/octo-llm-dpo-adapter"
FALLBACK_ADAPTER = "./training/octo-llm-adapter"
MERGED_DIR = "./training/octo-llm-merged"
GGUF_FP16 = "./models/octo-llm-f16.gguf"
GGUF_Q4 = "./models/octo-llm-v0.1.gguf"


def find_llama_cpp():
    """Find llama.cpp installation."""
    candidates = [
        os.path.expanduser("~/llama.cpp"),
        "C:/llama.cpp",
        "C:/tools/llama.cpp",
        "/usr/local/bin",
    ]
    for d in candidates:
        convert = os.path.join(d, "convert_hf_to_gguf.py")
        if os.path.exists(convert):
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and quantize to GGUF")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF conversion")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge (use existing)")
    args = parser.parse_args()

    # Find adapter
    adapter_dir = args.adapter
    if adapter_dir is None:
        if os.path.exists(DEFAULT_ADAPTER):
            adapter_dir = DEFAULT_ADAPTER
            print(f"Using DPO adapter: {adapter_dir}")
        elif os.path.exists(FALLBACK_ADAPTER):
            adapter_dir = FALLBACK_ADAPTER
            print(f"Using SFT adapter (no DPO found): {adapter_dir}")
        else:
            print("ERROR: No adapter found. Run train_sft.py first.")
            sys.exit(1)

    # ── Stage 1: Merge LoRA with base model ──────────────────────────────
    if not args.skip_merge:
        print("=" * 60)
        print("Stage 1: Merging LoRA adapter with base model")
        print("=" * 60)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"Loading base model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cpu",  # Merge on CPU to save VRAM
            trust_remote_code=True,
        )

        print(f"Loading adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

        print("Merging weights...")
        model = model.merge_and_unload()

        print(f"Saving merged model to {MERGED_DIR}")
        os.makedirs(MERGED_DIR, exist_ok=True)
        model.save_pretrained(MERGED_DIR, safe_serialization=True)
        tokenizer.save_pretrained(MERGED_DIR)

        # Free memory
        del model
        torch.cuda.empty_cache()
        print("Merge complete!")
    else:
        print(f"Skipping merge, using existing: {MERGED_DIR}")

    if args.skip_gguf:
        print("Skipping GGUF conversion (--skip-gguf)")
        return

    # ── Stage 2: Convert to GGUF ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 2: Converting to GGUF")
    print("=" * 60)

    llama_dir = find_llama_cpp()
    if llama_dir is None:
        print("WARNING: llama.cpp not found. Trying pip-installed llama-cpp-python...")
        # Try using the convert script from transformers/llama.cpp
        print("To convert manually:")
        print(f"  python convert_hf_to_gguf.py {MERGED_DIR} --outfile {GGUF_FP16} --outtype f16")
        print(f"  llama-quantize {GGUF_FP16} {GGUF_Q4} Q4_K_M")
        return

    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_dir, "build", "bin", "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(llama_dir, "llama-quantize")

    # Convert to FP16 GGUF
    print(f"Converting to FP16 GGUF: {GGUF_FP16}")
    os.makedirs(os.path.dirname(GGUF_FP16), exist_ok=True)
    subprocess.run([
        sys.executable, convert_script,
        MERGED_DIR,
        "--outfile", GGUF_FP16,
        "--outtype", "f16",
    ], check=True)

    # Quantize to Q4_K_M
    print(f"Quantizing to Q4_K_M: {GGUF_Q4}")
    subprocess.run([
        quantize_bin,
        GGUF_FP16,
        GGUF_Q4,
        "Q4_K_M",
    ], check=True)

    # Report sizes
    fp16_size = os.path.getsize(GGUF_FP16) / 1024**2
    q4_size = os.path.getsize(GGUF_Q4) / 1024**2
    print(f"\nFP16: {fp16_size:.0f} MB")
    print(f"Q4_K_M: {q4_size:.0f} MB")

    print("\n" + "=" * 60)
    print("Quantization complete!")
    print(f"Model: {GGUF_Q4}")
    print("Test: octoflow chat --model models/octo-llm-v0.1.gguf")
    print("=" * 60)


if __name__ == "__main__":
    main()
