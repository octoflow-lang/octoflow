"""
octo-llm DPO Training Script (Stage 2)
Refines the SFT model using preference pairs.

Usage:
    python training/train_dpo.py

Run AFTER train_sft.py completes.
"""

import json
import os
import sys
import torch
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SFT_ADAPTER_DIR = "./training/octo-llm-adapter"
DPO_OUTPUT_DIR = "./training/octo-llm-dpo-checkpoints"
FINAL_DPO_ADAPTER_DIR = "./training/octo-llm-dpo-adapter"

DPO_FILE = "G:/OctoFlow-Enhancement/training-data/dpo/dpo_001.jsonl"

SYSTEM_PROMPT = (
    "You are octo-llm, an expert OctoFlow programmer. "
    "Generate correct, efficient, and idiomatic OctoFlow code "
    "based on the user's description. Respond only with the OctoFlow code."
)


def load_dpo_data(fpath):
    """Load DPO JSONL: {prompt, chosen, rejected} → DPO format."""
    examples = []
    if not os.path.exists(fpath):
        print(f"ERROR: {fpath} not found")
        return examples
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt_text = obj.get("prompt", "")
            chosen_text = obj.get("chosen", "")
            rejected_text = obj.get("rejected", "")
            if prompt_text and chosen_text and rejected_text:
                examples.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                    ],
                    "chosen": [
                        {"role": "assistant", "content": chosen_text},
                    ],
                    "rejected": [
                        {"role": "assistant", "content": rejected_text},
                    ],
                })
    print(f"Loaded {len(examples)} DPO triples")
    return examples


def main():
    print("=" * 60)
    print("octo-llm DPO Training (Stage 2)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected")
        sys.exit(1)

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({free_gb:.1f} GB free)")

    # Check SFT adapter exists
    if not os.path.exists(SFT_ADAPTER_DIR):
        print(f"ERROR: SFT adapter not found at {SFT_ADAPTER_DIR}")
        print("Run train_sft.py first!")
        sys.exit(1)

    raw_data = load_dpo_data(DPO_FILE)
    if not raw_data:
        print("ERROR: No DPO data loaded")
        sys.exit(1)

    import random
    random.seed(42)
    random.shuffle(raw_data)
    val_size = max(50, int(len(raw_data) * 0.05))
    train_data = raw_data[val_size:]
    val_data = raw_data[:val_size]
    print(f"Train: {len(train_data)}, Val: {val_size}")

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig
    from trl import DPOConfig, DPOTrainer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit
    print(f"Loading base model in 4-bit: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load SFT adapter
    print(f"Loading SFT adapter from {SFT_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR, is_trainable=True)

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # DPO config (from spec: beta=0.1, lr=5e-5, 1 epoch)
    dpo_config = DPOConfig(
        output_dir=DPO_OUTPUT_DIR,
        beta=0.1,
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=512,
        max_prompt_length=256,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    print("\nStarting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"\nSaving DPO adapter to {FINAL_DPO_ADAPTER_DIR}")
    trainer.save_model(FINAL_DPO_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_DPO_ADAPTER_DIR)

    print("\n" + "=" * 60)
    print("DPO training complete!")
    print(f"Adapter saved to: {FINAL_DPO_ADAPTER_DIR}")
    print("Next: Run training/merge_and_quantize.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
