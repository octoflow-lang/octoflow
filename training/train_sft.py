"""octo-llm SFT Training Script.

Fine-tunes Qwen2.5-Coder-1.5B-Instruct on OctoFlow code generation pairs.

Usage: python training/train_sft.py
Hardware: GTX 1660 SUPER (6GB) - QLoRA, batch=1, ctx=512
"""

import json
import os
import sys
import torch
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "./training/octo-llm-checkpoints"
FINAL_ADAPTER_DIR = "./training/octo-llm-adapter"

# Training data paths
SFT_FILES = [
    "G:/OctoFlow-Enhancement/training-data/sft/batch_001.jsonl",
    "G:/OctoFlow-Enhancement/training-data/sft/batch_002_medium.jsonl",
    "G:/OctoFlow-Enhancement/training-data/sft/batch_003_hard.jsonl",
]

SYSTEM_PROMPT = (
    "You are octo-llm, an expert OctoFlow programmer. "
    "Generate correct, efficient, and idiomatic OctoFlow code "
    "based on the user's description. Respond only with the OctoFlow code."
)

# LoRA config (from spec: r=32, alpha=64)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparameters (adapted for 6GB GPU)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 1           # 6GB GPU: must be 1
GRAD_ACCUM_STEPS = 32          # effective batch = 32
MAX_SEQ_LENGTH = 512           # 6GB GPU: keep short
WARMUP_RATIO = 0.10
EVAL_SPLIT = 0.05              # 5% for validation


def check_gpu():
    """Verify GPU has enough VRAM."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. Training requires a GPU.")
        sys.exit(1)
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    total_gb = torch.cuda.mem_get_info()[1] / 1024**3
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} ({free_gb:.1f} GB free / {total_gb:.1f} GB total)")
    if free_gb < 3.0:
        print(f"WARNING: Only {free_gb:.1f} GB free. Close other apps for best results.")
    return free_gb


def load_sft_data(files):
    """Load SFT JSONL files and convert to ChatML format."""
    examples = []
    for fpath in files:
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found, skipping")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                completion = obj.get("completion", "")
                if prompt and completion:
                    examples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion},
                        ]
                    })
    print(f"Loaded {len(examples)} SFT examples from {len(files)} files")
    return examples


def main():
    print("=" * 60)
    print("octo-llm SFT Training")
    print("=" * 60)

    # Check GPU
    free_gb = check_gpu()

    # Load data
    raw_data = load_sft_data(SFT_FILES)
    if not raw_data:
        print("ERROR: No training data loaded")
        sys.exit(1)

    # Split train/val
    import random
    random.seed(42)
    random.shuffle(raw_data)
    val_size = max(100, int(len(raw_data) * EVAL_SPLIT))
    train_data = raw_data[val_size:]
    val_data = raw_data[:val_size]
    print(f"Train: {len(train_data)}, Val: {val_size}")

    # Import heavy libs after GPU check
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # Create HF datasets
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA: load model in 4-bit
    print(f"Loading model in 4-bit (QLoRA): {MODEL_NAME}")
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
        dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA adapter
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFT config (TRL 0.24+: SFTConfig replaces TrainingArguments)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        dataloader_num_workers=0,  # Windows: 0 workers
        remove_unused_columns=False,
        report_to="none",  # No wandb for now
        seed=42,
        max_length=MAX_SEQ_LENGTH,
        packing=False,  # Safer on low VRAM
    )

    # Formatting function for ChatML
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    # SFT Trainer
    print("\nStarting SFT training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch: {PER_DEVICE_BATCH} x {GRAD_ACCUM_STEPS} = {PER_DEVICE_BATCH * GRAD_ACCUM_STEPS}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Max seq: {MAX_SEQ_LENGTH}")

    total_steps = (len(train_data) // (PER_DEVICE_BATCH * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
    print(f"  Total steps: ~{total_steps}")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=None,  # Already applied via get_peft_model
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Train
    trainer.train()

    # Save adapter
    print(f"\nSaving adapter to {FINAL_ADAPTER_DIR}")
    trainer.save_model(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    print("\n" + "=" * 60)
    print("SFT training complete!")
    print(f"Adapter saved to: {FINAL_ADAPTER_DIR}")
    print("Next: Run training/merge_and_quantize.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
