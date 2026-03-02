"""octo-llm Pilot Fine-Tune.

Quick validation run: 1 epoch on batch_001 (5,293 examples).
~165 steps at ~45s/step = ~2 hours on GTX 1660 SUPER.

Usage: python training/train_pilot.py
"""

import json
import os
import sys
import random
import torch

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "./training/octo-llm-pilot"
ADAPTER_DIR = "./training/octo-llm-pilot/final"

# Pilot: batch_001 only (5,293 trivial+easy examples)
SFT_FILES = [
    "G:/OctoFlow-Enhancement/training-data/sft/batch_001.jsonl",
]

SYSTEM_PROMPT = (
    "You are octo-llm, an expert OctoFlow programmer. "
    "Generate correct, efficient, and idiomatic OctoFlow code "
    "based on the user's description. Respond only with the OctoFlow code."
)

# LoRA (same as full spec)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Pilot hyperparameters
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1              # Pilot: 1 epoch
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 32       # effective batch = 32
MAX_SEQ_LENGTH = 512


def load_data(files):
    examples = []
    for fpath in files:
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found")
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
    return examples


def main():
    print("=" * 60)
    print("octo-llm PILOT Fine-Tune (validation run)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU")
        sys.exit(1)

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({free_gb:.1f} GB free)")

    raw_data = load_data(SFT_FILES)
    print(f"Loaded {len(raw_data)} examples")

    random.seed(42)
    random.shuffle(raw_data)
    val_size = 200
    train_data = raw_data[val_size:]
    val_data = raw_data[:val_size]
    print(f"Train: {len(train_data)}, Val: {val_size}")

    steps = (len(train_data) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    est_min = steps * 45 / 60
    print(f"Estimated: ~{steps} steps, ~{est_min:.0f} min at 45s/step")

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in 4-bit: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    print(f"\nStarting pilot training: {steps} steps, 1 epoch")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=None,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.train()

    # Save
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Quick sanity: generate one example
    print("\n" + "=" * 60)
    print("Pilot training complete! Running sanity check...")
    print("=" * 60)

    try:
        from peft import PeftModel
        del model
        torch.cuda.empty_cache()

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model.eval()

        test_prompt = "Read a CSV file and print the mean of column 'price'"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nPrompt: {test_prompt}")
        print(f"Generated:\n{generated}")

        # Check if it looks like valid OctoFlow
        has_let = "let " in generated
        has_print = "print(" in generated
        has_flow = any(kw in generated for kw in ["read_csv", "for ", "fn ", "use "])
        print(f"\nSanity: has 'let': {has_let}, has 'print': {has_print}, has flow keywords: {has_flow}")
    except Exception as e:
        print(f"Sanity check failed: {e}")

    print(f"\nAdapter: {ADAPTER_DIR}")
    print("Next steps:")
    print("  1. Full training: python training/train_sft.py (3 epochs, ~33h)")
    print("  2. Or cloud GPU: upload data + script to Colab/Lambda")
    print("  3. Merge + quantize: python training/merge_and_quantize.py --adapter training/octo-llm-pilot/final")


if __name__ == "__main__":
    main()
