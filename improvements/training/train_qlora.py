"""
Proper QLoRA Fine-tuning Script for Sudanese Dialects
=====================================================
Runnable on Google Colab (T4 GPU) or any machine with a GPU.

See improvements/training/COLAB_TRAIN.md for step-by-step Colab instructions.

After training, adapters are saved and can be pushed to Hugging Face.
"""

import os
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./sudanese-qlora-adapters"
# Preferred path after combining batches (see COLAB_TRAIN.md)
DATASET_PATH = "data/processed/llm_dataset/sudanese_llm_chat.jsonl"
# Fallback: auto-combine improvements/data/conversations_batch*.jsonl
BATCH_GLOB = "improvements/data/conversations_batch*.jsonl"
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32


def resolve_dataset_path() -> str:
    """Use combined dataset if present; otherwise merge batch files on the fly."""
    preferred = Path(DATASET_PATH)
    if preferred.exists():
        return str(preferred)

    # Try to find and merge batch files
    root = Path(".")
    batches = sorted(root.glob(BATCH_GLOB))
    if not batches:
        # Also try from repo root if script is run from improvements/training/
        batches = sorted(Path("../..").glob("improvements/data/conversations_batch*.jsonl"))

    if not batches:
        raise FileNotFoundError(
            f"No dataset found at {DATASET_PATH} and no batch files matched {BATCH_GLOB}.\n"
            "Run: mkdir -p data/processed/llm_dataset && "
            "cat improvements/data/conversations_batch*.jsonl > data/processed/llm_dataset/sudanese_llm_chat.jsonl"
        )

    preferred.parent.mkdir(parents=True, exist_ok=True)
    with open(preferred, "w", encoding="utf-8") as out:
        for b in batches:
            out.write(b.read_text(encoding="utf-8"))
            if not b.read_text(encoding="utf-8").endswith("\n"):
                out.write("\n")
    print(f"   Auto-merged {len(batches)} batch files → {preferred}")
    return str(preferred)


def load_sudanese_dataset(path: str) -> Dataset:
    """Load the chat-format JSONL dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    try:
        ds = load_dataset("json", data_files=str(path), split="train")
        return ds
    except Exception:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return Dataset.from_list(records)


def formatting_prompts_func(example):
    """Convert chat messages into a single training text (Qwen-style)."""
    messages = example.get("messages", [])
    if not messages:
        return {"text": ""}

    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    return {"text": "\n".join(text_parts)}


def main():
    print("=" * 60)
    print("Sudanese Dialect QLoRA Fine-tuning")
    print("=" * 60)

    print("\n[1/5] Loading dataset...")
    data_path = resolve_dataset_path()
    dataset = load_sudanese_dataset(data_path)
    print(f"   Loaded {len(dataset)} examples from {data_path}")

    dataset = dataset.map(formatting_prompts_func, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)
    print(f"   After filtering: {len(dataset)} examples")

    print("\n[2/5] Loading tokenizer and 4-bit model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    print("\n[3/5] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[4/5] Setting up trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    print("\n[5/5] Starting training...")
    trainer.train()

    print(f"\nSaving adapters to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Training complete!")
    print(f"Adapters saved to: {OUTPUT_DIR}")
    print("Load with: PeftModel.from_pretrained(base_model, \"sudanese-qlora-adapters\")")


if __name__ == "__main__":
    main()
