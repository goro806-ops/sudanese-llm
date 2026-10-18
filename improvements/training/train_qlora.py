"""
QLoRA Fine-tuning for Sudanese Dialects
=========================================
T4-safe: fp16 AMP + LoRA weights in float32 (no bf16 GradScaler path).
Ampere+: bf16 when compute capability >= 8.0.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from trl import SFTTrainer, SFTConfig
    USE_SFT_CONFIG = True
except ImportError:
    from trl import SFTTrainer
    from transformers import TrainingArguments
    USE_SFT_CONFIG = False


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./sudanese-qlora-adapters"
DATASET_PATH = "data/processed/llm_dataset/sudanese_llm_chat.jsonl"
BATCH_GLOB = "improvements/data/conversations_batch*.jsonl"
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32


def supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def resolve_dataset_path() -> str:
    preferred = Path(DATASET_PATH)
    if preferred.exists():
        return str(preferred)

    root = Path(".")
    batches = sorted(root.glob(BATCH_GLOB))
    if not batches:
        batches = sorted(Path("../..").glob("improvements/data/conversations_batch*.jsonl"))

    if not batches:
        raise FileNotFoundError(
            f"No dataset at {DATASET_PATH} and no {BATCH_GLOB}."
        )

    preferred.parent.mkdir(parents=True, exist_ok=True)
    with open(preferred, "w", encoding="utf-8") as out:
        for b in batches:
            content = b.read_text(encoding="utf-8")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
    print(f"   Auto-merged {len(batches)} batch files → {preferred}")
    return str(preferred)


def load_sudanese_dataset(path: str) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    try:
        return load_dataset("json", data_files=str(path), split="train")
    except Exception:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return Dataset.from_list(records)


def formatting_prompts_func(example):
    messages = example.get("messages", [])
    if not messages:
        return {"text": ""}
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return {"text": "\n".join(parts)}


def force_trainable_fp32(model):
    """LoRA/trainable weights must not be bf16 on T4 (GradScaler crash)."""
    cast = 0
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            cast += 1
    if cast:
        print(f"   Cast {cast} trainable tensors → float32")


def main():
    print("=" * 60)
    print("Sudanese Dialect QLoRA Fine-tuning")
    print("=" * 60)

    use_bf16 = supports_bf16()
    # On T4: pure fp16 AMP; compute dtype float16 for 4-bit math
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a"
    print(f"\nPrecision: {'bf16' if use_bf16 else 'fp16'} (CUDA capability {cap})")

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
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,  # newer transformers prefer dtype= over torch_dtype=
    )
    model = prepare_model_for_kbit_training(model)

    print("\n[3/5] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    force_trainable_fp32(model)
    model.print_trainable_parameters()

    print("\n[4/5] Setting up trainer...")
    # T4: fp16=True only; never bf16. Avoid paged optim issues → adamw_torch.
    common = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="epoch",
        fp16=(not use_bf16),
        bf16=use_bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
    )

    if USE_SFT_CONFIG:
        args = SFTConfig(
            **common,
            max_length=MAX_SEQ_LENGTH,
            packing=False,
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        args = TrainingArguments(**common)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,
            dataset_text_field="text",
        )

    # Ensure gradient checkpointing works with PEFT
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("\n[5/5] Starting training...")
    trainer.train()

    print(f"\nSaving adapters to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Training complete!")
    print(f"Adapters saved to: {OUTPUT_DIR}")
    print('Load with: PeftModel.from_pretrained(base_model, "sudanese-qlora-adapters")')


if __name__ == "__main__":
    main()
