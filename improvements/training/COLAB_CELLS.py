# Paste these as separate Colab cells (in order).
# Fixed for torch/torchvision conflict on Colab T4.

# --- Cell 1: clone ---
# !git clone https://github.com/goro806-ops/sudanese-llm.git
# %cd sudanese-llm

# --- Cell 2: install (safe) ---
# !pip uninstall -y torchvision torchaudio 2>/dev/null
# !pip install -q transformers datasets peft bitsandbytes accelerate trl
# !pip install -q torchvision --no-cache-dir

# --- Cell 3: verify ---
# import torch
# from transformers import TrainingArguments
# from peft import LoraConfig
# from trl import SFTTrainer
# print(torch.__version__, torch.cuda.is_available())

# --- Cell 4: data ---
# !mkdir -p data/processed/llm_dataset
# !cat improvements/data/conversations_batch*.jsonl > data/processed/llm_dataset/sudanese_llm_chat.jsonl
# !wc -l data/processed/llm_dataset/sudanese_llm_chat.jsonl

# --- Cell 5: train ---
# !python improvements/training/train_qlora.py
