# Sudanese Dialect QLoRA – Colab Quick Start

## 1. Open Google Colab
Runtime → Change runtime type → **T4 GPU**

## 2. Clone the repo
```python
!git clone https://github.com/goro806-ops/sudanese-llm.git
%cd sudanese-llm
```

## 3. Install dependencies
```python
!pip install -q -U torch transformers datasets peft bitsandbytes accelerate trl
```

## 4. Prepare the dataset
```python
!mkdir -p data/processed/llm_dataset
!cat improvements/data/conversations_batch*.jsonl > data/processed/llm_dataset/sudanese_llm_chat.jsonl
!wc -l data/processed/llm_dataset/sudanese_llm_chat.jsonl
```

## 5. Run training
```python
!python improvements/training/train_qlora.py
```

Training will save adapters to `./sudanese-qlora-adapters`.

## 6. (Optional) Push adapters to Hugging Face
```python
from huggingface_hub import login, HfApi
login()  # paste your HF token

api = HfApi()
api.upload_folder(
    folder_path="./sudanese-qlora-adapters",
    repo_id="YOUR_USERNAME/sudanese-qlora-adapters",
    repo_type="model",
)
```

## Notes
- Conversational data: batches 1–4 (~85 examples)
- For better quality, keep adding real native-speaker conversations
- Base model: Qwen2.5-7B-Instruct
- After training: `PeftModel.from_pretrained(base_model, "sudanese-qlora-adapters")`
