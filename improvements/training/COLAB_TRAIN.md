# Sudanese Dialect QLoRA – Colab Quick Start (Fixed)

## 1. Runtime
Runtime → Change runtime type → **T4 GPU**

## 2. Clone
```python
!git clone https://github.com/goro806-ops/sudanese-llm.git
%cd sudanese-llm
```

## 3. Install (IMPORTANT — do not upgrade torch with -U)

Colab already has torch + CUDA. Upgrading torch alone breaks torchvision and then transformers/peft.

```python
# Uninstall conflicting bits if you already ran a broken install
!pip uninstall -y torchvision torchaudio 2>/dev/null

# Install training stack WITHOUT forcing a new torch
!pip install -q transformers datasets peft bitsandbytes accelerate trl

# Reinstall torchvision to match the torch Colab already has
!pip install -q torchvision --no-cache-dir
```

If you still see `torchvision::nms does not exist`, run this and **restart runtime**, then re-run install:

```python
!pip install -q --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets peft bitsandbytes accelerate trl
```

## 4. Prepare dataset
```python
!mkdir -p data/processed/llm_dataset
!cat improvements/data/conversations_batch*.jsonl > data/processed/llm_dataset/sudanese_llm_chat.jsonl
!wc -l data/processed/llm_dataset/sudanese_llm_chat.jsonl
```

## 5. Verify imports (optional but recommended)
```python
import torch
from transformers import TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer
print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
print("imports OK")
```

## 6. Train
```python
!python improvements/training/train_qlora.py
```

Adapters save to `./sudanese-qlora-adapters`.

## 7. Upload to Hugging Face (only after training succeeds)
```python
from huggingface_hub import login, HfApi
login()

from pathlib import Path
assert Path("./sudanese-qlora-adapters").is_dir(), "Train first — adapters folder missing"

api = HfApi()
api.upload_folder(
    folder_path="./sudanese-qlora-adapters",
    repo_id="goro806/sudanese-qlora-adapters",  # change if needed
    repo_type="model",
)
```

## Notes
- Batches 1–6 ≈ 100+ conversation examples
- Base model: Qwen2.5-7B-Instruct
- First run may take ~15–40 min on T4 depending on steps
- Do **not** run the HF upload cell until training finishes
