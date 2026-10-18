# Sudanese Dialect QLoRA – Colab (Working Recipe)

## Critical checks before anything else

1. **Runtime → Change runtime type → T4 GPU** (not CPU)
2. **Runtime → Restart session** after changing GPU
3. Do **not** use `pip install -U torch` (breaks torchvision on Colab)

---

## Cell 1 – Clone (repo root)

```python
%cd /content
!rm -rf sudanese-llm
!git clone https://github.com/goro806-ops/sudanese-llm.git
%cd /content/sudanese-llm
!ls improvements/training/
```

## Cell 2 – Safe install

```python
# Fix dependency chaos from previous -U installs
!pip uninstall -y torchvision torchaudio 2>/dev/null
!pip install -q "numpy<2.3" "pandas<2.4"
!pip install -q transformers datasets peft bitsandbytes accelerate trl
!pip install -q torchvision --no-cache-dir
```

## Cell 3 – Verify (must pass)

```python
import torch, numpy
from transformers import TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer

print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Switch runtime to T4 GPU and Restart session"
print("imports OK")
```

If CUDA is False → you are still on CPU. Change runtime to **T4 GPU**, restart, re-run from Cell 1.

## Cell 4 – Dataset

```python
!mkdir -p data/processed/llm_dataset
!cat improvements/data/conversations_batch*.jsonl > data/processed/llm_dataset/sudanese_llm_chat.jsonl
!wc -l data/processed/llm_dataset/sudanese_llm_chat.jsonl
```

Expect ~100+ lines.

## Cell 5 – Train

```python
!python improvements/training/train_qlora.py
```

15–40 min on T4. Success creates `./sudanese-qlora-adapters/`.

## Cell 6 – Upload (only after success)

```python
from pathlib import Path
from huggingface_hub import login, HfApi
assert Path("./sudanese-qlora-adapters").is_dir(), "Train first"
login()
HfApi().upload_folder(
    folder_path="./sudanese-qlora-adapters",
    repo_id="goro806/sudanese-qlora-adapters",
    repo_type="model",
)
```

---

## Errors this recipe avoids

| Error | Cause |
|-------|--------|
| `torchvision::nms does not exist` | Upgraded torch without matching torchvision |
| `numpy.dtypes has no attribute StringDType` | numpy 2.5+ vs transformers/trl |
| QLoRA on CPU | Runtime set to Standard (CPU) |
| No adapters folder | Training never finished |

Do not let Colab AI rewrite `train_qlora.py` mid-run. Use the script from git as-is.
