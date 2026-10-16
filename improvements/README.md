# Sudanese LLM – Snowball Toward the Goal

**Goal:** Make *any* LLM understand and generate authentic multi-regional Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern / Nubian Rotana).

This package contains the concrete upgrades that move the original project from a prototype to a serious foundation.

## Current Package Contents

```
improvements/
├── prompts/system_prompts.py          # Strong master + regional prompts + few-shots
├── rag/improved_vector_store.py       # Semantic RAG (sentence-transformers + Chroma)
├── training/train_qlora.py            # Runnable QLoRA fine-tuning script
├── client/sudanese_client.py          # Model-agnostic client for any LLM
├── web_server/improved_app.py         # Improved FastAPI
├── data/
│   ├── expanded_sudanese_knowledge.json
│   └── CONTRIBUTION_TEMPLATE.jsonl
├── evaluation/eval_rubric.md
├── GOAL_CHECKLIST.md
└── requirements.txt
```

## Quick Wins

### 1. Better prompts with any LLM
```python
from improvements.prompts.system_prompts import build_sudanese_prompt
prompt = build_sudanese_prompt("northern", "كيف الحال؟")
```

### 2. Model-agnostic client
```python
from improvements.client.sudanese_client import SudaneseClient
client = SudaneseClient(region="darfur")
messages = client.build_messages("عوافي عليك، كيفك؟")
```

### 3. Real fine-tuning
```bash
python improvements/training/train_qlora.py
```

See GOAL_CHECKLIST.md for the full path to the goal.
