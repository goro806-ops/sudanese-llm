# Sudanese LLM Improvements – Phase 1 Complete

**Goal:** Make *any* LLM understand and generate authentic multi-regional Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern / Nubian Rotana).

## What is now in this folder (on main)

| Path | Description |
|------|-------------|
| `prompts/system_prompts.py` | Strong master + region-specific prompts + few-shot examples |
| `client/sudanese_client.py` | Model-agnostic client (works with Grok, GPT, Claude, local models…) |
| `rag/improved_vector_store.py` | Semantic RAG with sentence-transformers + Chroma (keyword fallback) |
| `training/train_qlora.py` | Fully runnable QLoRA fine-tuning script |
| `data/expanded_sudanese_knowledge.json` | Expanded multi-regional seed knowledge |
| `data/CONTRIBUTION_TEMPLATE.jsonl` | Template for adding real conversations |
| `evaluation/eval_rubric.md` | Native-speaker scoring rubric |
| `GOAL_CHECKLIST.md` | Clear roadmap to the goal |
| `requirements.txt` | Dependencies for the new components |

## Quick Start

### 1. Use better prompts with any LLM right now
```python
from improvements.prompts.system_prompts import build_sudanese_prompt

prompt = build_sudanese_prompt("khartoum", "ازيك يا زول كيفك؟")
# → Copy into Grok / ChatGPT / Claude
```

### 2. Model-agnostic client
```python
from improvements.client.sudanese_client import SudaneseClient

client = SudaneseClient(region="northern")
messages = client.build_messages("مسكاقمي كيف الحال؟")
```

### 3. Semantic RAG
```bash
pip install sentence-transformers chromadb
```
```python
from improvements.rag.improved_vector_store import SudaneseVectorStore
store = SudaneseVectorStore()
store.load_from_json("improvements/data/expanded_sudanese_knowledge.json")
results = store.search("حبابك عشرة", region="khartoum")
```

### 4. Fine-tune
```bash
# On Colab with T4 GPU
python improvements/training/train_qlora.py
```

## Next Priorities (to reach the goal)

1. **Expand the dataset** to thousands of high-quality conversations (use the contribution template)
2. Train and release the first adapters on Hugging Face
3. Collect native-speaker ratings with the evaluation rubric
4. Integrate the improved components into the main FastAPI app

See `GOAL_CHECKLIST.md` for the full definition of success.

The snowball is rolling on main.
