# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM understands/generates regionally accurate Sudanese Arabic (≥4.2/5 native).

---

## Phase 1 – Foundation ✅

- [x] Prompts, client, RAG, demo, T4 QLoRA, infer helper, HF adapters v1

---

## Phase 2 – Data 🔄 ROLL MODE

- [x] Batches 1–10 (**~215 curated conversations**)
- [x] Batch 8: O96a MT markers
- [x] Batch 9: food/market/lexicon
- [x] Batch 10: multi-turn hard (taxi, family, wedding, dialect ID, regions)
- [ ] 500+ high-quality turns
- [ ] Native verification

---

## Phase 3 – Model

- [x] v1: https://huggingface.co/goro806/sudanese-qlora-adapters (~3.7/5)
- [ ] **Retrain v2** on batches 1–10 (~215)

---

## Retrain v2 (Colab T4)

```
%cd /content/sudanese-llm
!git pull
!mkdir -p data/processed/llm_dataset
!cat improvements/data/conversations_batch*.jsonl > data/processed/llm_dataset/sudanese_llm_chat.jsonl
!wc -l data/processed/llm_dataset/sudanese_llm_chat.jsonl
!python improvements/training/train_qlora.py
# then re-upload to goro806/sudanese-qlora-adapters
```
