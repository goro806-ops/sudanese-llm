# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM can understand and generate natural, regionally accurate Sudanese Arabic with high authenticity (≥4.2/5).

---

## Phase 1 – Foundation ✅

- [x] Prompts, client, RAG, demo, eval rubric
- [x] T4-safe QLoRA script
- [x] **infer_sudanese.py** (region-strong system prompts + HF adapters)

---

## Phase 2 – Data 🔄

- [x] Batches 1–7 (**~126 conversations**)
- [x] Stronger greetings + Nubian Rotana examples in batch 7
- [x] External sources + import script
- [ ] 500+ conversations
- [ ] Native verification

---

## Phase 3 – Model ✅ FIRST ADAPTERS LIVE

- [x] QLoRA on Colab T4 (loss 3.6→1.56)
- [x] **HF: https://huggingface.co/goro806/sudanese-qlora-adapters**
- [x] Inference test (~3.7/5 first pass)
- [ ] Retrain after batch 7+ merge
- [ ] Native rubric ≥ 4.2

---

## Phase 4 – Usability ✅

- [x] Prompts-only path
- [x] Adapter path on HF
- [x] Stronger region prompts at inference

---

## Phase 5 – Measurement

- [x] eval_set_v1 + keyword proxy script
- [ ] Native scores

Snowball continues.
