# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM can understand and generate natural, regionally accurate Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern/Nubian Rotana) with high authenticity.

---

## Phase 1 – Foundation ✅ COMPLETE

- [x] Strong system prompts + region-specific prompts
- [x] Few-shot examples
- [x] Model-agnostic client
- [x] Semantic RAG
- [x] QLoRA training script (auto-merges batch files)
- [x] Expanded knowledge seed
- [x] Evaluation rubric + **eval_set_v1.json** (10 held-out prompts)
- [x] Colab guide

---

## Phase 2 – Data Scale 🔄 IN PROGRESS

- [x] Batches 1–5
- [x] **~100 multi-turn conversational examples**
- [x] All 5 regions covered
- [ ] 500+ conversations
- [ ] Native-speaker verification

---

## Phase 3 – Model Quality 🟡 READY

- [x] Training script + Colab guide
- [x] Dataset auto-discovery in train_qlora.py
- [x] Eval set v1 for scoring
- [ ] Run first QLoRA training (on Colab)
- [ ] Publish adapters
- [ ] Score with rubric + eval set

---

## Phase 4 – Usability ✅

- [x] Prompts + client for any LLM

---

## Phase 5 – Measurement

- [x] Eval set v1 (starter)
- [ ] Expand to 100+ rated examples

---

## Definition of Goal Reached

Base model + prompts + RAG + (optional) adapters ≥ **4.2 / 5** on native-speaker rubric.

Snowball continues.
