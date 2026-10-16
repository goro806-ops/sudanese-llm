# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM can understand and generate natural, regionally accurate Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern/Nubian Rotana) with high authenticity.

---

## Phase 1 – Foundation ✅ COMPLETE

- [x] Strong system prompts + region-specific prompts
- [x] Few-shot examples
- [x] Model-agnostic client
- [x] Improved semantic RAG
- [x] Runnable QLoRA training script
- [x] Expanded seed knowledge base
- [x] Evaluation rubric
- [x] Contribution template

---

## Phase 2 – Data Scale 🔄 IN PROGRESS

- [x] Batches 1–4
- [x] **~85 multi-turn conversational examples**
- [x] All 5 regions + food, travel, emotion, translation, Nubian depth
- [ ] Reach 500+ high-quality conversational examples
- [ ] Native-speaker verification

**Data:** `conversations_batch1.jsonl` … `batch4.jsonl`  
**How to combine:** see `data/README_DATA.md` and `training/COLAB_TRAIN.md`

---

## Phase 3 – Model Quality 🟡 READY TO START

- [x] Training script ready
- [x] Colab guide ready (`training/COLAB_TRAIN.md`)
- [x] Dataset ready to combine and train
- [ ] Run first QLoRA training
- [ ] Publish adapters on Hugging Face
- [ ] Evaluate with the rubric

---

## Phase 4 – Usability ✅ MOSTLY DONE

- [x] System prompts for any LLM
- [x] Model-agnostic client

---

## Phase 5 – Measurement

- [ ] Public test set + native speaker ratings

---

## Definition of Goal Reached

Strong base model + prompts + RAG + (optional) adapters  
score **≥ 4.2 / 5** on the native-speaker rubric across regions.

---

## Immediate next step

**Run the first training on Colab** using `improvements/training/COLAB_TRAIN.md`.

Snowball is rolling toward the goal.
