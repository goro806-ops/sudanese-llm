# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM can understand and generate natural, regionally accurate Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern/Nubian Rotana) with high authenticity.

---

## Phase 1 – Foundation ✅ COMPLETE

- [x] System prompts + region prompts + few-shots
- [x] Model-agnostic client
- [x] Semantic RAG
- [x] QLoRA script (T4-safe: AMP off, fp32 LoRA)
- [x] Colab guide
- [x] Eval rubric + eval_set_v1
- [x] demo_prompts.py

---

## Phase 2 – Data Scale 🔄

- [x] Batches 1–6 (~112 conversations)
- [x] External sources guide + import_external.py
- [ ] Import ArSyra / more scale
- [ ] 500+ conversations
- [ ] Native-speaker verification

---

## Phase 3 – Model Quality ✅ FIRST TRAIN DONE

- [x] **First QLoRA run on Colab T4 (2026-10-18)**
  - 112 examples, 3 epochs, 21 steps
  - Loss: 3.605 → 1.558 | token accuracy: 0.45 → 0.68
  - Adapters: `./sudanese-qlora-adapters`
- [ ] Upload adapters to Hugging Face
- [ ] Score with eval_set_v1 + rubric
- [ ] Retrain when data ≥ 500

---

## Phase 4 – Usability ✅

- [x] Prompts + client + demo (works without adapters)
- [ ] Inference notebook loading base + adapters

---

## Phase 5 – Measurement

- [x] Eval set v1
- [ ] Native scores ≥ 4.2 / 5

---

## Definition of Goal Reached

Base + prompts + RAG + adapters ≥ **4.2 / 5** on native-speaker rubric.

**Milestone unlocked:** first Sudanese dialect adapters exist.
