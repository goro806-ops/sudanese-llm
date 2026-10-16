# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM can understand and generate natural, regionally accurate Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern/Nubian Rotana) with high authenticity.

---

## Phase 1 – Foundation ✅ COMPLETE

- [x] Strong system prompts + region-specific prompts
- [x] Few-shot examples
- [x] Model-agnostic client
- [x] Semantic RAG
- [x] QLoRA training script (auto-merges batches)
- [x] Colab guide
- [x] Evaluation rubric + eval_set_v1

---

## Phase 2 – Data Scale 🔄 IN PROGRESS

- [x] Batches 1–5 (~100 curated conversations)
- [x] **External sources documented** (`data/SOURCES.md`)
- [x] **Import converter** (`scripts/import_external.py` for ArSyra / Flores / generic)
- [ ] Import ArSyra sample (when license accepted)
- [ ] 500+ total high-quality conversations
- [ ] Native-speaker verification

---

## Phase 3 – Model Quality 🟡 READY

- [x] Training script + Colab guide + auto dataset merge
- [ ] Run first QLoRA on Colab
- [ ] Publish adapters
- [ ] Score with eval set + rubric

---

## Phase 4 – Usability ✅

- [x] Prompts + client for any LLM

---

## Phase 5 – Measurement

- [x] Eval set v1
- [ ] Expand rated set + native scores

---

## Definition of Goal Reached

Base model + prompts + RAG + (optional) adapters ≥ **4.2 / 5** on native-speaker rubric.

**Next:** Colab first train, and/or pull ArSyra sample through the importer.

Snowball continues.
