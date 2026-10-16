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

- [x] Batch 1 (26) + Batch 2 (24) + Batch 3 (20)
- [x] **Total: 70 multi-turn conversational examples**
- [x] All 5 regions covered with multi-turn, vocabulary, culture, translation, edge cases
- [ ] Reach 500+ high-quality conversational examples
- [ ] Reach 3,000+ total training records
- [ ] Native-speaker verification

**Data files:**
- `conversations_batch1.jsonl`
- `conversations_batch2.jsonl`
- `conversations_batch3.jsonl`
- `expanded_sudanese_knowledge.json` (45 entries)
- See `data/README_DATA.md` for how to combine for training

---

## Phase 3 – Model Quality

- [ ] Train first QLoRA adapters (script ready + 70 conversations ready)
- [ ] Publish adapters on Hugging Face
- [ ] Evaluate with the rubric

---

## Phase 4 – Usability

- [x] System prompts ready for any LLM
- [x] Model-agnostic client ready

---

## Phase 5 – Measurement

- [ ] Public test set + native speaker ratings

---

## Definition of Goal Reached

Strong base model + prompts + RAG + (optional) adapters  
score **≥ 4.2 / 5** on the native-speaker rubric across regions.

---

## Next highest-leverage actions

1. Keep adding conversation batches (or real native data)
2. **Run first real QLoRA training** on Colab with the 70 examples
3. Integrate improved components into main FastAPI app

Snowball is rolling.
