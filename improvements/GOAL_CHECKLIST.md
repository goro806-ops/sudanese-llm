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

- [x] Conversational Batch 1 (26 examples)
- [x] Conversational Batch 2 (24 examples)
- [x] **Total: 50 multi-turn conversational examples** across all 5 regions
- [ ] Reach 500+ high-quality conversational examples
- [ ] Reach 3,000+ total training records
- [ ] More parallel MSA ↔ Sudanese pairs
- [ ] Native-speaker verification

**Current data:**
- `conversations_batch1.jsonl` + `conversations_batch2.jsonl` = 50 conversations
- `expanded_sudanese_knowledge.json` = 45 knowledge entries
- Original project datasets still available

---

## Phase 3 – Model Quality

- [ ] Train first QLoRA adapters on the new conversational data
- [ ] Publish adapters on Hugging Face
- [ ] Evaluate with the rubric

---

## Phase 4 – Usability

- [x] System prompts ready for any LLM
- [x] Model-agnostic client ready
- [ ] One-line examples for Grok / GPT / Claude / Ollama

---

## Phase 5 – Measurement

- [ ] Public test set + native speaker ratings

---

## Definition of Goal Reached

Strong base model + prompts + RAG + (optional) adapters  
score **≥ 4.2 / 5** on the native-speaker rubric across regions.

---

## Next highest-leverage actions

1. Keep adding conversation batches
2. Run first real training on Colab with the 50 examples + knowledge data
3. Integrate into main FastAPI app

Snowball is rolling.
