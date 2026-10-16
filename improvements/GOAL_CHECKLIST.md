# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM can understand and generate natural, regionally accurate Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern/Nubian Rotana) with high authenticity.

---

## Phase 1 – Foundation ✅ COMPLETE

- [x] Strong system prompts + region-specific prompts
- [x] Few-shot examples
- [x] Model-agnostic client
- [x] Improved semantic RAG (sentence-transformers + Chroma)
- [x] Runnable QLoRA training script
- [x] Expanded seed knowledge base
- [x] Evaluation rubric
- [x] Contribution template

---

## Phase 2 – Data Scale (IN PROGRESS)

- [x] First high-quality conversational batch (26 multi-turn examples across all 5 regions)
- [ ] Reach **500+** high-quality conversational examples
- [ ] Reach **3,000+** total training records
- [ ] Parallel data: MSA ↔ Sudanese, English ↔ Sudanese
- [ ] Stronger Nubian Rotana coverage
- [ ] Native-speaker verified examples

**Current data assets:**
- `improvements/data/expanded_sudanese_knowledge.json` (45 knowledge entries)
- `improvements/data/conversations_batch1.jsonl` (26 multi-turn conversations)
- Original project datasets still available

---

## Phase 3 – Model Quality

- [ ] Train first QLoRA adapters using the new data
- [ ] Publish adapters on Hugging Face
- [ ] Evaluate with the rubric

---

## Phase 4 – Usability for Any LLM

- [x] System prompts ready for any LLM
- [x] Model-agnostic client ready
- [ ] One-line integration examples for Grok / GPT / Claude / Ollama
- [ ] Optional Sudanese Skill for agents

---

## Phase 5 – Measurement & Trust

- [ ] Public test set of 100+ examples
- [ ] Native speaker ratings (3–5 raters from different regions)
- [ ] Scorecard / leaderboard

---

## Definition of “Goal Reached”

A strong base model + our system prompt + RAG + (optional) adapters  
consistently scores **≥ 4.2 / 5** on the evaluation rubric  
across regions and task types, according to native speakers.

---

## Immediate Next Actions

1. Keep adding conversational batches (highest leverage)
2. Run first real QLoRA training on Colab
3. Recruit native speakers for evaluation
4. Integrate improved components into main FastAPI app

The snowball is rolling.
