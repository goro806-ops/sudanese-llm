# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM can understand and generate natural, regionally accurate Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern/Nubian Rotana) with high authenticity.

## Phase 1 – Foundation (Complete)

- [x] Strong system prompts + region-specific prompts
- [x] Few-shot examples
- [x] Model-agnostic client
- [x] Improved semantic RAG
- [x] Runnable QLoRA training script
- [x] Expanded seed knowledge base
- [x] Improved FastAPI app
- [x] Evaluation rubric

## Phase 2 – Data Scale (Main Remaining Work)

- [ ] Reach 3,000+ high-quality conversational examples
- [ ] Reach 8,000–15,000+ total training records
- [ ] Parallel data: MSA ↔ Sudanese, English ↔ Sudanese
- [ ] Stronger Nubian Rotana coverage
- [ ] Native-speaker verified examples

## Phase 3 – Model Quality

- [ ] Train and publicly release QLoRA adapters on Hugging Face
- [ ] Evaluate adapters vs base model using the rubric

## Phase 4 – Usability for Any LLM

- [ ] Simple one-line integration examples
- [ ] Optional “Sudanese Skill” for agent frameworks

## Phase 5 – Measurement & Trust

- [ ] Public test set + native speaker ratings
- [ ] Leaderboard or scorecard

## Definition of “Goal Reached”

A strong base model + our system prompt + RAG + (optional) adapters consistently scores ≥ 4.2 / 5 on the evaluation rubric across regions and task types, according to native speakers.
