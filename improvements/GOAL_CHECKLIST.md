# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM understands/generates regionally accurate Sudanese Arabic (≥4.2/5 native).

---

## Phase 1 – Foundation ✅

- [x] Prompts, client, RAG, demo, T4 QLoRA, infer helper, HF adapters v1

---

## Phase 2 – Data 🔄 ROLL MODE

- [x] Batches 1–9 (**~192 curated conversations**)
- [x] Batch 8: O96a MT benchmark (markers: يا زول، شنو، حيجيلك…)
- [x] Batch 9: denser multi-region (food, market, lexicon, multi-turn)
- [x] SOURCES.md (ArSyra gated NC — skip until license OK)
- [ ] 500+ high-quality turns
- [ ] Native verification

---

## Phase 3 – Model

- [x] v1 adapters: https://huggingface.co/goro806/sudanese-qlora-adapters
- [x] Inference ~3.7/5
- [ ] **Retrain v2** after batches 8–9 merge

---

## Next roll options

1. Retrain v2 on full batch1–9 (~192)  
2. More curated batches (target 300+)  
3. ArSyra only after accepting HF gate + NC OK  
