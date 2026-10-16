# Path to the Goal – Checklist

**Ultimate Goal**  
Any LLM can understand and generate natural, regionally accurate Sudanese Arabic (Khartoum, Darfur, Kordofan, Eastern, Northern/Nubian Rotana) with high authenticity.

---

## Phase 1 – Foundation ✅ COMPLETE

- [x] Strong system prompts + region prompts + few-shots
- [x] Model-agnostic client
- [x] Semantic RAG
- [x] QLoRA script + Colab guide
- [x] Eval rubric + eval_set_v1
- [x] **demo_prompts.py** (zero GPU — try immediately)

---

## Phase 2 – Data Scale 🔄

- [x] Batches 1–6 (**~112 conversations**)
- [x] External sources guide + import_external.py
- [ ] Import ArSyra / more scale
- [ ] 500+ conversations
- [ ] Native-speaker verification

---

## Phase 3 – Model Quality 🟡 READY

- [x] Train script + auto-merge batches
- [ ] First Colab QLoRA run
- [ ] Publish adapters + score

---

## Phase 4 – Usability ✅

- [x] Prompts + client + **local demo script**

---

## Phase 5 – Measurement

- [x] Eval set v1

---

## Try right now (no GPU)

```bash
cd sudanese-llm
python improvements/scripts/demo_prompts.py --region northern --message "مسكاقمي"
```

Copy output into any LLM.

Snowball continues.
