# External Sudanese Dialect Sources

Guide for scaling data beyond our curated batches.  
Goal: authentic multi-regional Sudanese Arabic for any LLM.

---

## Priority sources (text / LLM)

### 1. ArSyra Sudanese (highest priority for scale)
- **What:** ~14,167 records, quality-scored, MSA↔dialect pairs, 18 categories
- **Where:** [Hugging Face](https://huggingface.co/datasets/ArSyra/arsyra-sudanese) · [Kaggle](https://www.kaggle.com/datasets/aqlomate/arsyra-sudanese)
- **License:** CC-BY-NC-SA-4.0 (non-commercial share-alike — check before commercial use)
- **Why useful:** Largest dedicated Sudanese set with verified native-speaker style data
- **How to use:** Accept HF terms → download → run `scripts/import_external.py --source arsyra --input path/to/file.jsonl`

### 2. Lisan Sudanese (Birzeit / Currasat)
- **What:** ~52K morphologically annotated tokens (Facebook/YouTube)
- **Where:** [Currasat portal](https://sina.birzeit.edu/currasat/about-en.html) · [arXiv](https://arxiv.org/abs/2212.06468)
- **License:** Check portal (often CC BY for research)
- **Why useful:** Real social-media orthography, lemmas, glosses → vocabulary & normalization
- **How to use:** Export text → convert lines into short user/assistant pairs or knowledge entries

### 3. Sudanese-Flores
- **What:** ~2,009 sentences (DEV + DEVTEST) MSA/EN ↔ Sudanese (Khartoum-centric)
- **Where:** Search "Sudanese-Flores" / OpenReview / related MT benchmarks
- **Why useful:** Clean parallel data for translation fine-tuning and eval
- **How to use:** Map each pair to a `translation` topic conversation

### 4. Community / open GitHub
| Project | Notes |
|---------|--------|
| [Sudaverse](https://github.com/sudaverse) | Ecosystem: normalizer, corpus tools, synthetic generators |
| [AnwarCS/Sudanese-Arabic-LLM](https://github.com/AnwarCS/Sudanese-Arabic-AI-Dataset) | Collaborative dataset + fine-tune intent |
| [ombabiker89-collab/sudanese-dialect-corpus](https://github.com/ombabiker89-collab/sudanese-dialect-corpus) | Small conversational focus; similar goals |

---

## Speech (later — not required for text LLM goal)

- Zenodo Sudanese dialect speech (~4h, Khartoum-leaning)
- HF: `arbml/sudanese_dialect_speech`, Ahmed Gobara ~19h single-speaker
- Commercial: larger hour-count sets (SPEECH-DATA.AI etc.)

Useful for ASR/TTS later; transcripts can feed text after cleanup.

---

## Nubian / Rotana

Very little public text data.  
Plan: community contribution + our northern batches + optional collaboration with Nubian language projects (Kenya Nubi Common Voice is a different variety — use carefully).

---

## Recommended pipeline

1. Keep expanding our `conversations_batch*.jsonl` (controlled quality).
2. Import ArSyra (or sample) → convert with `import_external.py` → merge.
3. Score with `evaluation/eval_set_v1.json` + rubric.
4. Native-speaker review of a random sample before calling the goal reached.

---

## License caution

- **CC-BY-NC-SA:** fine for research / non-commercial demos; not for unrestricted commercial products without permission.
- Always keep original attribution and a note in any derived training set.
- Our own batches are project-owned; keep them separate from NC-licensed merges if needed.
