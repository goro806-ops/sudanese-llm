# External Sudanese Dialect Sources

Prioritized for **sudanese-llm** goal: any LLM handles multi-regional Sudanese Arabic inputs/outputs well.

Updated from research notes (Google AI Mode / dialect sources PDF, Oct 2026) + live HF checks.

---

## Tier A — Use next (text LLM fine-tune)

| Source | Size / type | How to use | License caution |
|--------|-------------|------------|-----------------|
| **ArSyra/arsyra-sudanese** | ~14K quality-scored records, MSA↔dialect | `import_external.py --source arsyra` | CC-BY-NC-SA (non-commercial) |
| **O96a/sudanese-mt-benchmark** | Small but gold MT pairs + dialect markers (`يا زول`, etc.) | Map `source`→assistant, EN→user translation turns | CC-BY-4.0 |
| **Lisan Sudanese** (Birzeit Currasat) | ~52K morph tokens, social media | Vocab + normalization seed | Check portal |
| **Sudanese-Flores** | ~2K parallel sentences | Translation chat turns | Research |
| **Our batches 1–7** | ~126 curated multi-turn, 5 regions | Primary quality signal | Project-owned |

```python
from datasets import load_dataset
# Example (after accepting HF terms where required)
ds = load_dataset("O96a/sudanese-mt-benchmark", split="test")
```

---

## Tier B — Scale / speech transcripts (later or ASR)

| Source | Notes |
|--------|--------|
| **oddadmix/arabic-audio-collection-sudanese-sudan-podcast** | ~27K chunks, podcast transcripts — lively spoken Sudanese |
| **oddadmix/...-ahmed-gobara** | ~19h single-speaker + text |
| **Zenodo Sudanese dialect speech** | TV/dialogue orthography without full diacritics |
| **Sudaverse** (github.com/O96a/sudaverse) | Normalizer, synthetic generators, ecosystem |

Transcripts can become chat data after cleaning; audio itself is for ASR/TTS.

---

## Tier C — Learning books (manual extract / fair use excerpts)

| Resource | Why useful |
|----------|------------|
| **Sudanese Colloquial Arabic for Beginners** (Persson & Persson, SIL) | Structured Khartoum dialogues — gold for few-shot |
| **Concise Dictionary** Sudanese Arabic↔English (Tamis & Persson, SIL) | Vocab pairs → translation / definition turns |
| **Sudan Arabic: An English-Arabic Vocabulary** (older) | Historical lexicon |
| Academic: عجب الفيا، بشير إبراهيم بشير، مجلة الدراسات السودانية | Etymology / authenticity |

Not bulk-scrapable; use for curated examples and eval design.

---

## Tier D — Platforms (inspiration, not dumps)

- Speak Sudanese / OpenArabic — phrase banks with audio  
- Talk In Arabic — video situational paths  
- ArabicNLP.uk poetry corpus — cultural tokens (multi-dialect; filter Sudanese carefully)

---

## Linguistic layers (from PDF — inform prompts & tags)

Sudanese colloquial draws from:

1. **Classical Arabic** — retained older forms  
2. **Nubian** — north (مسكاقمي، إسي، إكسي…)  
3. **Beja / Cushitic** — east  
4. **Darfur / W. African contact**  
5. **Turkish / admin loanwords** (historical)

Keep **region tags** in training data; do not flatten to one “generic Sudanese.”

---

## What the PDF scraper ideas get right vs wrong

**Right**
- Prefer HF datasets over random site scrape  
- Clean stretch letters (`كييييف` → `كيف`)  
- Instruction format for fine-tune  
- Colab for large downloads  

**Wrong / risky for our goal**
- Blind BeautifulSoup on random dictionary sites (brittle, ToS, noise)  
- Feeding raw social text without native QA pairs → model learns noise  
- Alpaca templates that put the same scrap as both input and output  
- Scraping Facebook/X without official APIs and compliance  

**Our pipeline instead**
1. Curated multi-turn batches (quality)  
2. Import ArSyra / MT benchmark via `import_external.py`  
3. Optional: podcast **transcripts only**, cleaned  
4. Retrain adapters → push HF  

---

## Recommended next actions

1. Accept ArSyra terms if non-commercial use is OK → import 500–2000 rows  
2. Convert `O96a/sudanese-mt-benchmark` into chat JSONL  
3. Add 20–30 lines from Persson-style dialogues manually  
4. Retrain `goro806/sudanese-qlora-adapters`  
5. Re-run inference eval (target path to ≥4.2 native)

See also: `scripts/import_external.py`, `conversations_batch*.jsonl`.
