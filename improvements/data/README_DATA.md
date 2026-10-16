# Sudanese Conversational Data

## Files

| File | Examples | Description |
|------|----------|-------------|
| `conversations_batch1.jsonl` | 26 | First multi-turn batch |
| `conversations_batch2.jsonl` | 24 | Multi-turn, Nubian, translations, edge cases |
| `conversations_batch3.jsonl` | 20 | Practical scenarios, deeper regional flavor |
| **Total** | **70** | Across Khartoum, Darfur, Kordofan, Eastern, Northern |

## Format

Each line is a JSON object:
```json
{
  "region": "khartoum",
  "topic": "multi_turn",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## How to use for training

```bash
# Combine all batches
cat improvements/data/conversations_batch*.jsonl > data/processed/llm_dataset/sudanese_llm_chat.jsonl

# Then run QLoRA
python improvements/training/train_qlora.py
```

Or point `DATASET_PATH` in `train_qlora.py` to any of the batch files.

## Next

Keep adding batches. Target: 500+ high-quality conversations.
