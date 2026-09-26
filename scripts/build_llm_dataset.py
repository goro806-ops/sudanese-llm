#!/usr/bin/env python3
"""
Sudanese LLM Dataset Builder Script
-----------------------------------
Exports the curated regional Sudanese knowledge base into universal LLM fine-tuning,
RAG, and instruction-tuning formats:
1. JSONL Chat Messages format (OpenAI / Hugging Face SFT / Axolotl / Unsloth)
2. JSONL Instruction format (Alpaca / Standard Instruction Fine-tuning)
3. CSV format (Pandas / SQL / Data Analysis)
4. Parquet format (Hugging Face Datasets / PyArrow / Spark / High-performance storage)
5. Comprehensive documentation (README.md) in data/processed/llm_dataset/
"""

import json
import csv
import sys
from pathlib import Path

def build_datasets():
    raw_path = Path("data/raw/sudanese_regional_knowledge.json")
    output_dir = Path("data/processed/llm_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"❌ Raw knowledge base not found at {raw_path}")
        sys.exit(1)

    with open(raw_path, "r", encoding="utf-8") as f:
        knowledge_entries = json.load(f)

    print(f"📦 Loaded {len(knowledge_entries)} knowledge entries for LLM dataset generation...")

    chat_entries = []
    instruction_entries = []
    flat_rows = []

    system_prompt = (
        "أنت مساعد ذكي متخصص في الثقافة والتراث واللهجات السودانية بمختلف أقاليمها "
        "(اللهجة الخرطومية، الدارفورية، الكردفانية، الشرقية، والشهيرة النوبية/الرطانة)."
    )

    for entry in knowledge_entries:
        topic = entry.get("topic", "ثقافة سودانية")
        region = entry.get("region", "general")
        category = entry.get("category", "general")
        content = entry.get("content", "")
        source = entry.get("source", "curated")
        entry_id = entry.get("id", "unk")

        user_query = f"تحدث عن {topic} في إقليم {region} بالسودان والمعنى/الأهمية المرتبطة به."
        assistant_reply = content

        # 1. Chat format (OpenAI / HF SFT)
        chat_item = {
            "id": entry_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_reply}
            ],
            "metadata": {
                "region": region,
                "category": category,
                "topic": topic,
                "source": source
            }
        }
        chat_entries.append(chat_item)

        # 2. Instruction format (Alpaca style)
        instruction_item = {
            "id": entry_id,
            "instruction": user_query,
            "input": f"الإقليم: {region} | التصنيف: {category}",
            "output": assistant_reply,
            "region": region,
            "category": category,
            "source": source
        }
        instruction_entries.append(instruction_item)

        # 3. Flat row for CSV & Parquet
        flat_rows.append({
            "id": entry_id,
            "region": region,
            "category": category,
            "topic": topic,
            "query": user_query,
            "response": assistant_reply,
            "source": source
        })

    # Save JSONL Chat
    chat_file = output_dir / "sudanese_llm_chat.jsonl"
    with open(chat_file, "w", encoding="utf-8") as f:
        for item in chat_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Generated JSONL Chat Dataset: {chat_file} ({len(chat_entries)} items)")

    # Save JSONL Instruction
    instruction_file = output_dir / "sudanese_instruction.jsonl"
    with open(instruction_file, "w", encoding="utf-8") as f:
        for item in instruction_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Generated JSONL Instruction Dataset: {instruction_file} ({len(instruction_entries)} items)")

    # Save CSV
    csv_file = output_dir / "sudanese_dataset.csv"
    fieldnames = ["id", "region", "category", "topic", "query", "response", "source"]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"✅ Generated CSV Dataset: {csv_file} ({len(flat_rows)} rows)")

    # Save Parquet using pandas/pyarrow if available, or lightweight PyArrow fallback
    parquet_file = output_dir / "sudanese_dataset.parquet"
    try:
        import pandas as pd
        df = pd.DataFrame(flat_rows)
        df.to_parquet(parquet_file, index=False)
        print(f"✅ Generated Parquet Dataset via Pandas: {parquet_file}")
    except Exception as e:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pylist(flat_rows)
            pq.write_table(table, parquet_file)
            print(f"✅ Generated Parquet Dataset via PyArrow: {parquet_file}")
        except Exception as e2:
            print(f"⚠️ Parquet generation skipped ({e2}). CSV/JSONL are primary formats.")

    # Generate README.md
    readme_file = output_dir / "README.md"
    readme_content = f"""# Sudanese Regional Dialects & Culture LLM Dataset

This directory contains the standardized, ready-to-train LLM dataset for Sudanese regional dialects, culture, Nubian Rotana, and heritage.

## 📊 Dataset Statistics
- **Total Records:** {len(knowledge_entries)}
- **Regions Covered:** Khartoum, Darfur, Kordofan, Eastern, Northern (Nubian Rotana)
- **Supported Paradigms:** Chat SFT, Instruction Tuning, RAG Vector Indexing, Data Analytics

## 📁 File Formats & Descriptions

| File Name | Format | Target Use Case | Compatibility |
| :--- | :--- | :--- | :--- |
| `sudanese_llm_chat.jsonl` | JSONL Chat Messages | Multi-turn Chat / SFT / LoRA Fine-tuning | OpenAI Fine-tuning, Hugging Face `SFTTrainer`, Unsloth, Axolotl |
| `sudanese_instruction.jsonl` | JSONL Alpaca Instruction | Direct Instruction-Following Fine-tuning | LLaMA-Factory, Alpaca, TRL `SFTTrainer` |
| `sudanese_dataset.csv` | CSV Tabular | Data analysis, SQL loading, custom parsers | Pandas, Excel, SQL databases |
| `sudanese_dataset.parquet` | Apache Parquet | High-performance binary columnar dataset | Hugging Face `datasets`, PyArrow, Apache Spark |

## 🚀 How to Load & Use

### 1. Load with Hugging Face `datasets`
```python
from datasets import load_dataset

# Load Chat JSONL format
dataset = load_dataset('json', data_files='data/processed/llm_dataset/sudanese_llm_chat.jsonl')

# Load Parquet format
dataset_parquet = load_dataset('parquet', data_files='data/processed/llm_dataset/sudanese_dataset.parquet')
print(dataset)
```

### 2. Load with Pandas
```python
import pandas as pd

df = pd.read_parquet('data/processed/llm_dataset/sudanese_dataset.parquet')
print(df.head())
```

### 3. Fine-tuning with Hugging Face TRL (`SFTTrainer`)
```python
from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset('json', data_files='data/processed/llm_dataset/sudanese_llm_chat.jsonl')

# Pass `dataset` directly into SFTTrainer
```
"""
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"✅ Generated Dataset Documentation: {readme_file}")

if __name__ == "__main__":
    build_datasets()
