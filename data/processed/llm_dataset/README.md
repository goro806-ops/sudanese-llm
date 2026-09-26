# Sudanese Regional Dialects & Culture LLM Dataset

This directory contains the standardized, ready-to-train LLM dataset for Sudanese regional dialects, culture, Nubian Rotana, and heritage.

## 📊 Dataset Statistics
- **Total Records:** 229
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
