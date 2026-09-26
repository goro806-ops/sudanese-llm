"""Export complete offline Sudanese regional knowledge dataset for offline AI chatbots."""
import json
import os
from pathlib import Path

def export_offline_dataset():
    raw_file = Path("data/raw/sudanese_regional_knowledge.json")
    out_dir = Path("data/processed")
    out_file = out_dir / "offline_sudanese_knowledge_base.json"

    if not raw_file.exists():
        print(f"Error: Raw dataset {raw_file} not found.")
        return

    with open(raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Successfully exported offline dataset to {out_file} with {len(data)} entries.")

if __name__ == "__main__":
    export_offline_dataset()
