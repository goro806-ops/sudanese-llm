#!/usr/bin/env python3
"""
Hugging Face Hub Exporter for Sudanese Dialects Dataset
-------------------------------------------------------
Allows pushing the compiled multi-regional Sudanese dataset directly to
Hugging Face Hub so anyone can load it with `datasets.load_dataset()`.
"""

import os
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "llm_dataset"

def push_to_huggingface(repo_id: str, hf_token: str = None):
    """Push local Parquet / CSV / JSONL dataset directory to Hugging Face Hub."""
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not provided. Set os.environ['HF_TOKEN'] or pass --token argument.")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"🚀 Creating Hugging Face dataset repository: {repo_id}...")
        api.create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)

        print(f"📤 Uploading dataset folder '{DATASET_DIR}' to Hugging Face Hub...")
        api.upload_folder(
            folder_path=str(DATASET_DIR),
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"✅ Successfully published dataset to https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face Hub: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Push Sudanese Dialect Dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Dataset Repo ID (e.g. username/sudanese-dialects)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token")
    args = parser.parse_args()

    push_to_huggingface(args.repo_id, args.token)

if __name__ == "__main__":
    main()
