#!/usr/bin/env python3
"""
Quick demo: build Sudanese dialect prompts without any GPU or API key.

Usage:
  python improvements/scripts/demo_prompts.py
  python improvements/scripts/demo_prompts.py --region northern --message "مسكاقمي"
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prompts.system_prompts import build_sudanese_prompt, REGION_PROMPTS


def main():
    ap = argparse.ArgumentParser(description="Demo Sudanese dialect prompts")
    ap.add_argument("--region", default="khartoum", choices=list(REGION_PROMPTS.keys()))
    ap.add_argument("--message", default="ازيك يا زول؟")
    ap.add_argument("--no-few-shot", action="store_true")
    args = ap.parse_args()

    prompt = build_sudanese_prompt(
        region=args.region,
        user_message=args.message,
        include_few_shot=not args.no_few_shot,
    )

    print("=" * 60)
    print(f"Region: {args.region}")
    print(f"User message: {args.message}")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print("\nCopy the prompt above into Grok / ChatGPT / Claude / any LLM.")
    print("Or use SudaneseClient for structured messages.")


if __name__ == "__main__":
    main()
