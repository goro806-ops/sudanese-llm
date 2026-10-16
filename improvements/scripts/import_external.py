#!/usr/bin/env python3
"""
Import external Sudanese dialect data into our conversation JSONL format.

Usage:
  python improvements/scripts/import_external.py --source generic --input other.jsonl --output improvements/data/imported_external.jsonl
  python improvements/scripts/import_external.py --source arsyra --input arsyra_sample.jsonl --output improvements/data/imported_arsyra.jsonl

Supported --source values:
  generic  - records with text / content / messages
  arsyra   - ArSyra-style: text, msa_text, category, dialect_group
  flores   - parallel: sudanese / msa / english fields (flexible keys)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REGION_HINTS = {
    "khartoum": "khartoum",
    "central": "khartoum",
    "darfur": "darfur",
    "kordofan": "kordofan",
    "eastern": "eastern",
    "east": "eastern",
    "northern": "northern",
    "nubian": "northern",
    "sudanese": "khartoum",
}


def guess_region(text: str, meta: Dict[str, Any]) -> str:
    blob = " ".join(
        str(meta.get(k, "")) for k in ("region", "dialect", "dialect_group", "category", "topic")
    ).lower()
    blob += " " + (text or "")[:200].lower()
    for key, region in REGION_HINTS.items():
        if key in blob:
            return region
    return "khartoum"


def system_for(region: str) -> str:
    return {
        "khartoum": "أنت مساعد سوداني يتحدث باللهجة الخرطومية الأصيلة.",
        "darfur": "أنت مساعد سوداني يتحدث بلهجة دارفور الأصيلة.",
        "kordofan": "أنت مساعد سوداني يتحدث بلهجة كردفان الأصيلة.",
        "eastern": "أنت مساعد سوداني يتحدث بلهجة شرق السودان.",
        "northern": "أنت مساعد سوداني يتحدث بلهجة الشمال والرطانة النوبية.",
    }.get(region, "أنت مساعد سوداني متخصص في اللهجات السودانية.")


def to_conversation(region: str, topic: str, user: str, assistant: str) -> Dict[str, Any]:
    return {
        "region": region,
        "topic": topic,
        "messages": [
            {"role": "system", "content": system_for(region)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": "external",
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_arsyra(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        msa = (r.get("msa_text") or "").strip()
        if not text:
            continue
        region = guess_region(text, r)
        category = (r.get("category") or "general").lower()
        if msa and msa != text:
            out.append(
                to_conversation(region, "translation", f"ترجم أو وضّح بالسوداني: {msa}", text)
            )
        else:
            out.append(
                to_conversation(
                    region,
                    category if category else "general",
                    "وضّح أو أكمل بالسوداني:",
                    text,
                )
            )
    return out


def convert_flores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        sud = (
            r.get("sudanese") or r.get("sd") or r.get("dialect") or r.get("target") or ""
        ).strip()
        msa = (r.get("msa") or r.get("arabic") or r.get("source") or "").strip()
        eng = (r.get("english") or r.get("en") or "").strip()
        if not sud:
            continue
        region = guess_region(sud, r)
        if msa:
            out.append(to_conversation(region, "translation", f"بالسوداني: {msa}", sud))
        elif eng:
            out.append(
                to_conversation(
                    region, "translation", f"Translate to Sudanese Arabic: {eng}", sud
                )
            )
        else:
            out.append(to_conversation(region, "general", "قول بالسوداني:", sud))
    return out


def convert_generic(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if "messages" in r and isinstance(r["messages"], list):
            region = r.get("region") or guess_region("", r)
            out.append(
                {
                    "region": region,
                    "topic": r.get("topic", "imported"),
                    "messages": r["messages"],
                    "source": "external",
                }
            )
            continue
        text = (r.get("text") or r.get("content") or r.get("utterance") or "").strip()
        if not text:
            continue
        region = guess_region(text, r)
        out.append(
            to_conversation(region, r.get("topic", "imported"), "بالسوداني:", text)
        )
    return out


CONVERTERS = {
    "arsyra": convert_arsyra,
    "flores": convert_flores,
    "generic": convert_generic,
}


def main():
    ap = argparse.ArgumentParser(description="Import external Sudanese data to conversation JSONL")
    ap.add_argument("--source", choices=list(CONVERTERS), default="generic")
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output conversation JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Optional max records (0 = all)")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]

    converted = CONVERTERS[args.source](rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Source: {args.source}")
    print(f"Input rows: {len(rows)}")
    print(f"Output conversations: {len(converted)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
