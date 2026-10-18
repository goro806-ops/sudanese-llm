#!/usr/bin/env python3
"""Score replies against eval_set_v1 keyword checks (automatic proxy only)."""
import argparse
import json
from pathlib import Path

def score_reply(reply: str, keywords: list) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for k in keywords if k.lower() in reply.lower() or k in reply)
    return hits / len(keywords)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="improvements/evaluation/eval_set_v1.json")
    ap.add_argument("--replies", required=True, help="JSONL with id + reply")
    args = ap.parse_args()

    items = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    by_id = {x["id"]: x for x in items}
    scores = []
    with open(args.replies, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            item = by_id.get(row["id"])
            if not item:
                continue
            s = score_reply(row["reply"], item.get("expected_keywords", []))
            scores.append(s)
            print(f"{row['id']}: {s:.2f}  ({item['notes']})")
    if scores:
        print(f"\nMean keyword hit rate: {sum(scores)/len(scores):.2f}")
        print("(Proxy only — native speaker rubric is the real goal metric)")

if __name__ == "__main__":
    main()
