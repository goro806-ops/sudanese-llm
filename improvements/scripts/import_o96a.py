#!/usr/bin/env python3
"""Import O96a/sudanese-mt-benchmark into our chat JSONL schema."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

SYS = (
    "أنت مساعد سوداني يتحدث باللهجة السودانية العامية الأصيلة. "
    "استخدم كلمات زي: زول، شنو، مافي، يا زول. تجنب اللهجة المصرية والشامية."
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="improvements/data/external_o96a.jsonl")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("O96a/sudanese-mt-benchmark", "data/sudanese_test.json", repo_type="dataset")
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for x in items:
            src, ref = x["source"], x["reference"]
            markers = x.get("dialect_markers") or []
            notes = x.get("notes", "")
            rows = [
                {"region": "general", "topic": "translation_en_sud", "source": "O96a/sudanese-mt-benchmark",
                 "id": x["id"] + "_en2sud",
                 "messages": [{"role": "system", "content": SYS},
                              {"role": "user", "content": f"ترجم للسوداني العامي: {ref}"},
                              {"role": "assistant", "content": src}]},
                {"region": "general", "topic": "translation_sud_en", "source": "O96a/sudanese-mt-benchmark",
                 "id": x["id"] + "_sud2en",
                 "messages": [{"role": "system", "content": SYS},
                              {"role": "user", "content": f"شنو معنى: {src}"},
                              {"role": "assistant", "content": f"يعني تقريباً: {ref}" + (f" (ملاحظة: {notes})" if notes else "")}]},
            ]
            if markers:
                rows.append({"region": "general", "topic": "dialect_markers", "source": "O96a/sudanese-mt-benchmark",
                             "id": x["id"] + "_markers",
                             "messages": [{"role": "system", "content": SYS},
                                          {"role": "user", "content": f"هل الجملة دي سودانية؟ {src}"},
                                          {"role": "assistant", "content": f"أيوا سودانية. فيها علامات زي: {', '.join(markers)}."}]})
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} rows to {out}")

if __name__ == "__main__":
    main()
