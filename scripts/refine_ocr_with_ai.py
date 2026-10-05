#!/usr/bin/env python3
"""
AI & Rule-Based OCR Dictionary Refiner Script
---------------------------------------------
Refines raw "blind" OCR text extracted from scanned PDF dictionary pages in `sudanese_lexicon.db`:
1. Cleans noise characters, random OCR symbols, and misplaced page numbers.
2. Expands Sudanese dictionary abbreviations:
   - (س) -> سودانية / عامية
   - (ف) -> فصيح / أصل فصيح
   - (م) -> مثل / مقولة شعبية
   - (ج) -> الجمع
   - (غرب) / (دارفور) / (كردفان) / (شمال) / (بطانة) -> تحديد الإقليم
3. Parses structured entry attributes: term, meaning, region, category, example, phonetic.
4. Uses OpenAI API (if OPENAI_API_KEY is available) or a robust offline rule-based heuristic parser fallback.
"""

import os
import re
import sys
import json
import sqlite3
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "processed" / "sudanese_lexicon.db"

# Dictionary region mapping rules
REGION_KEYWORDS = {
    "دارفور": "darfur",
    "غرب": "darfur",
    "كردفان": "kordofan",
    "شرق": "eastern",
    "بجا": "eastern",
    "شمال": "northern",
    "دنقلا": "northern",
    "نوبية": "northern",
    "بطانة": "khartoum",
    "خرطوم": "khartoum",
    "شايقية": "northern",
    "رباطاب": "northern",
    "جعليين": "khartoum"
}

ABBREVIATION_REPLACEMENTS = [
    (r'\(\s*س\s*\)', ' [سودانية عامية] '),
    (r'\(\s*ف\s*\)', ' [فصيحة] '),
    (r'\(\s*م\s*\)', ' [مثل شعبي] '),
    (r'\(\s*ج\s*\)', ' [الجمع] '),
    (r'\(\s*غرب\s*\)', ' [إقليم غرب السودان] '),
    (r'\(\s*دارفور\s*\)', ' [إقليم دارفور] '),
    (r'\(\s*كردفان\s*\)', ' [إقليم كردفان] '),
    (r'\(\s*شمال\s*\)', ' [الشمال] '),
    (r'\(\s*بطانة\s*\)', ' [البطانة] '),
    (r'\(\s*ن\s*\)', ' [نوبية] '),
]

def clean_ocr_text(raw_text: str) -> str:
    """Clean random OCR noise symbols and whitespace."""
    if not raw_text:
        return ""

    # Remove random OCR artifacts like isolated non-Arabic characters or numbers attached to symbols
    cleaned = re.sub(r'[\^~\*#_%<>{}\|\\\]\[]', ' ', raw_text)
    # Remove isolated English letters that OCR mistook
    cleaned = re.sub(r'\b[a-zA-Z]\b', ' ', cleaned)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def expand_abbreviations(text: str) -> str:
    """Expand parenthetical dictionary code abbreviations."""
    for pattern, replacement in ABBREVIATION_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text

def detect_region(text: str) -> str:
    """Detect Sudanese region from context keywords."""
    for kw, reg in REGION_KEYWORDS.items():
        if kw in text:
            return reg
    return "khartoum"

def heuristic_refine_entry(raw_term: str, raw_meaning: str) -> dict:
    """Offline heuristic refiner for raw OCR entries when API is unavailable."""
    cleaned_term = clean_ocr_text(raw_term)
    cleaned_meaning = clean_ocr_text(raw_meaning)

    # Clean the term from leading non-words
    cleaned_term = re.sub(r'^[^\w\s]+', '', cleaned_term)
    words = cleaned_term.split()
    term = words[0] if words else "مفردة"

    # Expand dictionary codes in meaning
    expanded_meaning = expand_abbreviations(cleaned_meaning)
    region = detect_region(expanded_meaning)

    # Determine category based on context tags
    category = "مفردات وشرح ثقافي"
    if "[مثل شعبي]" in expanded_meaning or "مثل" in expanded_meaning:
        category = "مثل_ومقولة"
    elif "ترحيب" in expanded_meaning or "سلام" in expanded_meaning:
        category = "ترحيب / تحية"

    example = f"من القاموس: {expanded_meaning[:120]}"

    return {
        "term": term,
        "meaning": expanded_meaning,
        "region": region,
        "category": category,
        "example": example,
        "phonetic": term
    }

def openai_refine_entry(raw_term: str, raw_meaning: str, api_key: str) -> dict:
    """Use OpenAI GPT to clean and structure raw OCR dictionary text."""
    try:
        import urllib.request
        prompt = f"""أنت خبير في التراث واللهجات السودانية وتحقيق المعاجم.
قم بتنظيف وتنسيق النص التالي المستخرج بتقنية OCR من قاموس العامية السودانية:

المفردة الخام: {raw_term}
النص الخام: {raw_meaning}

التعليمات:
1. استخرج المفردة الدقيقة (term).
2. فك الاختصارات مثل (س -> سودانية)، (ف -> فصيح)، (م -> مثل شعبي)، (ج -> الجمع).
3. اكتب الشرح والمعنى بلغة عربية واضحة ومفهومة (meaning).
4. حدد الإقليم (khartoum, darfur, kordofan, eastern, northern) بناءً على سياق النص (region).
5. حدد التصنيف المناسب (category).

أرجع النتيجة بصيغة JSON فقط:
{{"term": "...", "meaning": "...", "region": "...", "category": "...", "example": "...", "phonetic": "..."}}
"""
        req_data = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed
    except Exception as err:
        print(f"⚠️ OpenAI API call failed ({err}), using heuristic fallback...")
        return heuristic_refine_entry(raw_term, raw_meaning)

def refine_ocr_database(db_path: Path = DB_PATH, limit: int = None, use_ai: bool = True):
    """Process and refine raw OCR dictionary records in the SQLite database."""
    if not db_path.exists():
        print(f"❌ Database file not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT id, term, meaning FROM lexicon WHERE category = 'dictionary_book'"
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()

    print(f"🔄 Found {len(rows)} raw OCR dictionary entries to refine...")

    api_key = os.getenv("OPENAI_API_KEY") if use_ai else None
    if use_ai and not api_key:
        print("ℹ️ No OPENAI_API_KEY found in environment. Using rule-based heuristic OCR cleaner...")

    updated_count = 0
    for row_id, raw_term, raw_meaning in rows:
        if api_key:
            refined = openai_refine_entry(raw_term, raw_meaning, api_key)
        else:
            refined = heuristic_refine_entry(raw_term, raw_meaning)

        cursor.execute("""
            UPDATE lexicon
            SET term = ?, meaning = ?, region = ?, category = ?, example = ?, phonetic = ?
            WHERE id = ?
        """, (
            refined["term"],
            refined["meaning"],
            refined["region"],
            refined["category"],
            refined.get("example", f"من القاموس: {refined['meaning'][:100]}"),
            refined.get("phonetic", refined["term"]),
            row_id
        ))
        updated_count += 1

        if updated_count % 500 == 0:
            conn.commit()
            print(f"  Processed {updated_count}/{len(rows)} entries...")

    conn.commit()
    conn.close()
    print(f"✅ Successfully refined {updated_count} OCR entries in database ({db_path})!")

def main():
    parser = argparse.ArgumentParser(description="Refine raw OCR dictionary entries in database")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to refine")
    parser.add_argument("--no-ai", action="store_true", help="Force rule-based heuristic refiner without API calls")
    args = parser.parse_args()

    refine_ocr_database(limit=args.limit, use_ai=not args.no_ai)

if __name__ == "__main__":
    main()
