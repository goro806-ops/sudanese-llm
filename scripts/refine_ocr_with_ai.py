#!/usr/bin/env python3
"""
AI & Domain Knowledge OCR Dictionary Refiner Script
--------------------------------------------------
Refines raw "blind" OCR text extracted from Prof. Awn Al-Sharif Qasim's
scanned reference book "قاموس العامية السودانية" in `sudanese_lexicon.db`:

Book-Specific Domain Knowledge & Conventions:
1. Dictionary Abbreviations:
   - (س) -> سودانية / عامية سودانية (Sudanese Colloquial term)
   - (ف) -> فصيح / أصل عربي فصيح (Classical Arabic root)
   - (م) -> مثل شعبي سوداني (Sudanese Folk Proverb)
   - (ج) -> الجمع (Plural form)
   - (ش) -> شعر / دوبيت / مسدار (Folk Poetry / Dobait citation)
   - (ع) -> عبارة شعبية / تعبير (Popular phrase / Expression)
   - (ر) -> رطانة / لغة نوبية (Nubian / Rotana language origin)
   - (ك) -> كشاف / إحالة لجذر آخر (Dictionary cross-reference)
   - (ب) -> بجاوية / لغة البجا (Beja language origin)
   - (غرب) / (دارفور) -> إقليم دارفور والغرب
   - (كردفان) -> إقليم كردفان
   - (شمال) / (ن) -> الشمال والنوبية
   - (بطانة) -> إقليم البطانة
   - (شرق) -> شرق السودان

2. Structural Cleansing:
   - Separates headwords (المفردة) from definition body.
   - Cleans OCR scanning noise (isolated non-Arabic characters, random punctuation like ^, *, ~, %, column borders).
   - Identifies examples and proverbs for structured database columns.
   - Determines dialect region (khartoum, darfur, kordofan, eastern, northern).
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

# Comprehensive Sudanese Dictionary Abbreviations
ABBREVIATION_REPLACEMENTS = [
    (r'\(\s*س\s*\)', ' [عامية سودانية] '),
    (r'\(\s*ف\s*\)', ' [أصل فصيح] '),
    (r'\(\s*م\s*\)', ' [مثل شعبي] '),
    (r'\(\s*ج\s*\)', ' [الجمع] '),
    (r'\(\s*ش\s*\)', ' [شعر ودوبيت] '),
    (r'\(\s*ع\s*\)', ' [تعبير شعبي] '),
    (r'\(\s*ر\s*\)', ' [لغة نوبية / رطانة] '),
    (r'\(\s*ك\s*\)', ' [إحالة معجمية] '),
    (r'\(\s*ب\s*\)', ' [لغة بجاوية] '),
    (r'\(\s*غرب\s*\)', ' [إقليم غرب السودان] '),
    (r'\(\s*دارفور\s*\)', ' [إقليم دارفور] '),
    (r'\(\s*كردفان\s*\)', ' [إقليم كردفان] '),
    (r'\(\s*شمال\s*\)', ' [إقليم شمال السودان] '),
    (r'\(\s*بطانة\s*\)', ' [إقليم البطانة] '),
    (r'\(\s*شرق\s*\)', ' [إقليم شرق السودان] '),
    (r'\(\s*ن\s*\)', ' [لغة نوبية] '),
]

REGION_KEYWORDS = {
    "دارفور": "darfur",
    "غرب": "darfur",
    "كردفان": "kordofan",
    "بجا": "eastern",
    "بجاوية": "eastern",
    "شرق": "eastern",
    "دنقلا": "northern",
    "نوبية": "northern",
    "رطانة": "northern",
    "شمال": "northern",
    "شايقية": "northern",
    "رباطاب": "northern",
    "بطانة": "khartoum",
    "خرطوم": "khartoum",
    "جعليين": "khartoum",
}

BOOK_DOMAIN_SYSTEM_PROMPT = """أنت خبير ومعجمي متخصص في تحقيق وترميم "قاموس العامية السودانية" للبروفيسور عون الشريف قاسم.
تقوم بتنظيف وتحليل النصوص المستخرجة بتقنية OCR من القاموس وتحويلها إلى بيانات معجمية دقيقة 100%.

قواعد المعجم السوداني المعتمدة:
1. المفردة (term): اسم الكلمة أو الجذر الأساسي مع ضبط الحركات إن وجدت.
2. الاختصارات المعجمية:
   - (س) تعني: عامية سودانية
   - (ف) تعني: أصل عربي فصيح
   - (م) تعني: مثل شعبي سوداني
   - (ج) تعني: صيغة الجمع
   - (ش) تعني: شاهد شعر شعبى / دوبيت / مسدار
   - (ع) تعني: تعبير أو عبارة شعبية
   - (ر) تعني: رطانة / لغة نوبية
   - (ب) تعني: لغة بجاوية
   - (غرب / دارفور / كردفان / شمال / بطانة / شرق) تعني تحديد الإقليم أو المنطقة.
3. التصفية والتنظيف:
   - إزالة رموز OCR المشوهة مثل (^, *, %, ~, _, <, >).
   - التمييز بين المعنى والشرح (meaning) والشواهد والأمثال (example).
4. تحديد الإقليم (region):
   - اختر من بين: khartoum, darfur, kordofan, eastern, northern.
5. التصنيف (category):
   - اختر التصنيف الدقيق (مثل: مثل_ومقولة, ترحيب / تحية, مفردات وشرح ثقافي, فنون وتراث, أواني ومأكولات).
"""

def clean_ocr_text(raw_text: str) -> str:
    """Clean scanning artifacts, non-Arabic noise symbols, and page numbers."""
    if not raw_text:
        return ""

    # Remove random OCR noise symbols
    cleaned = re.sub(r'[\^~\*#_%<>{}\|\\\]\[]', ' ', raw_text)
    # Remove isolated Latin characters
    cleaned = re.sub(r'\b[a-zA-Z]\b', ' ', cleaned)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def expand_dictionary_abbreviations(text: str) -> str:
    """Expand all parenthetical codes from Awn Al-Sharif Qasim's dictionary."""
    for pattern, replacement in ABBREVIATION_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text

def detect_region(text: str) -> str:
    """Detect regional dialect domain from text content."""
    for kw, reg in REGION_KEYWORDS.items():
        if kw in text:
            return reg
    return "khartoum"

def heuristic_refine_entry(raw_term: str, raw_meaning: str) -> dict:
    """Offline heuristic refiner using full dictionary domain rules."""
    cleaned_term = clean_ocr_text(raw_term)
    cleaned_meaning = clean_ocr_text(raw_meaning)

    # Clean headword
    cleaned_term = re.sub(r'^[^\w\s]+', '', cleaned_term)
    words = cleaned_term.split()
    term = words[0] if words else "مفردة"

    # Expand dictionary codes
    expanded_meaning = expand_dictionary_abbreviations(cleaned_meaning)
    region = detect_region(expanded_meaning)

    # Determine category
    category = "مفردات وشرح ثقافي"
    if "[مثل شعبي]" in expanded_meaning or "مثل" in expanded_meaning:
        category = "مثل_ومقولة"
    elif "[شعر ودوبيت]" in expanded_meaning or "دوبيت" in expanded_meaning or "مسدار" in expanded_meaning:
        category = "فنون وتراث"
    elif "ترحيب" in expanded_meaning or "سلام" in expanded_meaning:
        category = "ترحيب / تحية"

    example = f"من قاموس العامية: {expanded_meaning[:140]}"

    return {
        "term": term,
        "meaning": expanded_meaning,
        "region": region,
        "category": category,
        "example": example,
        "phonetic": term
    }

def openai_refine_entry(raw_term: str, raw_meaning: str, api_key: str) -> dict:
    """Use GPT with full dictionary domain knowledge to parse raw OCR text."""
    try:
        import urllib.request
        prompt = f"""قم بتحقيق وتنظيف النص المعجمي التالي من قاموس العامية السودانية:

المفردة الخام: {raw_term}
النص الخام: {raw_meaning}

قم بالتحليل وإرجاع JSON بالصيغة التالية فقط:
{{"term": "المفردة المنقحة", "meaning": "الشرح مع فك الاختصارات (س، ف، م، ج، ش، ر، ب)", "region": "إقليم (khartoum, darfur, kordofan, eastern, northern)", "category": "التصنيف", "example": "المثل أو الشاهد إن وجد", "phonetic": "المفردة"}}
"""
        req_data = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": BOOK_DOMAIN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
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
        print(f"⚠️ API error ({err}), falling back to domain heuristic refiner...")
        return heuristic_refine_entry(raw_term, raw_meaning)

def refine_ocr_database(db_path: Path = DB_PATH, limit: int = None, use_ai: bool = True):
    """Refine raw OCR dictionary records in SQLite database."""
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

    print(f"🔄 Refining {len(rows)} raw dictionary entries using domain knowledge rules...")

    api_key = os.getenv("OPENAI_API_KEY") if use_ai else None
    if use_ai and not api_key:
        print("ℹ️ No OPENAI_API_KEY set. Running with domain-aware heuristic parser...")

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
    print(f"✅ Successfully refined {updated_count} OCR entries in {db_path}!")

def main():
    parser = argparse.ArgumentParser(description="Domain Knowledge Dictionary Refiner")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows to refine")
    parser.add_argument("--no-ai", action="store_true", help="Force domain heuristic refiner without API calls")
    args = parser.parse_args()

    refine_ocr_database(limit=args.limit, use_ai=not args.no_ai)

if __name__ == "__main__":
    main()
