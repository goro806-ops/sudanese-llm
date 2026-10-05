#!/usr/bin/env python3
"""
AI & Multi-Provider OCR Dictionary Refiner Script
--------------------------------------------------
Refines raw "blind" OCR text extracted from Prof. Awn Al-Sharif Qasim's
scanned reference book "قاموس العامية السودانية" in `sudanese_lexicon.db`.

Supported Free AI Providers:
1. Groq API (GROQ_API_KEY) -> Free fast LLaMA-3.3-70B / Qwen2.5-72B
2. Google Gemini API (GEMINI_API_KEY) -> Free gemini-1.5-flash / gemini-2.0-flash
3. Hugging Face Inference API (HF_TOKEN) -> Free open-weights LLMs
4. OpenAI API (OPENAI_API_KEY) -> GPT-4o-mini
5. Offline Rule-Based Heuristic Parser (Default fallback when no key is set)

Book-Specific Conventions & Abbreviation Expansions:
   - (س) -> عامية سودانية
   - (ف) -> أصل فصيح
   - (م) -> مثل شعبي
   - (ج) -> الجمع
   - (ش) -> شعر ودوبيت
   - (ع) -> تعبير شعبي
   - (ر) -> لغة نوبية / رطانة
   - (ب) -> لغة بجاوية
"""

import os
import re
import sys
import json
import sqlite3
import argparse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "processed" / "sudanese_lexicon.db"

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
تقوم بتنظيف وتحليل النصوص المستخرجة بتقنية OCR من القاموس وتحويلها إلى بيانات معجمية دقيقة.

قواعد المعجم السوداني المعتمدة:
1. المفردة (term): اسم الكلمة أو الجذر الأساسي.
2. الاختصارات المعجمية: (س -> عامية سودانية), (ف -> أصل فصيح), (م -> مثل شعبي), (ج -> الجمع), (ش -> شعر ودوبيت), (ع -> تعبير شعبي), (ر -> رطانة نوبية), (ب -> بجاوية).
3. إزالة رموز OCR المشوهة (^, *, %, ~).
4. تحديد الإقليم (region): khartoum, darfur, kordofan, eastern, northern.
5. التصنيف (category): مثل_ومقولة, ترحيب / تحية, مفردات وشرح ثقافي, فنون وتراث.
"""

def clean_ocr_text(raw_text: str) -> str:
    """Clean scanning artifacts and non-Arabic noise symbols."""
    if not raw_text:
        return ""
    cleaned = re.sub(r'[\^~\*#_%<>{}\|\\\]\[]', ' ', raw_text)
    cleaned = re.sub(r'\b[a-zA-Z]\b', ' ', cleaned)
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

    cleaned_term = re.sub(r'^[^\w\s]+', '', cleaned_term)
    words = cleaned_term.split()
    term = words[0] if words else "مفردة"

    expanded_meaning = expand_dictionary_abbreviations(cleaned_meaning)
    region = detect_region(expanded_meaning)

    category = "مفردات وشرح ثقافي"
    if "[مثل شعبي]" in expanded_meaning or "مثل" in expanded_meaning:
        category = "مثل_ومقولة"
    elif "[شعر ودوبيت]" in expanded_meaning or "دوبيت" in expanded_meaning:
        category = "فنون وتراث"
    elif "ترحيب" in expanded_meaning or "سلام" in expanded_meaning:
        category = "ترحيب / تحية"

    return {
        "term": term,
        "meaning": expanded_meaning,
        "region": region,
        "category": category,
        "example": f"من قاموس العامية: {expanded_meaning[:140]}",
        "phonetic": term
    }

def groq_refine_entry(raw_term: str, raw_meaning: str, api_key: str) -> dict:
    """Refine entry using Groq Free API (LLaMA-3.3-70B / Qwen2.5-72B)."""
    prompt = f"المفردة الخام: {raw_term}\nالنص الخام: {raw_meaning}\nأرجع JSON فقط: {{\"term\": \"...\", \"meaning\": \"...\", \"region\": \"...\", \"category\": \"...\", \"example\": \"...\", \"phonetic\": \"...\"}}"
    req_data = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": BOOK_DOMAIN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=req_data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return json.loads(result["choices"][0]["message"]["content"])

def gemini_refine_entry(raw_term: str, raw_meaning: str, api_key: str) -> dict:
    """Refine entry using Google Gemini Free API (gemini-1.5-flash)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    prompt = f"{BOOK_DOMAIN_SYSTEM_PROMPT}\n\nالمفردة الخام: {raw_term}\nالنص الخام: {raw_meaning}\nأرجع JSON بنفس المفاتيح (term, meaning, region, category, example, phonetic) فقط."

    req_data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }).encode("utf-8")

    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        text_resp = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text_resp)

def hf_refine_entry(raw_term: str, raw_meaning: str, token: str) -> dict:
    """Refine entry using Hugging Face Free Inference API."""
    url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions"
    prompt = f"المفردة الخام: {raw_term}\nالنص الخام: {raw_meaning}\nأرجع JSON فقط: {{\"term\": \"...\", \"meaning\": \"...\", \"region\": \"...\", \"category\": \"...\", \"example\": \"...\", \"phonetic\": \"...\"}}"
    req_data = json.dumps({
        "messages": [
            {"role": "system", "content": BOOK_DOMAIN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        text_content = result["choices"][0]["message"]["content"]
        match = re.search(r'\{.*\}', text_content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text_content)

def openai_refine_entry(raw_term: str, raw_meaning: str, api_key: str) -> dict:
    """Refine entry using OpenAI API."""
    prompt = f"المفردة الخام: {raw_term}\nالنص الخام: {raw_meaning}\nأرجع JSON فقط بنفس المفاتيح."
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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return json.loads(result["choices"][0]["message"]["content"])

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

    print(f"🔄 Refining {len(rows)} raw dictionary entries...")

    # Detect provider based on environment variables
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    provider = "heuristic"
    if use_ai:
        if groq_key:
            provider = "groq"
            print("🚀 Using GROQ Free API (LLaMA-3.3-70B)...")
        elif gemini_key:
            provider = "gemini"
            print("✨ Using Google Gemini Free API...")
        elif hf_token:
            provider = "hf"
            print("🤗 Using Hugging Face Free Inference API...")
        elif openai_key:
            provider = "openai"
            print("⚡ Using OpenAI API...")
        else:
            print("ℹ️ No AI API keys set. Running with domain-aware heuristic parser...")

    updated_count = 0
    for row_id, raw_term, raw_meaning in rows:
        try:
            if provider == "groq":
                refined = groq_refine_entry(raw_term, raw_meaning, groq_key)
            elif provider == "gemini":
                refined = gemini_refine_entry(raw_term, raw_meaning, gemini_key)
            elif provider == "hf":
                refined = hf_refine_entry(raw_term, raw_meaning, hf_token)
            elif provider == "openai":
                refined = openai_refine_entry(raw_term, raw_meaning, openai_key)
            else:
                refined = heuristic_refine_entry(raw_term, raw_meaning)
        except Exception as err:
            refined = heuristic_refine_entry(raw_term, raw_meaning)

        cursor.execute("""
            UPDATE lexicon
            SET term = ?, meaning = ?, region = ?, category = ?, example = ?, phonetic = ?
            WHERE id = ?
        """, (
            refined.get("term", raw_term),
            refined.get("meaning", raw_meaning),
            refined.get("region", "khartoum"),
            refined.get("category", "مفردات وشرح ثقافي"),
            refined.get("example", f"من القاموس: {raw_meaning[:100]}"),
            refined.get("phonetic", raw_term),
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
    parser = argparse.ArgumentParser(description="Multi-Provider Dictionary Refiner")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows to refine")
    parser.add_argument("--no-ai", action="store_true", help="Force domain heuristic refiner without API calls")
    args = parser.parse_args()

    refine_ocr_database(limit=args.limit, use_ai=not args.no_ai)

if __name__ == "__main__":
    main()
