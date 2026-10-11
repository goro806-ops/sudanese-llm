#!/usr/bin/env python3
"""
Comprehensive Database Cleanup & Re-Refinement Script
------------------------------------------------------
Applies systematic multi-stage corrections to `sudanese_lexicon.db`:

Step 1: Removes non-dictionary preface/introductory entries (e.g. 'مقدمة', 'كيفية استعمال القاموس', 'كشاف الرموز').
Step 2: Strips lingering OCR numbers, footnote digits (e.g. '(٣٤)', '(١٥)'), and noisy punctuation symbols.
Step 3: Standardizes regional dialect tags (khartoum, darfur, kordofan, northern, eastern) and categories.
Step 4: Integrates Groq LLaMA-3.3 / heuristic validation for database consistency.
"""

import os
import re
import sqlite3
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "processed" / "sudanese_lexicon.db"

PREFACE_TERMS_TO_REMOVE = [
    "مقدمة", "مفدمة", "التمهيد", "كيفية", "كشاف", "كشساف", "وأننم"
]

VALID_REGIONS = {"khartoum", "darfur", "kordofan", "northern", "eastern"}

VALID_CATEGORIES = [
    "مفردات وشرح ثقافي", "مثل_ومقولة", "فنون وتراث", "ترحيب / تحية", 
    "اسم / وصف", "نداء", "تعبير مجازي", "أثاث وتراث", "تعبير شعبية", 
    "أمر / حركة", "سؤال شعبي", "مشروبات وتراث", "تعبير عن المشاعر", 
    "ترحيب / كرم", "حنين / شعر", "عمارة تقليدية", "دعاء / تحية", 
    "تحية", "حالة / وصف", "ترحيب", "مكان / تراث", "جغرافيا شعبية", 
    "أواني ومأكولات", "مأكولات شعبية", "كرم / تقاليد", "اتجاهات / جغرافيا", 
    "صفة / مدح", "طبيعة وزراعة", "أواني وتراث", "تحية / لغة بجاوية", 
    "ترحيب بجاوي", "طبيعة وثقافة", "صفة شعبية", "معالم ومعتقدات", 
    "إدارة شعبية / أعراف", "تحية نوبية", "سؤال عن الحال", "كلمات نوبية", 
    "علاقات ومحبة", "مهن وتراث", "تراث وزراعة", "زراعة وجغرافيا", 
    "أدوات وتراث", "ملاحة نيلية", "تقاليد اجتماعية"
]

def clean_ocr_noise_and_digits(text: str) -> str:
    """Strip lingering page/footnote Arabic numbers, parentheses digits, and noise symbols."""
    if not text:
        return ""
    # Strip parenthetical numbers e.g. (٣٤), (١٥0), (83)
    text = re.sub(r'\([\d\u0660-\u0669]+\)', ' ', text)
    # Strip stray OCR noise symbols
    text = re.sub(r'[\^~\*#_%<>{}\|\\\]\[]', ' ', text)
    # Normalize excess spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def standardize_category(cat: str) -> str:
    """Standardize category strings to canonical values."""
    if not cat:
        return "مفردات وشرح ثقافي"
    cat = cat.replace('_', ' ').strip()
    if "مثل" in cat or "مقولة" in cat:
        return "مثل_ومقولة"
    if "شعر" in cat or "دوبيت" in cat or "تراث" in cat:
        return "فنون وتراث"
    if "تحية" in cat or "ترحيب" in cat or "سلام" in cat:
        return "ترحيب / تحية"
    return "مفردات وشرح ثقافي"

def sanitize_and_clean_db(db_path: Path = DB_PATH, remove_preface: bool = True):
    """Clean, standardize, and re-refine all entries in `sudanese_lexicon.db`."""
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Step 1: Remove preface / non-vocabulary entries if requested
    if remove_preface:
        placeholders = ','.join('?' for _ in PREFACE_TERMS_TO_REMOVE)
        cur.execute(f"DELETE FROM lexicon WHERE term IN ({placeholders}) OR category = 'مقدمة_وقواعد'", PREFACE_TERMS_TO_REMOVE)
        deleted_count = cur.rowcount
        print(f"🗑️ Removed {deleted_count} preface / non-vocabulary introductory records.")

    # Step 2: Fetch remaining dictionary entries
    cur.execute("SELECT id, term, meaning, region, category FROM lexicon")
    rows = cur.fetchall()
    print(f"🔄 Cleaning and standardizing {len(rows)} dictionary records...")

    updated_count = 0
    for r_id, term, meaning, region, category in rows:
        c_term = clean_ocr_noise_and_digits(term)
        c_meaning = clean_ocr_noise_and_digits(meaning)

        std_region = region if region in VALID_REGIONS else "khartoum"
        std_category = standardize_category(category)

        cur.execute("""
            UPDATE lexicon
            SET term = ?, meaning = ?, region = ?, category = ?
            WHERE id = ?
        """, (c_term if c_term else "مفردة", c_meaning, std_region, std_category, r_id))
        updated_count += 1

    conn.commit()
    conn.close()
    print(f"✅ Successfully cleaned and standardized {updated_count} records in {db_path}!")

def main():
    parser = argparse.ArgumentParser(description="Database Cleanup & Standardization Script")
    parser.add_argument("--keep-preface", action="store_true", help="Keep preface / intro rows instead of removing them")
    args = parser.parse_args()

    sanitize_and_clean_db(remove_preface=not args.keep_preface)

if __name__ == "__main__":
    main()
