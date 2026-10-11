#!/usr/bin/env python3
"""
Lexicon Reconstruction & OCR Correction Script
------------------------------------------------
Reconstructs fragmented, garbled, or line-broken OCR dictionary entries in `sudanese_lexicon.db`
into clean, coherent Sudanese Arabic dictionary entries.

Key Features:
1. Merges mid-sentence fragmented rows into logical dictionary entries.
2. Corrects reversed and corrupt OCR abbreviation brackets (e.g., `)س`, `) ف س (`, `) م (`).
3. Fixes common Arabic OCR corruptions (e.g., `ويقسولسون` -> `ويقولون`, `نقسول` -> `نقول`, `دطل` -> `رطل`).
4. Identifies true dictionary headwords (terms) and eliminates stopword / fragment headwords (`ف`, `في`, `من`, `انظر`, `أي`, etc.).
5. Standardizes regions and categories.
"""

import re
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "processed" / "sudanese_lexicon.db"

# STOPWORD TERMS THAT ARE NOT VALID HEADWORDS
STOPWORDS = {
    "ف", "في", "من", "عن", "على", "إلى", "أن", "إن", "أو", "ثم", "مع", "حتى",
    "قال", "تقول", "وقال", "تقول الودادعية", "انظر", "أم", "أي", "ولعلها", "وقد",
    "س", "فص", "فس", "مص", "مغ", "ق", "ب", "ر", "ك", "ج", "ش", "ع", "م",
    "وكان", "يكون", "فهو", "وهي", "وهو", "أو", "قد", "ما", "لا", "لم", "لن", "كما"
}

# REVERSED & STANDARD OCR ABBREVIATION REPLACEMENTS
ABBREVIATION_MAP = [
    # Reversed parentheses
    (r'\)\s*ف\s*س\s*\(?', ' [أصل فصيح وعامي] '),
    (r'\)\s*س\s*\(?', ' [عامية سودانية] '),
    (r'\)\s*ف\s*\(?', ' [أصل فصيح] '),
    (r'\)\s*م\s*\(?', ' [مثل شعبي] '),
    (r'\)\s*ج\s*\(?', ' [الجمع] '),
    (r'\)\s*ش\s*\(?', ' [شعر ودوبيت] '),
    (r'\)\s*ع\s*\(?', ' [تعبير شعبي] '),
    (r'\)\s*ر\s*\(?', ' [لغة نوبية / رطانة] '),
    (r'\)\s*ب\s*\(?', ' [لغة بجاوية] '),
    (r'\)\s*ك\s*\(?', ' [إحالة معجمية] '),
    (r'\)\s*مص\s*\(?', ' [مصرية / عبر مصر] '),
    (r'\)\s*مغ\s*\(?', ' [مغربية] '),
    (r'\)\s*ق\s*\(?', ' [قبطية] '),
    (r'\)\s*غرب\s*\(?', ' [إقليم غرب السودان] '),
    (r'\)\s*دارفور\s*\(?', ' [إقليم دارفور] '),
    (r'\)\s*كردفان\s*\(?', ' [إقليم كردفان] '),
    (r'\)\s*شمال\s*\(?', ' [إقليم شمال السودان] '),
    (r'\)\s*بادية\s*\(?', ' [أهل البادية] '),

    # Standard parentheses
    (r'\(\s*ف\s*س\s*\)', ' [أصل فصيح وعامي] '),
    (r'\(\s*س\s*\)', ' [عامية سودانية] '),
    (r'\(\s*ف\s*\)', ' [أصل فصيح] '),
    (r'\(\s*م\s*\)', ' [مثل شعبي] '),
    (r'\(\s*ج\s*\)', ' [الجمع] '),
    (r'\(\s*ش\s*\)', ' [شعر ودوبيت] '),
    (r'\(\s*ع\s*\)', ' [تعبير شعبي] '),
    (r'\(\s*ر\s*\)', ' [لغة نوبية / رطانة] '),
    (r'\(\s*ب\s*\)', ' [لغة بجاوية] '),
    (r'\(\s*ك\s*\)', ' [إحالة معجمية] '),
    (r'\(\s*مص\s*\)', ' [مصرية / عبر مصر] '),
    (r'\(\s*مغ\s*\)', ' [مغربية] '),
    (r'\(\s*ق\s*\)', ' [قبطية] '),
    (r'\(\s*غرب\s*\)', ' [إقليم غرب السودان] '),
    (r'\(\s*بادية\s*\)', ' [أهل البادية] '),
]

# COMMON ARABIC OCR TYPO FIXES
OCR_TYPO_FIXES = [
    (r'\bويقسولسون\b', 'ويقولون'),
    (r'\bنقسول\b', 'نقول'),
    (r'\bالسرجسل\b', 'الرجل'),
    (r'\bدطل\b', 'رطل'),
    (r'\bأخسذته\b', 'أخذته'),
    (r'\bمرعزنا\b', 'موزونة'),
    (r'\bفبل\b', 'قبل'),
    (r'\bالمطيفل\b', 'الطفل'),
    (r'\bالماجالسن\b', 'المجالس'),
    (r'\bشبيسه\b', 'شبيهه'),
    (r'\bاستشدمًر\b', 'استخدم'),
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
    "جزيرة": "khartoum",
}

def clean_ocr_noise(text: str) -> str:
    """Clean scanning artifacts and fix OCR typos."""
    if not text:
        return ""

    # Apply abbreviation mappings
    for pattern, repl in ABBREVIATION_MAP:
        text = re.sub(pattern, repl, text)

    # Apply typo fixes
    for pattern, repl in OCR_TYPO_FIXES:
        text = re.sub(pattern, repl, text)

    # Remove non-Arabic noise symbols except brackets & punctuation
    text = re.sub(r'[\^~\*#_%<>{}\|]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_headword(term: str, meaning: str) -> tuple[str, str]:
    """
    Extract clean dictionary headword from term and meaning.
    If term is a stopword or fragment, find the first valid Arabic noun/verb in the entry.
    """
    clean_t = re.sub(r'^[^\w\s]+', '', term).strip()
    words = clean_t.split()
    first_word = words[0] if words else ""

    # If term is a stopword or single letter, attempt headword extraction from meaning
    if first_word in STOPWORDS or len(first_word) <= 1:
        # Search for first significant word in meaning
        m_words = re.findall(r'[\u0621-\u064A]+', meaning)
        valid_words = [w for w in m_words if w not in STOPWORDS and len(w) > 1]
        if valid_words:
            headword = valid_words[0]
            # Strip the headword prefix from meaning if redundant
            clean_m = re.sub(r'^\s*' + re.escape(headword) + r'\s*', '', meaning).strip()
            return headword, clean_m if clean_m else meaning
        return "مفردة", meaning

    return first_word, meaning

def detect_region(text: str) -> str:
    """Detect region from text."""
    for kw, reg in REGION_KEYWORDS.items():
        if kw in text:
            return reg
    return "khartoum"

def reconstruct_database(db_path: Path = DB_PATH):
    """Process and reconstruct all database entries."""
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT id, term, meaning FROM lexicon ORDER BY id ASC")
    rows = c.fetchall()

    print(f"🔄 Processing and reconstructing {len(rows)} entries...")

    reconstructed = []
    current_entry = None

    for row_id, raw_term, raw_meaning in rows:
        cleaned_term = clean_ocr_noise(raw_term)
        cleaned_meaning = clean_ocr_noise(raw_meaning)

        # Skip preface or empty noise rows
        if cleaned_term in ["مقدمة", "التمهيد", "كشاف"] or len(cleaned_meaning) < 2:
            continue

        headword, refined_meaning = extract_headword(cleaned_term, cleaned_meaning)

        # Merge fragmented rows that belong to the previous entry
        if current_entry and (cleaned_term in STOPWORDS or len(cleaned_term) <= 1):
            current_entry["meaning"] += " " + cleaned_meaning
            current_entry["meaning"] = clean_ocr_noise(current_entry["meaning"])
            current_entry["region"] = detect_region(current_entry["meaning"])
        else:
            if current_entry:
                reconstructed.append(current_entry)

            region = detect_region(refined_meaning)
            category = "مفردات وشرح ثقافي"
            if "[مثل شعبي]" in refined_meaning or "مثل" in refined_meaning:
                category = "مثل_ومقولة"
            elif "[شعر ودوبيت]" in refined_meaning or "دوبيت" in refined_meaning:
                category = "فنون وتراث"

            current_entry = {
                "term": headword,
                "meaning": refined_meaning,
                "region": region,
                "category": category,
                "example": f"من قاموس العامية: {refined_meaning[:120]}",
                "phonetic": headword
            }

    if current_entry:
        reconstructed.append(current_entry)

    # Re-build lexicon table cleanly
    c.execute("DELETE FROM lexicon")
    c.execute("DELETE FROM sqlite_sequence WHERE name='lexicon'")

    print(f"✨ Writing {len(reconstructed)} reconstructed entries to database...")
    for entry in reconstructed:
        c.execute("""
            INSERT INTO lexicon (term, meaning, region, category, example, phonetic)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry["term"],
            entry["meaning"],
            entry["region"],
            entry["category"],
            entry["example"],
            entry["phonetic"]
        ))

    conn.commit()
    conn.close()
    print(f"✅ Database reconstruction complete! {len(reconstructed)} clean entries saved to {db_path}")

if __name__ == "__main__":
    reconstruct_database()
