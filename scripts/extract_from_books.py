"""Semantic Book Extractor for Sudanese Dialect & Cultural Books.
Parses full book pages with PyMuPDF/EasyOCR, performs semantic paragraph analysis,
and extracts headwords, roots/origins, detailed descriptions, proverbs, and regional contexts.
"""
import os
import re
import sqlite3
import json
from pathlib import Path
import pymupdf
import easyocr
from vector_db.vector_store import VectorStore

DB_PATH = Path("data/processed/sudanese_lexicon.db")
VECTOR_STORE_PATH = Path("data/processed/vector_store.json")

def init_ocr():
    """Initialize EasyOCR reader for Arabic."""
    print("Initializing EasyOCR Arabic reader...")
    return easyocr.Reader(['ar'])

def parse_semantic_paragraph(paragraph: str, page_num: int) -> list:
    """Analyze full paragraph text and extract semantic dictionary & cultural entities."""
    entries = []
    clean_p = re.sub(r'\s+', ' ', paragraph).strip()

    if len(clean_p) < 20:
        return entries

    # Identify potential regional references in text
    regions = []
    region_map = {
        "خرطوم": "khartoum", "أم درمان": "khartoum", "بحري": "khartoum",
        "دارفور": "darfur", "فاشر": "darfur", "نيالا": "darfur",
        "كردفان": "kordofan", "أبيض": "kordofan", "بارا": "kordofan",
        "شرق": "eastern", "بجا": "eastern", "بورتسودان": "eastern", "كسلا": "eastern",
        "شمالية": "northern", "دنقلا": "northern", "نوبة": "northern", "محس": "northern", "سكوت": "northern"
    }

    for key, reg in region_map.items():
        if key in clean_p:
            regions.append(reg)
    primary_region = regions[0] if regions else "khartoum"

    # Identify headwords (words often appearing at beginning of dictionary sections or before colon/dash)
    words = clean_p.split()
    headword = words[0] if words else ""
    headword = re.sub(r'[^\w\s]', '', headword)

    # Detect etymology/origin hints (e.g. نوبي، فصيح، بجاوي، تركي)
    origin = "عامية سودانية"
    if "نوبي" in clean_p or "رطانة" in clean_p:
        origin = "أصل نوبي / رطانة"
    elif "فصيح" in clean_p or "فصحى" in clean_p or "أصلها" in clean_p:
        origin = "أصل عربي فصيح"
    elif "بجا" in clean_p:
        origin = "أصل بجاوي"

    # Detect proverbs / poetic illustrations
    is_proverb = "مثل" in clean_p or "شعر" in clean_p or "يقال" in clean_p
    category = "مثل_ومقولة" if is_proverb else "مفردات_وشرح_ثقافي"

    if len(headword) >= 2 and len(clean_p) >= 25:
        entries.append({
            "term": headword,
            "meaning": clean_p,
            "region": primary_region,
            "category": category,
            "example": f"من كتاب قاموس العامية (ص {page_num}): {clean_p[:150]}...",
            "phonetic": headword
        })

    return entries

def extract_book_semantics(pdf_path: str, start_page: int = 25, end_page: int = 35):
    """Process PDF page range, extract full paragraphs, and parse semantic entities."""
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} not found.")
        return []

    doc = pymupdf.open(pdf_path)
    reader = init_ocr()
    all_extracted = []

    os.makedirs("tmp_book_pages", exist_ok=True)

    for p in range(start_page, min(end_page + 1, len(doc))):
        print(f"Reading page {p}/{len(doc)}...")
        page = doc[p]
        pix = page.get_pixmap(dpi=150)
        img_path = f"tmp_book_pages/page_{p}.png"
        pix.save(img_path)

        ocr_lines = reader.readtext(img_path, detail=0)
        full_page_text = "\n".join(ocr_lines)

        # Split into logical paragraphs
        paragraphs = full_page_text.split("\n\n") if "\n\n" in full_page_text else [full_page_text]

        for para in paragraphs:
            entries = parse_semantic_paragraph(para, p)
            all_extracted.extend(entries)

        if os.path.exists(img_path):
            os.remove(img_path)

    return all_extracted

def save_to_db_and_vectorstore(entries):
    """Store extracted semantic entities into SQLite DB and update Vector Store."""
    if not entries or not DB_PATH.exists():
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    saved = 0

    for e in entries:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO lexicon (term, meaning, region, category, example, phonetic)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (e["term"], e["meaning"], e["region"], e["category"], e["example"], e["phonetic"]))
            if cursor.rowcount > 0:
                saved += 1
        except Exception as err:
            print(f"Error inserting {e['term']}: {err}")

    conn.commit()
    conn.close()

    # Re-index Vector Store
    vs = VectorStore()
    vs.load_raw_dataset()
    print(f"Saved {saved} semantic entries to DB and re-indexed vector store ({len(vs.documents)} total docs)!")
    return saved

def main():
    pdf_path = "downloaded_book_1.pdf"
    print("Running semantic book extraction pipeline...")
    entries = extract_book_semantics(pdf_path, start_page=20, end_page=30)
    print(f"Extracted {len(entries)} rich semantic entries from book.")
    if entries:
        print("\n--- SAMPLE EXTRACTED ENTRY ---")
        print(json.dumps(entries[0], ensure_ascii=False, indent=2))
        print("------------------------------\n")
    save_to_db_and_vectorstore(entries)

if __name__ == "__main__":
    main()
