"""Script to extract Sudanese dialect terms from scanned PDF books using EasyOCR and populate the lexicon database and vector store."""
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

def init_easyocr():
    print("Initializing EasyOCR for Arabic...")
    return easyocr.Reader(['ar'])

def process_pdf_pages(pdf_path: str, start_page: int = 25, end_page: int = 40):
    """Render PDF pages to images and extract text using EasyOCR."""
    doc = pymupdf.open(pdf_path)
    reader = init_easyocr()
    extracted_entries = []

    os.makedirs("tmp_book_pages", exist_ok=True)

    for page_num in range(start_page, min(end_page + 1, len(doc))):
        print(f"Processing page {page_num}/{len(doc)}...")
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img_path = f"tmp_book_pages/page_{page_num}.png"
        pix.save(img_path)

        ocr_results = reader.readtext(img_path, detail=0)
        page_text = " ".join(ocr_results)

        # Parse potential dictionary lines (Arabic term followed by definition)
        # Dictionary format often has terms in quotes or followed by colon/dash
        terms = parse_dictionary_text(page_text)
        extracted_entries.extend(terms)

        if os.path.exists(img_path):
            os.remove(img_path)

    return extracted_entries

def parse_dictionary_text(text: str):
    """Extract Sudanese dialect terms and definitions from OCR text."""
    entries = []

    # Clean text
    clean_text = re.sub(r'\s+', ' ', text)

    # Split into sentences or clauses
    parts = re.split(r'[.:؛\n]', clean_text)
    for part in parts:
        part = part.strip()
        if len(part) < 10:
            continue
        
        # Simple heuristic: terms often start sentences or are 1-3 words
        words = part.split()
        if len(words) >= 3:
            term = words[0]
            # Strip punctuation
            term = re.sub(r'[^\w\s]', '', term)
            meaning = " ".join(words[1:])
            
            if len(term) >= 2 and len(meaning) >= 8:
                entries.append({
                    "term": term,
                    "meaning": meaning,
                    "region": "khartoum",
                    "category": "dictionary_book",
                    "example": f"من قاموس اللهجة العامية: {part}",
                    "phonetic": term
                })

    return entries

def save_entries_to_db(entries):
    """Insert newly extracted dictionary entries into SQLite DB."""
    if not entries or not DB_PATH.exists():
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted_count = 0
    for entry in entries:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO lexicon (term, meaning, region, category, example, phonetic)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entry["term"],
                entry["meaning"],
                entry["region"],
                entry["category"],
                entry["example"],
                entry["phonetic"]
            ))
            if cursor.rowcount > 0:
                inserted_count += 1
        except Exception as e:
            pass

    conn.commit()
    conn.close()
    return inserted_count

def update_vector_store():
    """Rebuild vector store index to include new book lexicon items."""
    vs = VectorStore()
    vs.load_raw_dataset()
    print(f"Vector store successfully re-indexed with {len(vs.documents)} documents!")

def main():
    pdf_path = "downloaded_book_1.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF file {pdf_path} not found!")
        return

    print("Starting automated book ingestion...")
    entries = process_pdf_pages(pdf_path, start_page=25, end_page=35)
    print(f"Extracted {len(entries)} candidate lexicon entries from PDF.")

    inserted = save_entries_to_db(entries)
    print(f"Successfully saved {inserted} new terms into {DB_PATH}.")

    update_vector_store()

if __name__ == "__main__":
    main()
