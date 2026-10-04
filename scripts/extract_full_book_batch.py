"""Batch Book Extraction Worker.
Processes large PDF dictionary books in page chunks using 300 DPI image rendering,
OpenCV contrast enhancement, and EasyOCR line-by-line parsing.
Runs safely in background or cloud environments (e.g., GitHub Actions, Kaggle, background server).
Auto-downloads the Google Drive PDF source if not present locally.
"""
import os
import sys
import argparse
import sqlite3
import json
import re
import urllib.request
from pathlib import Path

# Ensure repository root directory is in Python module search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pymupdf
import easyocr
import cv2
import numpy as np
from vector_db.vector_store import VectorStore

DB_PATH = ROOT_DIR / "data" / "processed" / "sudanese_lexicon.db"
CHECKPOINT_PATH = ROOT_DIR / "data" / "processed" / "extraction_checkpoint.json"
GDRIVE_PDF_ID = "1qyPmkzNgvyyJrWk2NMBD4w_aE6i9TIuG"
GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_PDF_ID}"

def save_checkpoint(page_num: int, pdf_path: str):
    """Save extraction progress checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_processed_page": page_num, "pdf_path": pdf_path}, f, indent=2)

def load_checkpoint():
    """Load last saved extraction progress checkpoint if available."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_processed_page", 0)
        except Exception as err:
            print(f"Warning: Failed to load checkpoint file: {err}")
    return 0

def download_source_pdf(target_path: str):
    """Download source PDF from Google Drive if not present locally."""
    if not os.path.exists(target_path):
        print(f"PDF source file '{target_path}' not found locally. Downloading from Google Drive...")
        try:
            urllib.request.urlretrieve(GDRIVE_DOWNLOAD_URL, target_path)
            print(f"Successfully downloaded '{target_path}' ({os.path.getsize(target_path)} bytes).")
        except Exception as err:
            print(f"Error downloading PDF from Google Drive: {err}")
            sys.exit(1)

def init_ocr():
    print("Initializing EasyOCR for Arabic (GPU/CPU)...")
    return easyocr.Reader(['ar'], gpu=False)

def preprocess_image(img_bytes):
    """Enhance scanned PDF page image contrast for OCR accuracy."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to clean background noise
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed

def process_page(page, reader, page_num):
    """Extract line-by-line dictionary entries from a single PDF page at 300 DPI."""
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    enhanced_img = preprocess_image(img_bytes)

    # Save temporary image for EasyOCR
    tmp_path = f"tmp_page_{page_num}.png"
    cv2.imwrite(tmp_path, enhanced_img)

    results = reader.readtext(tmp_path, detail=1, paragraph=False)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    entries = []
    current_entry_text = []

    for bbox, text, prob in results:
        text = text.strip()
        if not text or prob < 0.2:
            continue

        # If line looks like a headword or new section, start a new dictionary item
        words = text.split()
        if len(words) <= 3 and len(text) >= 2 and not text.isdigit():
            if current_entry_text:
                full_text = " ".join(current_entry_text)
                headword = current_entry_text[0] if current_entry_text else "مفردة"
                headword = re.sub(r'[^\w\s]', '', headword)
                if len(headword) >= 2 and len(full_text) >= 15:
                    entries.append({
                        "term": headword,
                        "meaning": full_text,
                        "region": "khartoum",
                        "category": "dictionary_book",
                        "example": f"من كتاب قاموس العامية (ص {page_num}): {full_text[:120]}...",
                        "phonetic": headword
                    })
            current_entry_text = [text]
        else:
            current_entry_text.append(text)

    if current_entry_text:
        full_text = " ".join(current_entry_text)
        headword = current_entry_text[0] if current_entry_text else "مفردة"
        headword = re.sub(r'[^\w\s]', '', headword)
        if len(headword) >= 2 and len(full_text) >= 15:
            entries.append({
                "term": headword,
                "meaning": full_text,
                "region": "khartoum",
                "category": "dictionary_book",
                "example": f"من كتاب قاموس العامية (ص {page_num}): {full_text[:120]}...",
                "phonetic": headword
            })

    return entries

def save_batch(entries):
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
        except Exception:
            pass

    conn.commit()
    conn.close()
    return saved

def main():
    parser = argparse.ArgumentParser(description="Full Book Batch Extraction Worker")
    parser.add_argument("--pdf-path", type=str, default="downloaded_book_1.pdf")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=1251)
    parser.add_argument("--resume", action="store_true", help="Resume from last saved checkpoint page")
    args = parser.parse_args()

    # Automatically download the Google Drive PDF source if missing
    download_source_pdf(args.pdf_path)

    doc = pymupdf.open(args.pdf_path)
    reader = init_ocr()
    total_doc_pages = len(doc)

    start_page = args.start_page
    if args.resume:
        last_page = load_checkpoint()
        if last_page > 0:
            start_page = last_page + 1
            print(f"🔄 Resuming extraction from checkpoint page {start_page}...")
        else:
            print("ℹ️ No previous checkpoint found. Starting from page 1.")

    start_p = max(0, start_page - 1)
    end_p = min(total_doc_pages, args.end_page)

    print(f"Starting batch extraction on '{args.pdf_path}' (Pages {start_p + 1} to {end_p} of {total_doc_pages})...")

    total_saved = 0
    for p in range(start_p, end_p):
        print(f"Processing page {p + 1}/{end_p}...")
        try:
            page_entries = process_page(doc[p], reader, p + 1)
            saved = save_batch(page_entries)
            total_saved += saved
            save_checkpoint(p + 1, args.pdf_path)
            print(f"  Extracted {len(page_entries)} entries ({saved} new entries saved). Checkpoint saved at page {p + 1}.")
        except Exception as err:
            print(f"  Error processing page {p + 1}: {err}")

    # Re-index Vector Store at the end
    vs = VectorStore()
    vs.load_raw_dataset()
    print(f"\nBatch Completed! Total new entries saved: {total_saved}. Vector store re-indexed ({len(vs.documents)} total docs).")

if __name__ == "__main__":
    main()
