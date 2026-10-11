"""Make PDF Searchable Utility.
Converts image-based or scanned Arabic PDF documents into searchable PDFs with
an invisible text OCR overlay while strictly preserving the original visual page formatting.
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Ensure repo root is in module path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pymupdf
import easyocr
import cv2
import numpy as np
import arabic_reshaper
import torch

DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "data" / "processed" / "searchable_pdf_checkpoint.json"

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/unifont/unifont.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
]


def find_unicode_font(custom_path: str = None) -> str:
    """Locate an available Unicode font supporting full range glyphs or custom path."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            return font_path
    for root, _, files in os.walk("/usr/share/fonts"):
        for file in files:
            if file.endswith((".ttf", ".otf")):
                return os.path.join(root, file)
    raise FileNotFoundError("No TTF/OTF Unicode font found on the system for PDF text embedding.")


def prepare_arabic_word_for_pdf(word: str) -> str:
    """Shape Arabic word glyphs and reverse byte ordering so PDF text streams store visual left-to-right char sequence."""
    if not word:
        return ""
    try:
        reshaped = arabic_reshaper.reshape(word)
        return reshaped[::-1]
    except Exception:
        return word


def prepare_arabic_line_for_pdf(line_text: str) -> str:
    """Shape individual words and order words Right-To-Left for accurate PDF text extraction and search."""
    if not line_text:
        return ""
    try:
        words = line_text.split()
        shaped_words = [prepare_arabic_word_for_pdf(w) for w in words]
        return " ".join(reversed(shaped_words))
    except Exception:
        return line_text


def save_checkpoint(checkpoint_path: Path, last_page: int, pdf_path: str, output_path: str):
    """Save conversion progress checkpoint to JSON file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "last_processed_page": last_page,
        "input_pdf": pdf_path,
        "output_pdf": output_path
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> int:
    """Load last completed page number from checkpoint file if available."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_processed_page", 0)
        except Exception as err:
            print(f"Warning: Failed to read checkpoint file {checkpoint_path}: {err}")
    return 0


def preprocess_page_image(pix_bytes: bytes) -> np.ndarray:
    """Apply contrast enhancement and adaptive thresholding to image bytes before OCR."""
    nparr = np.frombuffer(pix_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed


def convert_pdf_to_searchable(
    input_pdf: str,
    output_pdf: str,
    start_page: int = 1,
    end_page: int = None,
    dpi: int = 150,
    use_gpu: bool = False,
    resume: bool = False,
    checkpoint_file: str = None,
    font_path: str = None,
    save_interval: int = 10
):
    """Convert an input PDF to a searchable PDF with invisible text layer."""
    if not os.path.exists(input_pdf):
        raise FileNotFoundError(f"Input PDF file not found: {input_pdf}")

    if use_gpu is None or use_gpu is True:
        use_gpu = torch.cuda.is_available()

    cp_path = Path(checkpoint_file) if checkpoint_file else DEFAULT_CHECKPOINT_PATH
    active_font_path = find_unicode_font(font_path)
    print(f"Using font for PDF text embedding: {active_font_path}")

    src_doc = pymupdf.open(input_pdf)
    total_pages = len(src_doc)

    if end_page is None or end_page > total_pages:
        end_page = total_pages

    actual_start_page = start_page
    if resume:
        last_done = load_checkpoint(cp_path)
        if last_done >= end_page:
            print(f"🎉 Searchable PDF conversion already completed up to page {last_done}/{total_pages}!")
            src_doc.close()
            return
        elif last_done > 0:
            actual_start_page = max(start_page, last_done + 1)
            print(f"🔄 Resuming searchable PDF generation from page {actual_start_page}...")

    print(f"Initializing EasyOCR for Arabic (GPU={use_gpu})...")
    reader = easyocr.Reader(['ar'], gpu=use_gpu)

    out_doc = pymupdf.open()
    if os.path.exists(output_pdf) and actual_start_page > 1:
        try:
            existing_doc = pymupdf.open(output_pdf)
            out_doc.insert_pdf(existing_doc)
            existing_doc.close()
            print(f"Loaded existing output document with {len(out_doc)} pages for resume.")
        except Exception as err:
            print(f"Could not load existing output file to resume ({err}), starting fresh output document.")
            out_doc = pymupdf.open()

    print(f"Processing pages {actual_start_page} to {end_page} of {total_pages}...")

    font_bytes = open(active_font_path, "rb").read()

    for page_idx in range(actual_start_page - 1, end_page):
        page_num = page_idx + 1
        page = src_doc[page_idx]

        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.show_pdf_page(page.rect, src_doc, page_idx)

        pix = page.get_pixmap(dpi=dpi)
        pix_bytes = pix.tobytes("png")
        enhanced_img = preprocess_page_image(pix_bytes)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_img_path = tmp_file.name

        try:
            if enhanced_img is not None:
                cv2.imwrite(tmp_img_path, enhanced_img)
            else:
                with open(tmp_img_path, "wb") as f:
                    f.write(pix_bytes)

            results = reader.readtext(tmp_img_path, detail=1, paragraph=False)
        finally:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)

        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height

        new_page.insert_font(fontname="ufont", fontbuffer=font_bytes)

        for bbox, text, prob in results:
            text = text.strip()
            if not text or prob < 0.15:
                continue

            x_coords = [p[0] * scale_x for p in bbox]
            y_coords = [p[1] * scale_y for p in bbox]

            x0, x1 = max(0, min(x_coords)), min(page.rect.width, max(x_coords))
            y0, y1 = max(0, min(y_coords)), min(page.rect.height, max(y_coords))

            rect = pymupdf.Rect(x0, y0, x1, y1)
            formatted_text = prepare_arabic_line_for_pdf(text)
            fontsize = max(6, (y1 - y0) * 0.75)

            try:
                new_page.insert_textbox(
                    rect,
                    formatted_text,
                    fontname="ufont",
                    fontsize=fontsize,
                    render_mode=3  # 3 = Invisible text (searchable overlay)
                )
            except Exception:
                try:
                    new_page.insert_text(
                        (x0, y1),
                        formatted_text,
                        fontname="ufont",
                        fontsize=fontsize,
                        render_mode=3
                    )
                except Exception:
                    pass

        save_checkpoint(cp_path, page_num, input_pdf, output_pdf)
        print(f"  Page {page_num}/{end_page} processed")

        if (page_num % save_interval == 0) or (page_num == end_page):
            out_doc.save(output_pdf, incremental=False)
            print(f" Checkpoint and output PDF batch saved up to page {page_num}/{end_page}.")

    out_doc.close()
    src_doc.close()
    print(f"\n🎉 Searchable PDF conversion successfully finished: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Convert Arabic PDF into Searchable PDF while preserving layout.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input PDF file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output searchable PDF file path")
    parser.add_argument("--start-page", type=int, default=1, help="Starting page number (1-based)")
    parser.add_argument("--end-page", type=int, default=None, help="Ending page number")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for OCR page rendering (default: 150)")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for EasyOCR")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to custom checkpoint JSON file")
    parser.add_argument("--font-path", type=str, default=None, help="Path to TTF/OTF Unicode Arabic font")
    parser.add_argument("--save-interval", type=int, default=10, help="Page batch interval for file saving")

    args = parser.parse_args()

    convert_pdf_to_searchable(
        input_pdf=args.input,
        output_pdf=args.output,
        start_page=args.start_page,
        end_page=args.end_page,
        dpi=args.dpi,
        use_gpu=args.gpu,
        resume=args.resume,
        checkpoint_file=args.checkpoint,
        font_path=args.font_path,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
