"""Unit tests for Arabic PDF searchable converter utility."""

import os
import json
import pytest
import pymupdf
from pathlib import Path

from scripts.make_pdf_searchable import (
    convert_pdf_to_searchable,
    prepare_arabic_for_pdf,
    load_checkpoint,
    save_checkpoint
)


@pytest.fixture
def temp_pdf_file(tmp_path):
    """Creates a sample PDF with visual elements."""
    pdf_path = tmp_path / "sample_input.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(pymupdf.Rect(50, 50, 545, 792), color=(0.1, 0.3, 0.5), fill=(0.95, 0.95, 0.95))
    page.insert_text((100, 100), "اختبار القاموس العربي", fontsize=20, fontname="helv")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_prepare_arabic_for_pdf():
    text = "قاموس العامية"
    shaped = prepare_arabic_for_pdf(text)
    assert isinstance(shaped, str)
    assert len(shaped) > 0


def test_checkpoint_save_and_load(tmp_path):
    checkpoint_file = tmp_path / "checkpoint.json"
    save_checkpoint(checkpoint_file, 15, "in.pdf", "out.pdf")

    assert checkpoint_file.exists()
    loaded_page = load_checkpoint(checkpoint_file)
    assert loaded_page == 15


def test_convert_pdf_to_searchable(temp_pdf_file, tmp_path):
    output_pdf = str(tmp_path / "searchable_output.pdf")
    checkpoint_file = str(tmp_path / "test_cp.json")

    convert_pdf_to_searchable(
        input_pdf=temp_pdf_file,
        output_pdf=output_pdf,
        dpi=150,
        checkpoint_file=checkpoint_file
    )

    assert os.path.exists(output_pdf)
    assert os.path.getsize(output_pdf) > 0

    # Verify visual layout page dimensions match original
    src_doc = pymupdf.open(temp_pdf_file)
    out_doc = pymupdf.open(output_pdf)

    assert len(out_doc) == len(src_doc)
    assert out_doc[0].rect.width == src_doc[0].rect.width
    assert out_doc[0].rect.height == src_doc[0].rect.height

    # Verify checkpoint saved
    assert os.path.exists(checkpoint_file)
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        cp_data = json.load(f)
        assert cp_data["last_processed_page"] == 1

    src_doc.close()
    out_doc.close()
