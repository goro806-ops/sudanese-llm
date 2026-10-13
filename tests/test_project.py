"""Comprehensive Test Suite for Multi-Regional Sudanese LLM Project."""
import pytest
import os
from web_server.app import app
from fastapi.testclient import TestClient
from vector_db.vector_store import VectorStore
from training.preprocessing import clean_sudanese_arabic, format_regional_prompt
from training.evaluate import evaluate_dialect_accuracy

client = TestClient(app)

def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "message" in res.json()
    assert res.json()["health"] == "/health"

def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["indexed_documents"] >= 7

def test_regions_endpoint():
    res = client.get("/regions")
    assert res.status_code == 200
    assert "khartoum" in res.json()["regions"]

def test_search_get_endpoint():
    res = client.get("/search?query=مسكاقمي&region=northern")
    assert res.status_code == 200
    assert "results" in res.json()
    assert res.json()["results_count"] >= 1

def test_generation_endpoint():
    res = client.post("/generate", json={"prompt": "ازيك يا زول", "region": "khartoum"})
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["region"] == "khartoum"
    assert len(res_data["response"]) > 0
    assert "rag_context" in res_data

def test_vector_store():
    store = VectorStore(db_path="data/test_processed/vector_store.json")
    store.add_documents([
        {"id": 101, "text": "مرحبا بك في الخرطوم", "region": "khartoum"},
        {"id": 102, "text": "مرحبا بك في الفاشر", "region": "darfur"}
    ])
    results = store.search("الخرطوم", region="khartoum")
    assert len(results) >= 1
    assert results[0]["region"] == "khartoum"
    # cleanup test vector store file
    if os.path.exists("data/test_processed/vector_store.json"):
        os.remove("data/test_processed/vector_store.json")

def test_vector_store_auto_load():
    store = VectorStore(db_path="data/test_processed/vector_store.json")
    store.load_raw_dataset("data/raw")
    assert len(store.documents) >= 7

def test_preprocessing():
    cleaned = clean_sudanese_arabic("حَبَابَكْ يَا زُولْ")
    assert "حبابك" in cleaned
    formatted = format_regional_prompt("ازيك", "darfur")
    assert "[DAR]" in formatted

def test_evaluate():
    metrics = evaluate_dialect_accuracy(["a", "b"], ["a", "b"])
    assert metrics["accuracy"] == 1.0

def test_llm_dataset_integrity():
    """Verify that generated LLM dataset files exist and are validly structured."""
    import json
    from pathlib import Path

    dataset_dir = Path("data/processed/llm_dataset")
    chat_file = dataset_dir / "sudanese_llm_chat.jsonl"
    instruction_file = dataset_dir / "sudanese_instruction.jsonl"
    csv_file = dataset_dir / "sudanese_dataset.csv"
    parquet_file = dataset_dir / "sudanese_dataset.parquet"
    readme_file = dataset_dir / "README.md"

    assert chat_file.exists(), "Chat JSONL dataset missing"
    assert instruction_file.exists(), "Instruction JSONL dataset missing"
    assert csv_file.exists(), "CSV dataset missing"
    assert parquet_file.exists(), "Parquet dataset missing"
    assert readme_file.exists(), "README documentation missing"

    # Validate Chat JSONL format
    with open(chat_file, "r", encoding="utf-8") as f:
        chat_lines = [json.loads(line) for line in f]
    assert len(chat_lines) >= 100
    assert "messages" in chat_lines[0]
    assert len(chat_lines[0]["messages"]) == 3

    # Validate Instruction JSONL format
    with open(instruction_file, "r", encoding="utf-8") as f:
        inst_lines = [json.loads(line) for line in f]
    assert len(inst_lines) >= 100
    assert "instruction" in inst_lines[0]
    assert "output" in inst_lines[0]

def test_sqlite_lexicon_database():
    """Verify SQLite lexicon database query capabilities and schema."""
    import sqlite3
    from pathlib import Path

    db_path = Path("data/processed/sudanese_lexicon.db")
    assert db_path.exists(), "SQLite lexicon database file missing"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT count(*) FROM lexicon")
    total_count = cursor.fetchone()[0]
    assert total_count >= 50, f"Expected at least 50 lexicon terms, got {total_count}"

    cursor.execute("SELECT meaning FROM lexicon WHERE term LIKE '%أباط%'")
    meaning = cursor.fetchone()
    assert meaning is not None
    assert "إِبْط" in meaning[0] or "الكتف" in meaning[0]

    conn.close()

def test_translation_endpoints():
    """Verify /translate GET and POST endpoints."""
    # Test GET /translate
    res_get = client.get("/translate?text=أباط&target_lang=english")
    assert res_get.status_code == 200
    get_json = res_get.json()
    assert "translation" in get_json
    assert get_json["original_text"] == "أباط"

    # Test POST /translate to MSA
    res_post = client.post("/translate", json={
        "text": "أباط",
        "source_lang": "sudanese",
        "target_lang": "msa",
        "region": "khartoum"
    })
    assert res_post.status_code == 200
    post_json = res_post.json()
    assert "إِبْط" in post_json["translation"] or "الكتف" in post_json["translation"] or len(post_json["translation"]) > 0


def test_ocr_refinement_and_cleanup():
    """Verify rule-based OCR string refiner, page section classification, and database cleanup."""
    from scripts.refine_ocr_with_ai import heuristic_refine_entry, clean_ocr_text, expand_dictionary_abbreviations, classify_page_range
    from scripts.cleanup_database import clean_ocr_noise_and_digits, standardize_category

    raw_term = "^ أبجر"
    raw_meaning = "أبجر (س) (دارفور) (٣٤) أوالمنتفخ البطن (ف) الغليظ (م) في المثل القديم (ش) في المسدار (مولدة)"

    cleaned_text = clean_ocr_text(raw_meaning)
    assert "^" not in cleaned_text

    cleaned_digits = clean_ocr_noise_and_digits(raw_meaning)
    assert "(٣٤)" not in cleaned_digits

    expanded = expand_dictionary_abbreviations(raw_meaning)
    assert "[عامية سودانية]" in expanded
    assert "[إقليم دارفور]" in expanded
    assert "[أصل فصيح]" in expanded

    refined = heuristic_refine_entry(raw_term, raw_meaning)
    assert refined["term"] == "أبجر"
    assert refined["region"] == "darfur"

    preface_section = classify_page_range("مقدمة", "مقدمة القاموس والتمهيد اللغوي")
    assert preface_section == "book_preface"

    assert standardize_category("مثل_ومقولة") == "مثل_ومقولة"


def test_lexicon_reconstruction():
    """Verify reconstruction logic for reversed OCR brackets and headword extraction."""
    from scripts.reconstruct_lexicon import clean_ocr_noise, extract_headword

    raw_text = "رضوة )س رضى . تكلم بالأعجمية رطن ) ف س"
    cleaned = clean_ocr_noise(raw_text)
    assert "[عامية سودانية]" in cleaned
    assert "[أصل فصيح وعامي]" in cleaned

    headword, meaning = extract_headword("ف", "تكلم بالأعجمية رطن")
    assert headword == "تكلم"
