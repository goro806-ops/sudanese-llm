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

    cursor.execute("SELECT meaning FROM lexicon WHERE term = 'مسكاقمي'")
    meaning = cursor.fetchone()
    assert meaning is not None
    assert "نوبية" in meaning[0] or "تحية" in meaning[0]

    conn.close()

def test_translation_endpoints():
    """Verify /translate GET and POST endpoints."""
    # Test GET /translate
    res_get = client.get("/translate?text=مسكاقمي&target_lang=english")
    assert res_get.status_code == 200
    get_json = res_get.json()
    assert "translation" in get_json
    assert get_json["original_text"] == "مسكاقمي"

    # Test POST /translate to MSA
    res_post = client.post("/translate", json={
        "text": "زول",
        "source_lang": "sudanese",
        "target_lang": "msa",
        "region": "khartoum"
    })
    assert res_post.status_code == 200
    post_json = res_post.json()
    assert "شخص" in post_json["translation"] or "إنسان" in post_json["translation"]
