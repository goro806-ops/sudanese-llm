"""Comprehensive Test Suite for Multi-Regional Sudanese LLM Project."""
import pytest
import os
from web_server.app import app
from fastapi.testclient import TestClient
from vector_db.vector_store import VectorStore
from training.preprocessing import clean_sudanese_arabic, format_regional_prompt
from training.evaluate import evaluate_dialect_accuracy

client = TestClient(app)

def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["indexed_documents"] >= 7

def test_regions_endpoint():
    res = client.get("/regions")
    assert res.status_code == 200
    assert "khartoum" in res.json()["regions"]

def test_generation_endpoint():
    res = client.post("/generate", json={"prompt": "ازيك يا زول", "region": "khartoum"})
    assert res.status_code == 200
    assert "KHARTOUM" in res.json()["response"]
    assert "rag_context" in res.json()

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
