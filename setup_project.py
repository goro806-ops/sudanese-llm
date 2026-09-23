import os

def create_project_structure():
    directories = [
        "data/raw",
        "data/processed",
        "data/regional",
        "web_server",
        "vector_db",
        "training",
        "config",
        "tests"
    ]
    
    files = {
        "web_server/app.py": '''"""FastAPI Web Server for Multi-Regional Sudanese LLM."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Multi-Regional Sudanese LLM API")

class GenerationRequest(BaseModel):
    prompt: str
    region: Optional[str] = "khartoum"
    max_tokens: Optional[int] = 100

class SearchRequest(BaseModel):
    query: str
    region: Optional[str] = None
    top_k: Optional[int] = 5

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Sudanese LLM API"}

@app.get("/regions")
def list_regions():
    return {"regions": ["khartoum", "darfur", "kordofan", "eastern", "northern"]}

@app.post("/generate")
def generate_text(req: GenerationRequest):
    return {
        "region": req.region,
        "prompt": req.prompt,
        "response": f"[{req.region.upper()} Dialect Response] {req.prompt}"
    }

@app.post("/search")
def search_vector_db(req: SearchRequest):
    return {
        "query": req.query,
        "region": req.region,
        "results": [
            {"id": 1, "text": f"Sudanese dialect context for {req.query}", "score": 0.95, "region": req.region or "general"}
        ]
    }
''',
        "vector_db/vector_store.py": '''"""Vector Database Interface for Multi-Regional Knowledge Store."""
import json
import math
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add documents with metadata e.g. {'id': 1, 'text': '...', 'region': 'khartoum'}."""
        for doc in docs:
            self.documents.append(doc)

    def search(self, query: str, region: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform search filtered by region if provided."""
        results = []
        query_words = set(query.lower().split())
        for doc in self.documents:
            if region and doc.get("region") and doc.get("region") != region:
                continue
            doc_words = set(doc.get("text", "").lower().split())
            overlap = len(query_words.intersection(doc_words))
            score = overlap / (len(query_words) + 1e-5)
            results.append({**doc, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
''',
        "training/preprocessing.py": '''"""Data Preprocessing for Multi-Regional Sudanese Dialects."""
import re
from typing import List, Dict

REGIONAL_DIALECT_TAGS = {
    "khartoum": "[KRT]",
    "darfur": "[DAR]",
    "kordofan": "[KOR]",
    "eastern": "[EST]",
    "northern": "[NTH]"
}

def clean_sudanese_arabic(text: str) -> str:
    """Normalize and clean Sudanese Arabic text."""
    text = re.sub(r'\\s+', ' ', text).strip()
    text = re.sub(r'[\\u064B-\\u0652]', '', text)  # remove harakat
    return text

def format_regional_prompt(text: str, region: str = "khartoum") -> str:
    """Prepend regional dialect tag to prompt text."""
    tag = REGIONAL_DIALECT_TAGS.get(region.lower(), "[GEN]")
    return f"{tag} {clean_sudanese_arabic(text)}"
''',
        "training/train.py": '''"""Multi-Regional LLM Fine-Tuning Pipeline."""
import json
from typing import Dict, Any

class RegionalTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.epochs = config.get("epochs", 3)
        self.lr = config.get("learning_rate", 5e-5)

    def train_epoch(self, dataset: list) -> float:
        """Simulate fine-tuning epoch on multi-regional corpus."""
        print(f"Training on {len(dataset)} regional samples for {self.epochs} epochs...")
        loss = 1.0 / (self.epochs + 1)
        return loss

    def save_adapter(self, region: str, output_path: str):
        """Save regional adapter weights."""
        meta = {"region": region, "adapter_status": "trained"}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
''',
        "training/evaluate.py": '''"""Evaluation Metrics for Multi-Regional Sudanese Models."""
from typing import List, Dict

def evaluate_dialect_accuracy(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate accuracy and dialect coverage score."""
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    total = max(len(targets), 1)
    return {
        "accuracy": correct / total,
        "sample_count": float(total)
    }
''',
        "config/config.json": '''{
  "project_name": "Multi-Regional Sudanese LLM",
  "version": "1.0.0",
  "regions": ["khartoum", "darfur", "kordofan", "eastern", "northern"],
  "model_config": {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "learning_rate": 0.00005,
    "epochs": 3
  },
  "vector_db_config": {
    "embedding_dim": 384,
    "distance_metric": "cosine"
  }
}
''',
        "tests/test_project.py": '''"""Comprehensive Test Suite for Multi-Regional Sudanese LLM Project."""
import pytest
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

def test_regions_endpoint():
    res = client.get("/regions")
    assert res.status_code == 200
    assert "khartoum" in res.json()["regions"]

def test_generation_endpoint():
    res = client.post("/generate", json={"prompt": "ازيك يا زول", "region": "khartoum"})
    assert res.status_code == 200
    assert "KHARTOUM" in res.json()["response"]

def test_vector_store():
    store = VectorStore()
    store.add_documents([
        {"id": 1, "text": "مرحبا بك في الخرطوم", "region": "khartoum"},
        {"id": 2, "text": "مرحبا بك في الفاشر", "region": "darfur"}
    ])
    results = store.search("الخرطوم", region="khartoum")
    assert len(results) == 1
    assert results[0]["region"] == "khartoum"

def test_preprocessing():
    cleaned = clean_sudanese_arabic("حَبَابَكْ يَا زُولْ")
    assert "حبابك" in cleaned
    formatted = format_regional_prompt("ازيك", "darfur")
    assert "[DAR]" in formatted

def test_evaluate():
    metrics = evaluate_dialect_accuracy(["a", "b"], ["a", "b"])
    assert metrics["accuracy"] == 1.0
'''
    }

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    for file_path, content in files.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_project_structure()
