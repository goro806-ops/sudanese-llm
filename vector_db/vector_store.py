"""Vector Database Interface for Multi-Regional Knowledge Store."""
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join("data", "processed", "vector_store.json")

class VectorStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.documents: List[Dict[str, Any]] = []
        self.load()

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add documents with metadata e.g. {'id': 1, 'text': '...', 'region': 'khartoum'}."""
        for doc in docs:
            # avoid exact duplicate ids/text if present
            if not any(d.get("id") == doc.get("id") and d.get("text") == doc.get("text") for d in self.documents):
                self.documents.append(doc)
        self.save()

    def load_raw_dataset(self, raw_dir: str = os.path.join("data", "raw")):
        """Scan raw_dir for .json and .txt dataset files and index them."""
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            return

        new_docs = []
        doc_id = len(self.documents) + 1

        for file in raw_path.glob("*"):
            if file.suffix == ".json":
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "text" in item:
                                    item.setdefault("id", doc_id)
                                    doc_id += 1
                                    new_docs.append(item)
                except Exception as e:
                    print(f"Error reading JSON file {file}: {e}")
            elif file.suffix == ".txt":
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                new_docs.append({
                                    "id": doc_id,
                                    "text": line,
                                    "region": "general"
                                })
                                doc_id += 1
                except Exception as e:
                    print(f"Error reading TXT file {file}: {e}")

        if new_docs:
            self.add_documents(new_docs)

    def save(self):
        """Persist documents to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def load(self):
        """Load persisted documents from disk if available."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.documents = []

    def search(self, query: str, region: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform search filtered by region if provided."""
        results = []
        query_words = set(query.lower().split())
        for doc in self.documents:
            if region and doc.get("region") and doc.get("region") != region and doc.get("region") != "general":
                continue
            doc_words = set(doc.get("text", "").lower().split())
            overlap = len(query_words.intersection(doc_words))
            score = overlap / (len(query_words) + 1e-5)
            results.append({**doc, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
