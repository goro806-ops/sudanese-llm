"""
Improved Vector Store for Sudanese LLM
======================================
Replaces the simple keyword search with real semantic embeddings.

Requirements:
    pip install sentence-transformers chromadb

Usage:
    from improved_vector_store import SudaneseVectorStore
    store = SudaneseVectorStore()
    store.load_from_json("data/raw/sudanese_regional_knowledge.json")
    results = store.search("حبابك عشرة", region="khartoum", top_k=5)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Warning: sentence-transformers or chromadb not installed. Falling back to keyword search.")


class SudaneseVectorStore:
    def __init__(
        self,
        collection_name: str = "sudanese_knowledge",
        persist_directory: str = "data/processed/chroma_db",
        embedding_model: str = "intfloat/multilingual-e5-base",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.documents: List[Dict[str, Any]] = []

        if HAS_EMBEDDINGS:
            self.model = SentenceTransformer(embedding_model)
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.model = None
            self.collection = None

    def load_from_json(self, json_path: str | Path):
        """Load documents from the project's raw knowledge JSON."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for item in data:
            text = item.get("text") or item.get("content") or ""
            if not text.strip():
                continue
            docs.append({
                "id": str(item.get("id", len(docs) + 1)),
                "text": text.strip(),
                "region": item.get("region", "general").lower(),
                "topic": item.get("topic", "general"),
                "source": item.get("source", "curated"),
            })

        self.add_documents(docs)
        print(f"Loaded {len(docs)} documents into the vector store.")

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add documents with optional region metadata."""
        if not docs:
            return

        self.documents.extend(docs)

        if not HAS_EMBEDDINGS:
            return

        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [
            {
                "region": d.get("region", "general"),
                "topic": d.get("topic", "general"),
                "source": d.get("source", "curated"),
            }
            for d in docs
        ]

        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        region: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic search with optional region filter."""
        if not query.strip():
            return []

        if not HAS_EMBEDDINGS:
            return self._keyword_search(query, region, top_k)

        where_filter = None
        if region:
            region = region.lower().strip()
            where_filter = {"region": region}

        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "region": results["metadatas"][0][i].get("region", "general"),
                    "topic": results["metadatas"][0][i].get("topic", "general"),
                    "score": 1.0 - results["distances"][0][i],
                })
        return output

    def _keyword_search(self, query: str, region: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        """Simple fallback when embeddings are not available."""
        query_words = [w for w in query.lower().split() if len(w) > 1]
        results = []

        for doc in self.documents:
            if region and doc.get("region") not in (region.lower(), "general"):
                continue
            text_lower = doc["text"].lower()
            score = sum(1 for w in query_words if w in text_lower)
            if score > 0:
                results.append({**doc, "score": float(score)})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        if HAS_EMBEDDINGS and self.collection:
            return self.collection.count()
        return len(self.documents)


if __name__ == "__main__":
    store = SudaneseVectorStore(persist_directory="data/processed/test_chroma")
    print(f"Embeddings available: {HAS_EMBEDDINGS}")
    print(f"Current document count: {store.count()}")
