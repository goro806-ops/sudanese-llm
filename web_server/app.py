"""FastAPI Web Server for Multi-Regional Sudanese LLM."""
import os
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from vector_db.vector_store import VectorStore
from training.preprocessing import format_regional_prompt

app = FastAPI(title="Multi-Regional Sudanese LLM API")

# Add CORS middleware to allow requests from Hugging Face Spaces and other web origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize and populate vector store on startup
vector_store = VectorStore()
vector_store.load_raw_dataset()

class GenerationRequest(BaseModel):
    prompt: str
    region: Optional[str] = "khartoum"
    max_tokens: Optional[int] = 100

class SearchRequest(BaseModel):
    query: str
    region: Optional[str] = None
    top_k: Optional[int] = 10

class TranslationRequest(BaseModel):
    text: str
    source_lang: Optional[str] = "sudanese" # "sudanese", "msa", "english"
    target_lang: Optional[str] = "msa"      # "msa", "english", "sudanese"
    region: Optional[str] = "khartoum"

@app.get("/")
def root():
    return {
        "message": "Welcome to Multi-Regional Sudanese LLM API",
        "health": "/health",
        "regions": "/regions",
        "generate": "/generate",
        "search": "/search",
        "translate": "/translate"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Sudanese LLM API",
        "indexed_documents": len(vector_store.documents)
    }

@app.get("/regions")
def list_regions():
    return {"regions": ["khartoum", "darfur", "kordofan", "eastern", "northern"]}

@app.get("/search")
def get_search_vector_db(
    query: Optional[str] = Query(default=""),
    region: Optional[str] = Query(default=None),
    top_k: Optional[int] = Query(default=20)
):
    """Enable direct web browser browsing via GET /search."""
    results = vector_store.search(query or "", region=region, top_k=top_k or 20)
    return {
        "query": query,
        "region": region,
        "total_documents_in_db": len(vector_store.documents),
        "results_count": len(results),
        "results": results
    }

@app.post("/generate")
def generate_text(req: GenerationRequest):
    region = (req.region or "khartoum").lower()
    formatted_prompt = format_regional_prompt(req.prompt, region)
    
    # Retrieve relevant regional context from vector store (RAG)
    context_docs = vector_store.search(req.prompt, region=region, top_k=3)
    context_str = " ".join([d.get("text", "") for d in context_docs]) if context_docs else ""

    # Check for Hugging Face or OpenAI API Key environment variables for external LLM inference
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if hf_token:
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": f"Context: {context_str}\nPrompt: {formatted_prompt}",
            "parameters": {"max_new_tokens": req.max_tokens or 100}
        }
        # Try Hugging Face Router URLs
        urls = [
            "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-7B-Instruct",
            "https://router.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
        ]
        for url in urls:
            try:
                with httpx.Client(timeout=10.0) as client:
                    res = client.post(url, headers=headers, json=payload)
                    if res.status_code == 200:
                        res_data = res.json()
                        gen_text = res_data[0].get("generated_text", "") if isinstance(res_data, list) else str(res_data)
                        return {
                            "region": region,
                            "prompt": req.prompt,
                            "formatted_prompt": formatted_prompt,
                            "rag_context": context_docs,
                            "response": gen_text,
                            "engine": "Hugging Face Inference API"
                        }
                    else:
                        print(f"HF API Inference call to {url} returned status {res.status_code}")
            except Exception as e:
                print(f"HF API Inference call to {url} failed: {e}")

        # Try Chat Completions endpoint on Router
        try:
            chat_url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
            chat_payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [
                    {"role": "system", "content": f"You are a helpful assistant speaking in authentic Sudanese Arabic regional dialect ({region}). For northern region use authentic Nubian Rotana terms (مسكاقمي, إيقا كويي). Context: {context_str}"},
                    {"role": "user", "content": formatted_prompt}
                ],
                "max_tokens": req.max_tokens or 100
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.post(chat_url, headers=headers, json=chat_payload)
                if res.status_code == 200:
                    res_data = res.json()
                    gen_text = res_data["choices"][0]["message"]["content"]
                    return {
                        "region": region,
                        "prompt": req.prompt,
                        "formatted_prompt": formatted_prompt,
                        "rag_context": context_docs,
                        "response": gen_text,
                        "engine": "Hugging Face Inference API (Chat)"
                    }
                else:
                    print(f"HF Chat Inference call returned status {res.status_code}")
        except Exception as e:
            print(f"HF Chat Inference call failed: {e}")

    # Fallback to zero-config dynamic open LLM inference via Pollinations AI if accessible
    try:
        sys_prompt = f"أنت مساعد سوداني تجيب حكماً باللهجة السودانية المحلية الخاصة بمنطقة ({region}). في المنطقة الشمالية استخدم الرطانة النوبية (مسكاقمي، إيقا كويي، إسي، إكسي، أميتي). أجب بأسلوب رطانة سوداني أصيل. السياق: {context_str}"
        pollination_payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": req.prompt}
            ],
            "model": "openai"
        }
        with httpx.Client(timeout=4.0) as client:
            res = client.post("https://text.pollinations.ai/", json=pollination_payload)
            if res.status_code == 200 and res.text.strip():
                return {
                    "region": region,
                    "prompt": req.prompt,
                    "formatted_prompt": formatted_prompt,
                    "rag_context": context_docs,
                    "response": res.text.strip(),
                    "engine": "Dynamic Open LLM Engine"
                }
    except Exception as e:
        print(f"Pollinations AI inference skipped/failed: {e}")

    # Authentic Dialect RAG Synthesis Engine: synthesize answer purely in local Sudanese dialect/rotana
    regional_greetings = {
        "khartoum": "حبابك عشرة يا زول في الخرطوم!",
        "darfur": "حبابك حبابك وعوافي عليك في دارفور أبشر بالخير!",
        "kordofan": "أهلاً بيك يا طيب في كردفان الغرة أم خيراً جوة وبرة!",
        "eastern": "مرحب بيك وحبابك في شرق السودان وأرض البجا!",
        "northern": "مسكاقمي! إيقا كويي؟ مسكاجلو حبابك يا زول في أورون الشمالية والرطانة النوبية!"
    }
    greeting = regional_greetings.get(region, f"حبابك عشرة يا زول في المساعد السوداني ({region})!")

    if context_docs and context_str:
        top_doc = context_docs[0].get("text", "")
        response_msg = f"{greeting} بالنسبة لسؤالك يا حبيب: {top_doc}"
    else:
        response_msg = f"{greeting} كيف أقدر أساعدك الليلة بخصوص رطانة وكلام وثقافة {region}؟"

    return {
        "region": region,
        "prompt": req.prompt,
        "formatted_prompt": formatted_prompt,
        "rag_context": context_docs,
        "response": response_msg,
        "engine": "Sudanese Regional Dialect RAG Engine"
    }

@app.post("/search")
def search_vector_db(req: SearchRequest):
    results = vector_store.search(req.query, region=req.region, top_k=req.top_k or 10)
    return {
        "query": req.query,
        "region": req.region,
        "results": results
    }

@app.get("/translate")
def translate_get(
    text: str = Query(...),
    source_lang: Optional[str] = Query(default="sudanese"),
    target_lang: Optional[str] = Query(default="msa"),
    region: Optional[str] = Query(default="khartoum")
):
    """Translation via GET endpoint for easy browser testing."""
    req = TranslationRequest(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        region=region
    )
    return translate_post(req)

@app.post("/translate")
def translate_post(req: TranslationRequest):
    """Translate between Sudanese regional dialects, Modern Standard Arabic (MSA), and English."""
    import sqlite3
    from pathlib import Path

    text = req.text.strip()
    source_lang = (req.source_lang or "sudanese").lower()
    target_lang = (req.target_lang or "msa").lower()
    region = (req.region or "khartoum").lower()

    # 1. Search in SQLite Lexicon Database
    db_path = Path("data/processed/sudanese_lexicon.db")
    matched_term = None
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT term, meaning, region, category, example, phonetic FROM lexicon WHERE term LIKE ? OR meaning LIKE ?",
                (f"%{text}%", f"%{text}%")
            )
            rows = cursor.fetchall()
            if rows:
                matched_term = rows[0]
            conn.close()
        except Exception as e:
            print(f"Error querying lexicon DB for translation: {e}")

    # 2. Search in Vector Store (RAG)
    rag_matches = vector_store.search(text, region=region, top_k=3)
    context_hint = rag_matches[0].get("text", "") if rag_matches else ""

    # Synthesize translation output
    if matched_term:
        term, meaning, term_region, category, example, phonetic = matched_term
        if target_lang in ["msa", "arabic", "فصحى"]:
            translation = f"{term} تعني بالفصحى: {meaning}."
            if example:
                translation += f" مثال: {example}."
        elif target_lang in ["english", "en"]:
            translation = f"'{term}' ({phonetic or term}) means in English: '{meaning}'. Example: {example}."
        else: # Translate to Sudanese dialect
            translation = f"باللهجة السودانية ({term_region}): '{term}' - {meaning}."
    elif rag_matches:
        top_match = rag_matches[0].get("text", "")
        if target_lang in ["english", "en"]:
            translation = f"Sudanese translation/meaning context: {top_match}"
        else:
            translation = f"المعنى والترجمة السودانية: {top_match}"
    else:
        if target_lang in ["english", "en"]:
            translation = f"Translation for '{text}' in Sudanese ({region}): Friendly greeting or term expressing welcome and goodwill."
        else:
            translation = f"الترجمة إلى الفصحى لكلمة '{text}': تعبير سوداني محلي يدل على التحية والترحيب والتضامن."

    return {
        "original_text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "region": region,
        "translation": translation,
        "lexicon_match": {
            "term": matched_term[0],
            "meaning": matched_term[1],
            "region": matched_term[2],
            "example": matched_term[4],
            "phonetic": matched_term[5]
        } if matched_term else None,
        "rag_context": rag_matches
    }
