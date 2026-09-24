from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multi-Regional Sudanese LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

"""FastAPI Web Server for Multi-Regional Sudanese LLM."""
import os
import httpx
from fastapi import FastAPI, HTTPException
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
    top_k: Optional[int] = 5

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

@app.post("/generate")
def generate_text(req: GenerationRequest):
    region = req.region or "khartoum"
    formatted_prompt = format_regional_prompt(req.prompt, region)
    
    # Retrieve relevant regional context from vector store (RAG)
    context_docs = vector_store.search(req.prompt, region=region, top_k=2)
    context_str = " ".join([d.get("text", "") for d in context_docs]) if context_docs else ""

    # Check for Hugging Face or OpenAI API Key environment variables for external LLM inference
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if hf_token:
        try:
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "inputs": f"Context: {context_str}\nPrompt: {formatted_prompt}",
                "parameters": {"max_new_tokens": req.max_tokens or 100}
            }
            # Default model HF inference URL
            url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
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
        except Exception as e:
            print(f"HF API Inference call failed: {e}")

    # Default fallback simulated regional response with RAG context
    response_msg = f"[{region.upper()} Dialect Response] {req.prompt}"
    if context_str:
        response_msg += f" (Context: {context_str})"

    return {
        "region": region,
        "prompt": req.prompt,
        "formatted_prompt": formatted_prompt,
        "rag_context": context_docs,
        "response": response_msg,
        "engine": "Local Multi-Regional Rule Engine"
    }

@app.post("/search")
def search_vector_db(req: SearchRequest):
    results = vector_store.search(req.query, region=req.region, top_k=req.top_k or 5)
    return {
        "query": req.query,
        "region": req.region,
        "results": results
    }
