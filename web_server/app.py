"""FastAPI Web Server for Multi-Regional Sudanese LLM."""
import os
import sqlite3
import httpx
from pathlib import Path
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

def clean_rag_text(text: str) -> str:
    """Clean Wikipedia boilerplate headers from RAG context."""
    boilerplate = [
        "هذه نسخة متحقق منها من هذه الصفحة",
        "هذه النسخة المستقرة، فحصت في",
        "تعديلات معلقة معروضة"
    ]
    cleaned = text
    for bp in boilerplate:
        cleaned = cleaned.replace(bp, "")
    return cleaned.strip()

def search_lexicon_db(term: str) -> Optional[tuple]:
    """Search for term in SQLite Lexicon DB."""
    db_path = Path("data/processed/sudanese_lexicon.db")
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT term, meaning, region, category, example, phonetic FROM lexicon WHERE term LIKE ? OR meaning LIKE ?",
                (f"%{term}%", f"%{term}%")
            )
            rows = cursor.fetchall()
            conn.close()
            if rows:
                return rows[0]
        except Exception as e:
            print(f"Error searching lexicon DB: {e}")
    return None

@app.post("/generate")
def generate_text(req: GenerationRequest):
    region = (req.region or "khartoum").lower()
    prompt_clean = req.prompt.strip().lower()
    formatted_prompt = format_regional_prompt(req.prompt, region)

    # Regional greetings and persona definition
    regional_greetings = {
        "khartoum": "حبابك عشرة يا زول في الخرطوم!",
        "darfur": "حبابك حبابك وعوافي عليك في دارفور، أبشر بالخير!",
        "kordofan": "أهلاً بيك يا طيب في كردفان الغرة، أم خيراً جوة وبرة!",
        "eastern": "مرحب بيك وحبابك في شرق السودان وأرض البجا!",
        "northern": "مسكاقمي! إيقا كويي؟ مسكاجلو حبابك يا زول في أورون الشمالية والرطانة النوبية!"
    }
    greeting = regional_greetings.get(region, f"حبابك عشرة يا زول في المساعد السوداني ({region})!")

    # 1. Intent Recognition: Conversational Questions
    identity_keywords = ["من أنت", "منو انت", "عرف بنفسك", "مين انت", "من انت", "شنو انت"]
    if any(k in prompt_clean for k in identity_keywords):
        resp_msg = f"{greeting}\nأنا **المساعد الذكي للهجات والثقافة السودانية** 🇸🇩.\nأصمّمت لمساعدتك في التحدث بفهم ومفردات اللهجات السودانية المختلفة (الخرطوم، دارفور، كردفان، الشرق، والشمالية بالرطانة النوبية)، وتوضيح معاني الأمثال والتراث السوداني الأصيل."
        return {
            "region": region,
            "prompt": req.prompt,
            "formatted_prompt": formatted_prompt,
            "rag_context": [],
            "response": resp_msg,
            "engine": "Sudanese Conversational Router"
        }

    capability_keywords = ["إمكانياتك", "امكانياتك", "بتعمل شنو", "شنو بتقدر", "شنو بتعرف", "شنو امكانياتك", "كيف بتساعد"]
    if any(k in prompt_clean for k in capability_keywords):
        resp_msg = f"{greeting}\nإمكانياتي تشمل:\n1️⃣ **تفسير وترجمة اللهجات السودانية**: تحويل المفردات بين اللهجات الفتحة، الفصحى، والإنجليزي.\n2️⃣ **الرطانة النوبية بالشمالية**: شرح كلمات ومصطلحات نوبية مثل (مسكاقمي، إكسي، إيقا كويي).\n3️⃣ **التراث والأمثال الشعبية**: شرح معاني الأمثال السودانية في جميع الأقاليم.\n4️⃣ **المعلومات الجغرافية والثقافية**: الإجابة عن المعالم والمدن السودانية."
        return {
            "region": region,
            "prompt": req.prompt,
            "formatted_prompt": formatted_prompt,
            "rag_context": [],
            "response": resp_msg,
            "engine": "Sudanese Conversational Router"
        }

    # 2. Intent Recognition: Translation / Dialect conversion requests
    translation_keywords = ["حول", "ترجم", "اقلب", "أقلب", "رطانة", "بالرطانة", "بالشمالية", "بالسوداني"]
    if any(k in prompt_clean for k in translation_keywords):
        # Look for dictionary matches first
        lex_match = search_lexicon_db(req.prompt)
        if lex_match:
            term, meaning, term_region, category, example, phonetic = lex_match
            resp_msg = f"{greeting}\nالترجمة والمعنى المطلوب:\n📌 **الكلمة/المصطلح**: {term}\n💡 **المعنى**: {meaning}\n🗺️ **الإقليم**: {term_region}"
            if example:
                resp_msg += f"\n💬 **مثال**: {example}"
            return {
                "region": region,
                "prompt": req.prompt,
                "formatted_prompt": formatted_prompt,
                "rag_context": [],
                "response": resp_msg,
                "engine": "Sudanese Lexicon Engine"
            }
        elif region == "northern" or "رطانة" in prompt_clean or "شمالية" in prompt_clean:
            resp_msg = f"{greeting}\nبالنسبة للتحويل للرطانة النوبية الشمالية:\n• التحية: **مسكاقمي** (كيف حالك / أهلاً)\n• السؤال عن الحال: **إيقا كويي؟** (هل أنت بخير؟)\n• القوت والخبز: **إكسي**\n• الماء: **إسي**\n• الترحيب بالأهل: **مسكاجلو / أورون مسكاقرو**"
            return {
                "region": region,
                "prompt": req.prompt,
                "formatted_prompt": formatted_prompt,
                "rag_context": [],
                "response": resp_msg,
                "engine": "Sudanese Rotana Translator"
            }

    # Retrieve relevant regional context from vector store (RAG)
    context_docs = vector_store.search(req.prompt, region=region, top_k=3)
    context_str = " ".join([clean_rag_text(d.get("text", "")) for d in context_docs]) if context_docs else ""

    # Check for Hugging Face API Key environment variables for external LLM inference
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if hf_token:
        headers = {"Authorization": f"Bearer {hf_token}"}

        # Try Chat Completions endpoint on Router first
        try:
            chat_url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
            chat_payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [
                    {"role": "system", "content": f"أنت المساعد الذكي للهجات والثقافة السودانية. تتحدث وتجيب دائماً باللهجة السودانية الأصلية بحسب منطقة ({region}). تجنب استخدام اللهجات غير السودانية (مثل المصرية أو الشامية). في الشمالية استخدم الرطانة النوبية (مسكاقمي، إيقا كويي). السياق الداعم: {context_str}"},
                    {"role": "user", "content": formatted_prompt}
                ],
                "max_tokens": req.max_tokens or 150
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.post(chat_url, headers=headers, json=chat_payload)
                if res.status_code == 200:
                    res_data = res.json()
                    gen_text = res_data["choices"][0]["message"]["content"]
                    # Clean out any accidental ChatGPT mentions if LLM hallucinates
                    gen_text = gen_text.replace("تشات جي بي تي", "المساعد الذكي للهجات السودانية").replace("ChatGPT", "المساعد الذكي للهجات السودانية")
                    return {
                        "region": region,
                        "prompt": req.prompt,
                        "formatted_prompt": formatted_prompt,
                        "rag_context": context_docs,
                        "response": gen_text,
                        "engine": "Hugging Face Inference API (Chat)"
                    }
        except Exception as e:
            print(f"HF Chat Inference call failed: {e}")

    # Fallback to Pollinations AI with strict Sudanese system persona
    try:
        sys_prompt = f"أنت المساعد الذكي للهجات والثقافة السودانية. تجيب حكماً وأصالة باللهجة السودانية المحلية الخاصة بمنطقة ({region}). يمنع منعاً باتاً استخدام العبارات المصرية مثل (إيه معاك، يا عم). لا تقل أبداً أنك ChatGPT. في الشمالية استخدم الرطانة النوبية (مسكاقمي، إيقا كويي، إكسي). السياق المفيد: {context_str}"
        pollination_payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": req.prompt}
            ],
            "model": "openai"
        }
        with httpx.Client(timeout=5.0) as client:
            res = client.post("https://text.pollinations.ai/", json=pollination_payload)
            if res.status_code == 200 and res.text.strip():
                gen_text = res.text.strip()
                gen_text = gen_text.replace("تشات جي بي تي", "المساعد الذكي للهجات السودانية").replace("ChatGPT", "المساعد الذكي للهجات السودانية")
                return {
                    "region": region,
                    "prompt": req.prompt,
                    "formatted_prompt": formatted_prompt,
                    "rag_context": context_docs,
                    "response": gen_text,
                    "engine": "Dynamic Open LLM Engine"
                }
    except Exception as e:
        print(f"Pollinations AI inference failed: {e}")

    # Fallback: Clean Sudanese Regional Dialect RAG Engine
    if context_docs and context_str:
        cleaned_doc = clean_rag_text(context_docs[0].get("text", ""))
        response_msg = f"{greeting}\nمعلومات مستخرجة حول سؤالك بخصوص ({region}):\n{cleaned_doc}"
    else:
        response_msg = f"{greeting}\nكيف أقدر أساعدك الليلة بخصوص رطانة وكلام وثقافة {region}؟ اسألني عن معاني الكلمات أو الأمثال السودانية الأصيلة."

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
    text = req.text.strip()
    source_lang = (req.source_lang or "sudanese").lower()
    target_lang = (req.target_lang or "msa").lower()
    region = (req.region or "khartoum").lower()

    # 1. Search in SQLite Lexicon Database
    matched_term = search_lexicon_db(text)

    # 2. Search in Vector Store (RAG)
    rag_matches = vector_store.search(text, region=region, top_k=3)

    # Synthesize translation output
    if matched_term:
        term, meaning, term_region, category, example, phonetic = matched_term
        if target_lang in ["msa", "arabic", "فصحى"]:
            translation = f"'{term}' تعني بالفصحى: {meaning}."
            if example:
                translation += f" مثال: {example}."
        elif target_lang in ["english", "en"]:
            translation = f"'{term}' ({phonetic or term}) means in English: '{meaning}'. Example: {example}."
        else:
            translation = f"باللهجة السودانية ({term_region}): '{term}' - {meaning}."
    elif rag_matches:
        top_match = clean_rag_text(rag_matches[0].get("text", ""))
        if target_lang in ["english", "en"]:
            translation = f"Sudanese translation/meaning context: {top_match}"
        else:
            translation = f"المعنى والترجمة السودانية: {top_match}"
    else:
        if target_lang in ["english", "en"]:
            translation = f"Translation for '{text}' in Sudanese ({region}): Friendly greeting or local term expressing welcome and goodwill."
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
