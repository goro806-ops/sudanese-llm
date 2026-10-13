#!/usr/bin/env python3
"""
Universal LLM System Prompts & Provider Adapters
-----------------------------------------------
Provides ready-to-use prompt wrappers and system instructions formatted for
OpenAI (GPT-4o), DeepSeek (R1/V3), Google Gemini, Anthropic Claude, and Ollama.
"""

SYSTEM_PROMPT_SUDANESE = (
    "أنت خبير ومعجمي ومساعد ذكي متخصص في اللهجات، الثقافة، والتراث السوداني "
    "لمختلف الأقاليم (الخرطوم، دارفور، كردفان، البجا/الشرق، والنوبية/الشمال)."
)

def get_openai_adapter(user_query: str, context: str = "") -> dict:
    """Format prompt for OpenAI GPT-4o / GPT-3.5 API."""
    content = f"السياق المعجمي:\n{context}\n\nسؤال المستخدم: {user_query}" if context else user_query
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SUDANESE},
            {"role": "user", "content": content}
        ],
        "temperature": 0.3
    }

def get_deepseek_adapter(user_query: str, context: str = "") -> dict:
    """Format prompt for DeepSeek R1 / V3 Chat API."""
    content = f"السياق:\n{context}\n\nالسؤال: {user_query}" if context else user_query
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SUDANESE},
            {"role": "user", "content": content}
        ],
        "model": "deepseek-chat"
    }

def get_gemini_adapter(user_query: str, context: str = "") -> dict:
    """Format prompt for Google Gemini API."""
    text = f"{SYSTEM_PROMPT_SUDANESE}\n\n"
    if context:
        text += f"السياق:\n{context}\n\n"
    text += f"السؤال: {user_query}"
    return {
        "contents": [{"parts": [{"text": text}]}]
    }

def get_claude_adapter(user_query: str, context: str = "") -> dict:
    """Format prompt for Anthropic Claude API."""
    content = f"السياق:\n{context}\n\nالسؤال: {user_query}" if context else user_query
    return {
        "system": SYSTEM_PROMPT_SUDANESE,
        "messages": [{"role": "user", "content": content}]
    }

def get_ollama_adapter(user_query: str, model_name: str = "llama3.3") -> dict:
    """Format request payload for Ollama local server."""
    return {
        "model": model_name,
        "prompt": user_query,
        "system": SYSTEM_PROMPT_SUDANESE,
        "stream": False
    }

if __name__ == "__main__":
    print("Testing Universal LLM Adapters...")
    sample_query = "يعني شنو بشتن العفش؟"
    sample_context = "بشتن: أفسد وشوش الترتيب."
    print("OpenAI Payload:", get_openai_adapter(sample_query, sample_context))
    print("DeepSeek Payload:", get_deepseek_adapter(sample_query, sample_context))
    print("Ollama Payload:", get_ollama_adapter(sample_query))
