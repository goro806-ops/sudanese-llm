"""
Sudanese Dialect Client – Works with ANY LLM
=============================================
This is the model-agnostic layer.

It provides:
- Strong system prompts
- Optional RAG context
- Region handling

You can plug it into OpenAI, Anthropic, xAI (Grok), local models, etc.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

# Allow importing from sibling packages
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.system_prompts import MASTER_SYSTEM_PROMPT, REGION_PROMPTS, build_sudanese_prompt


class SudaneseClient:
    """
    Lightweight wrapper that injects Sudanese dialect knowledge
    into any chat-based LLM.
    """

    def __init__(
        self,
        region: str = "khartoum",
        use_rag: bool = False,
        vector_store=None,
    ):
        self.region = region.lower().strip()
        self.use_rag = use_rag
        self.vector_store = vector_store

        if self.region not in REGION_PROMPTS:
            print(f"Warning: Unknown region '{region}'. Falling back to khartoum.")
            self.region = "khartoum"

    def build_messages(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k_rag: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Build a complete messages list ready to send to any chat API.
        """
        # System prompt
        system_content = MASTER_SYSTEM_PROMPT + "\n\n" + REGION_PROMPTS[self.region]

        # Optional RAG context
        if self.use_rag and self.vector_store is not None:
            results = self.vector_store.search(user_message, region=self.region, top_k=top_k_rag)
            if results:
                context_parts = ["\n### معلومات داعمة من قاعدة المعرفة السودانية:"]
                for r in results:
                    context_parts.append(f"- ({r.get('region', '')}) {r['text']}")
                system_content += "\n" + "\n".join(context_parts)

        messages = [{"role": "system", "content": system_content}]

        # Previous conversation
        if conversation_history:
            messages.extend(conversation_history)

        # Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def get_system_prompt(self) -> str:
        """Return just the system prompt for manual use."""
        return MASTER_SYSTEM_PROMPT + "\n\n" + REGION_PROMPTS[self.region]


if __name__ == "__main__":
    # Demo: just print the messages that would be sent
    client = SudaneseClient(region="northern")
    messages = client.build_messages("مسكاقمي كيف الحال؟")
    print("=== Messages that will be sent to the LLM ===\n")
    for m in messages:
        print(f"[{m['role'].upper()}]\n{m['content']}\n")
        print("-" * 50)
