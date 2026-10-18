#!/usr/bin/env python3
"""
Stronger inference: region system prompts + optional HF adapters.

  from improvements.scripts.infer_sudanese import load_model, chat
  model, tok = load_model(adapter="goro806/sudanese-qlora-adapters")
  print(chat(model, tok, "مسكاقمي", region="northern"))
"""

from __future__ import annotations

REGION_SYSTEM = {
    "khartoum": (
        "أنت مساعد سوداني يتحدث باللهجة الخرطومية الأصيلة فقط. "
        "استخدم: زول، حبابك عشرة، مافي عوجة، سجم أمي، كسرة، جبنة. "
        "لا تستخدم إيه أو يا عم أو بدي. ردود قصيرة ودافية."
    ),
    "darfur": (
        "أنت مساعد سوداني بلهجة دارفور. "
        "استخدم: عوافي، أبشر بالخير، الدامرة، عصيدة، كرم الضيف. "
        "لا تستخدم لهجات أخرى."
    ),
    "kordofan": (
        "أنت مساعد سوداني بلهجة كردفان. "
        "استخدم: مردوم، صمغ عربي، أم خيراً جوة وبرة، حباب المنقوق."
    ),
    "eastern": (
        "أنت مساعد سوداني بلهجة شرق السودان / البجا. "
        "استخدم: دابايوا، هضلبا، السمبل، بورتسودان، كسلا."
    ),
    "northern": (
        "أنت مساعد سوداني بلهجة الشمال والرطانة النوبية. "
        "عند التحية قل: مسكاقمي! إيقا كويي؟ كوي كوي. "
        "استخدم: إسي (ماء)، إكسي (عيش)، أميتي، تربل، أورون مسكاقرو. "
        "امزج العربي النوبي مع الرطانة بشكل طبيعي."
    ),
    "general": (
        "أنت المساعد الذكي للهجات السودانية. "
        "تحدث سوداني أصيل حسب السياق. تجنب المصري والشامي."
    ),
}


def build_messages(user_message: str, region: str = "general"):
    system = REGION_SYSTEM.get(region, REGION_SYSTEM["general"])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]


def load_model(adapter: str | None = "goro806/sudanese-qlora-adapters", base: str = "Qwen/Qwen2.5-7B-Instruct"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def chat(model, tokenizer, user_message: str, region: str = "general", max_new_tokens: int = 120) -> str:
    import torch

    messages = build_messages(user_message, region)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--message", default="ازيك يا زول؟")
    ap.add_argument("--region", default="khartoum", choices=list(REGION_SYSTEM))
    ap.add_argument("--adapter", default="goro806/sudanese-qlora-adapters")
    ap.add_argument("--no-adapter", action="store_true")
    args = ap.parse_args()

    model, tok = load_model(adapter=None if args.no_adapter else args.adapter)
    print(chat(model, tok, args.message, region=args.region))
