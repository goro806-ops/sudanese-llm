"""Data Preprocessing for Multi-Regional Sudanese Dialects."""
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
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\u064B-\u0652]', '', text)  # remove harakat
    return text

def format_regional_prompt(text: str, region: str = "khartoum") -> str:
    """Prepend regional dialect tag to prompt text."""
    tag = REGIONAL_DIALECT_TAGS.get(region.lower(), "[GEN]")
    return f"{tag} {clean_sudanese_arabic(text)}"
