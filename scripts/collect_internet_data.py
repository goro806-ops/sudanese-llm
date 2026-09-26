"""Automated Internet Scraper & Dataset Collector for Sudanese Dialects and Rotana."""
import json
import re
import os
import sys
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from pathlib import Path

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vector_db.vector_store import VectorStore

DATA_FILE = Path("data/raw/sudanese_regional_knowledge.json")

def fetch_wikipedia_text(title: str) -> str:
    """Fetch article content from Arabic Wikipedia."""
    url = f"https://ar.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
            return text
    except Exception as e:
        print(f"Error fetching Wikipedia page '{title}': {e}")
        return ""

def clean_and_split_sentences(text: str):
    """Split text into sentences and clean noise."""
    text = re.sub(r'\[\d+\]', '', text)  # remove Wikipedia citation brackets
    sentences = re.split(r'[\.\!\؟\n]+', text)
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) > 25 and not s_clean.startswith("coordinates"):
            cleaned.append(s_clean)
    return cleaned

def collect_online_dialect_data():
    print("🌐 Starting Automated Internet Scraper for Sudanese Dialects...")

    # Wikipedia articles related to Sudanese dialects and regions
    sources = [
        ("اللهجة_السودانية", "khartoum", "vocabulary"),
        ("اللهجة_الدارفورية", "darfur", "vocabulary"),
        ("كردفان", "kordofan", "culture"),
        ("البجا", "eastern", "culture"),
        ("اللغة_النوبية", "northern", "vocabulary"),
        ("الأهرامات_النوبية", "northern", "history"),
        ("جبل_مرة", "darfur", "geography"),
        ("الخرطوم", "khartoum", "culture"),
        ("كسلا", "eastern", "culture"),
        ("دنقلة", "northern", "culture"),
        ("الأبيض_(مدينة)", "kordofan", "culture")
    ]

    # Load existing entries
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_texts = set(item.get("text", "").strip() for item in existing_data)
    next_id = max([item.get("id", 0) for item in existing_data], default=0) + 1
    new_entries = []

    for wiki_title, region, topic in sources:
        print(f"  --> Scraping Wikipedia: {wiki_title} ({region})...")
        full_text = fetch_wikipedia_text(wiki_title)
        if full_text:
            sentences = clean_and_split_sentences(full_text)
            for s in sentences[:15]:  # top informative sentences
                if s not in existing_texts:
                    existing_texts.add(s)
                    new_entries.append({
                        "id": next_id,
                        "region": region,
                        "topic": topic,
                        "text": s,
                        "source": f"Wikipedia ({wiki_title})"
                    })
                    next_id += 1

    # Merge and write back
    updated_data = existing_data + new_entries
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Scraping finished! Added {len(new_entries)} new entries from the internet.")
    print(f"📊 Total database entries: {len(updated_data)}")

    # Update Vector Database index
    print("🔄 Updating Vector Database index...")
    vs = VectorStore()
    vs.documents = []
    vs.load_raw_dataset()
    vs.save()
    print("✅ Vector database successfully re-indexed!")

if __name__ == "__main__":
    collect_online_dialect_data()
