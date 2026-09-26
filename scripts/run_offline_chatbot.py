"""Offline Interactive Terminal Chatbot for Multi-Regional Sudanese LLM & Rotana."""
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vector_db.vector_store import VectorStore

def main():
    print("=" * 60)
    print("🇸🇩 المساعد الذكي للهجات والثقافة السودانية (Offline Mode)")
    print("=" * 60)
    
    vs = VectorStore()
    vs.load_raw_dataset()
    
    print(f"تم تحميل القاعدة المعرفية بنجاح! ({len(vs.documents)} وثائق ومصطلحات)")
    print("المناطق المتاحة: khartoum, darfur, kordofan, eastern, northern")
    print("اكتب 'exit' للخروج.")
    print("-" * 60)
    
    regional_greetings = {
        "khartoum": "حبابك عشرة يا زول في الخرطوم!",
        "darfur": "حبابك حبابك وعوافي عليك في دارفور أبشر بالخير!",
        "kordofan": "أهلاً بيك يا طيب في كردفان الغرة أم خيراً جوة وبرة!",
        "eastern": "مرحب بيك وحبابك في شرق السودان وأرض البجا!",
        "northern": "مسكاقمي! إيقا كويي؟ مسكاجلو حبابك يا زول في أورون الشمالية والرطانة النوبية!"
    }

    while True:
        try:
            region = input("\nاختر المنطقة (khartoum / darfur / kordofan / eastern / northern): ").strip().lower()
            if region in ["exit", "quit", "خروج"]:
                break
            if not region:
                region = "khartoum"
                
            prompt = input("اكتب سؤالك أو الكلمة التي تبحث عنها: ").strip()
            if prompt in ["exit", "quit", "خروج"]:
                break
            if not prompt:
                continue

            results = vs.search(prompt, region=region, top_k=3)
            greeting = regional_greetings.get(region, "حبابك عشرة يا زول!")
            
            print("\n🤖 [إجابة المساعد الأوفلاين]:")
            if results and results[0].get("score", 0) > 0.5:
                top_text = results[0].get("text", "")
                print(f"{greeting} بالنسبة لسؤالك يا حبيب: {top_text}")
            else:
                print(f"{greeting} كيف أقدر أساعدك الليلة بخصوص كلام وثقافة {region}؟")
                
        except KeyboardInterrupt:
            print("\nمع السلامة!")
            break

if __name__ == "__main__":
    main()
