#!/usr/bin/env python3
"""
Comprehensive Sudanese Dialect Lexicon & Dictionary Database Builder
----------------------------------------------------------------------
Constructs a structured, reliable dictionary database of Sudanese words, phrases, 
proverbs, idioms, and regional terms across:
- الخرطوم والوسط (Khartoum & Central)
- دارفور (Darfur)
- كردفان (Kordofan)
- الشرق (Eastern / Beja & Kassala)
- الشمال / النوبة والرطانات (Northern / Dongolawi, Mahas, Nobiin Rotana)

Outputs:
1. SQLite Database: data/processed/sudanese_lexicon.db (For backend SQL queries, APIs, and direct DB dependence)
2. JSON Database: data/processed/sudanese_dictionary_database.json (For web apps, vector indexing, and LLM fine-tuning)
3. CSV File: data/processed/llm_dataset/sudanese_lexicon.csv (For pandas/Excel analysis)
"""

import sqlite3
import json
import csv
from pathlib import Path

# Comprehensive Lexicon Dictionary Data
SUDANESE_LEXICON_ENTRIES = [
    # --- KHARTOUM & CENTRAL ---
    {"term": "زول", "meaning": "شخص أو إنسان", "region": "khartoum", "category": "اسم / وصف", "example": "الزول دا طيب شديد", "phonetic": "Zool"},
    {"term": "حبابك عشرة", "meaning": "ترحيب حار جداً ومضاعف بالضيف", "region": "khartoum", "category": "ترحيب / تحية", "example": "حبابك عشرة بلا كشرة تفضل داخل", "phonetic": "Hababak Ashara"},
    {"term": "آزول", "meaning": "نداء للشخص بكل احترام وتودد", "region": "khartoum", "category": "نداء", "example": "آزول تعال هنا دقيقة", "phonetic": "A Zool"},
    {"term": "شديد ونظيف", "meaning": "وصف للشيء الممتاز جداً أو الشخص الأنيق والصادق", "region": "khartoum", "category": "تعبير مجازي", "example": "الشغل دا طلع شديد ونظيف", "phonetic": "Shadeed wa Nazeef"},
    {"term": "قاعد في أمان الله", "meaning": "في حال سكون وسلام دون إزعاج", "region": "khartoum", "category": "تعبير مجازي", "example": "كنت قاعد في أمان الله وفجأة جاني اتصال", "phonetic": "Ga'ed fi aman Allah"},
    {"term": "عنقريب", "meaning": "سرير تقليدي سوداني مصنوع من الخشب والمحبوك بالحبال أو القماش", "region": "khartoum", "category": "أثاث وتراث", "example": "اتفضل استريح في العنقريب دا", "phonetic": "Angareeb"},
    {"term": "بنسبرين", "meaning": "الماء البارد جداً المصحوب بالراحة (تعبير شعبي عن الانتعاش)", "region": "khartoum", "category": "تعبير شعبية", "example": "جيب لينا كباية موية مثلجة بنسبرين", "phonetic": "Bensabreen"},
    {"term": "أقيف على حيلك", "meaning": "قف وانتصب واقفاً", "region": "khartoum", "category": "أمر / حركة", "example": "أقيف على حيلك سلم على الضيوف", "phonetic": "Ageef ala heelak"},
    {"term": "يا زول انتميت لشنو", "meaning": "ما هو أصلك أو وجهتك أو هدفك", "region": "khartoum", "category": "سؤال شعبي", "example": "يا زول انتميت لشنو في الموضوع دا؟", "phonetic": "Ya zool antamait li shno"},
    {"term": "جبنة", "meaning": "قهوة سودانية تقليدية معتقة بالبهارات تُصنع في القلاية وتُصب في الفنجان", "region": "khartoum", "category": "مشروبات وتراث", "example": "تعال نشرب الجبنة في الضلاية", "phonetic": "Jabana"},
    {"term": "سجم أمي", "meaning": "تعبير عن الصدمة أو الدهشة أو الحسرة", "region": "khartoum", "category": "تعبير عن المشاعر", "example": "سجم أمي النور قطع تاني!", "phonetic": "Sajam Ummi"},
    {"term": "أبشر بالخير", "meaning": "عبارة تشجيع وتضامن وكرم سوداني أصيل", "region": "khartoum", "category": "ترحيب / كرم", "example": "أبشر بالخير والكرامة، حاجتك مقضية", "phonetic": "Absher bil khair"},
    {"term": "يا حليل أيام زمان", "meaning": "التغني بالحنين والاشتياق للماضي الجميلة", "region": "khartoum", "category": "حنين / شعر", "example": "يا حليل أيام زمان واللمة الطيبة", "phonetic": "Ya haleel ayam zaman"},
    {"term": "راكوبة", "meaning": "مظلة مصنوعة من القش والحطب توفر الظل والبرودة", "region": "khartoum", "category": "عمارة تقليدية", "example": "عدنا للجلسة تحت الراكوبة في العصرية", "phonetic": "Rakooba"},
    {"term": "قدامك العافية", "meaning": "دعاء للشخص بالشفاء والسلامة والخير", "region": "khartoum", "category": "دعاء / تحية", "example": "سلامتك يا أخي وقدامك العافية", "phonetic": "Gaddamak al aafiya"},

    # --- DARFUR ---
    {"term": "عوافي", "meaning": "تحية السلام والراحة والصحة والعافية", "region": "darfur", "category": "تحية", "example": "عوافي عليكم يا أهل الدار", "phonetic": "Awafi"},
    {"term": "شِدِيدْ", "meaning": "بخير وصحة جيدة وقوة", "region": "darfur", "category": "حالة / وصف", "example": "كيف حالكم؟ - شديدين والحمد لله", "phonetic": "Shadeed"},
    {"term": "حَبَابَكْ فِيمَا جِئْتَ", "meaning": "ترحيب كامل ومطلق بمقدم الضيف", "region": "darfur", "category": "ترحيب", "example": "حبابك فيما جئت ونورت الفاشر", "phonetic": "Hababak feema ji'ta"},
    {"term": "تَكَلْ", "meaning": "المطبخ التقليدي المصنوع من القش والخشب للإعداد والمأكولات", "region": "darfur", "category": "مكان / تراث", "example": "الأكل جاهز في التكل تعالوا", "phonetic": "Takal"},
    {"term": "الدامرة", "meaning": "المستقر والقرية الموسمية للمستقرين والرعاة", "region": "darfur", "category": "جغرافيا شعبية", "example": "وصلنا الدامرة قبل غروب الشمس", "phonetic": "Al Damira"},
    {"term": "الفرقان", "meaning": "جمع فريق، وهي أحياء ومجموعات منازل العائلات المترابطة", "region": "darfur", "category": "جغرافيا شعبية", "example": "الفرقان في دارفور مترابطة وتشارك في كل المناسبات", "phonetic": "Al Furgan"},
    {"term": "المَرْسُوعْ", "meaning": "الطبق التقليدي المصنوع بعناية لتقديم عصيدة الدخن", "region": "darfur", "category": "أواني ومأكولات", "example": "قدموا العصيدة في المرسوع الكبيرة", "phonetic": "Al Marsoo'"},
    {"term": "عصيدة الدخن", "meaning": "الوجبة الرئيسة والتاريخية في دارفور المصنوعة من طحين الدخن الصحي", "region": "darfur", "category": "مأكولات شعبية", "example": "عصيدة الدخن مع المراس هي القوت الرئيسي", "phonetic": "AseEDAT al Dukhn"},
    {"term": "الكَمَالتِي", "meaning": "المكافأة أو العطاء الإضافي إكراماً وتقديراً", "region": "darfur", "category": "كرم / تقاليد", "example": "أعطاه الكمالتي فوق حقه إكراماً له", "phonetic": "Al Kamalti"},
    {"term": "دار صباح", "meaning": "مصطلح يشير للخرطوم والمناطق الشرقية والوسطى بالسودان", "region": "darfur", "category": "اتجاهات / جغرافيا", "example": "مسافرين دار صباح الأسبوع الجاي", "phonetic": "Dar Sabah"},

    # --- KORDOFAN ---
    {"term": "أَبُو جَالُوفْ", "meaning": "لقب يطلق على الرجل الشجاع والكريم القوي", "region": "kordofan", "category": "صفة / مدح", "example": "ود أب جالوف ما برجع الضيف خايب", "phonetic": "Abu Jaloof"},
    {"term": "المردوم", "meaning": "الرقصة والغناء التراثي الإيقاعي الشهير في كردفان", "region": "kordofan", "category": "فنون وتراث", "example": "ضربوا إيقاع المردوم في الحفلة والناس فرحانة", "phonetic": "Al Mardoom"},
    {"term": "جَرَّارِي", "meaning": "نوع من الغناء والتراث البدوي الوجداني الأصيل بكردفان", "region": "kordofan", "category": "فنون وتراث", "example": "غنى صوت الجراري في صحراء كردفان", "phonetic": "Jarrari"},
    {"term": "القُرَاضَة", "meaning": "الصمغ العربي الممتاز الناتج من أشجار الهشاب", "region": "kordofan", "category": "طبيعة وزراعة", "example": "كردفان هي موطن الصمغ العربي والقراضة الممتازة", "phonetic": "Al Guradha"},
    {"term": "البَخْسَة", "meaning": "وعاء تقليدي مصنوع من القرع اليابس لحفظ وتبريد اللبن الرائب", "region": "kordofan", "category": "أواني وتراث", "example": "اللبن الرائب بارد في البخسة", "phonetic": "Al Bakhsa"},
    {"term": "حَبَابْ المَنْقُوقْ", "meaning": "أهلاً وسهلاً بالشخص العزيز القادم من سفر بعيد", "region": "kordofan", "category": "ترحيب", "example": "حباب المنقوق اللي جانا بعد غيبة", "phonetic": "Habab al Mangooq"},
    {"term": "شَجَرَة الهَشَابْ", "meaning": "الشجرة التاريخية المصدرة لأجود أنواع الصمغ العربي في العالم", "region": "kordofan", "category": "طبيعة وزراعة", "example": "غابات الهشاب تمتد في عروس الرمل الأبيض", "phonetic": "Shajarat al Hashab"},
    {"term": "السَّعِينْ", "meaning": "قربة الجلد المصنوعة لحفظ الماء البارد أثناء التنقل", "region": "kordofan", "category": "أواني وتراث", "example": "شربنا من السعين موية عذبة وباردة", "phonetic": "Al Sa'een"},

    # --- EASTERN (BEJA & KASSALA) ---
    {"term": "دابايوا", "meaning": "تحية السلام والترحيب بلغة البجا (البداويت) في شرق السودان", "region": "eastern", "category": "تحية / لغة بجاوية", "example": "دابايوا بكل زائر لمدينة كسلا والشرق", "phonetic": "Dabaywa"},
    {"term": "دابايوا هَضَلْبَا", "meaning": "مرحباً بكم وأهلاً وسهلاً بأعلى درجات الحفاوة", "region": "eastern", "category": "ترحيب بجاوي", "example": "دابايوا هضلبا في جبال التاكا", "phonetic": "Dabaywa Hadalba"},
    {"term": "السِّيفَابْ", "meaning": "النسمات والرياح العليلة الباردة القادمة من جهة البحر والجبال", "region": "eastern", "category": "طبيعة وثقافة", "example": "السيفاب عليل في عصريات كسلا", "phonetic": "Al Seefab"},
    {"term": "السَّمْبَلْ", "meaning": "الجمال والأناقة والهدوء في المشي والحركة", "region": "eastern", "category": "صفة شعبية", "example": "زول سمبل ومحترم في تعامله", "phonetic": "Al Sambal"},
    {"term": "جَبَنَة الشَّرْقْ", "meaning": "طريقة إعداد القهوة البجاوية في الجبنة الفخارية مع الزنجبيل والبهارات", "region": "eastern", "category": "مشروبات وتراث", "example": "جبنة الشرق بالزنجبيل ما ليها مثيل", "phonetic": "Jabana al Sharq"},
    {"term": "جَبَل التَّاكَا", "meaning": "المعلم الطبيعي والتاريخي الأبرز بمدينة كسلا بشرق السودان", "region": "eastern", "category": "معالم ومعتقدات", "example": "جبل التاكا يزين مدينة كسلا الشامخة", "phonetic": "Jabal al Taka"},
    {"term": "السِّدْرَة", "meaning": "جلسة التسامح والشورى بين شيوخ القبائل لحل القضايا", "region": "eastern", "category": "إدارة شعبية / أعراف", "example": "اجتمع الشيوخ في السدرة لحل المشكلة بالود", "phonetic": "Al Sidra"},

    # --- NORTHERN & NUBIAN ROTANA ---
    {"term": "مسكاقمي", "meaning": "تحية السلام بالنوبية (الدنقلاوية والمحسية والنوبين) وتعني: كيف حالكم؟", "region": "northern", "category": "تحية نوبية", "example": "مسكاقمي يا أهلنا في دنقلا وحلفا", "phonetic": "Meskaqmi"},
    {"term": "إيقا كويي", "meaning": "عبارة بالنوبية تعني: هل أنت بخير وبصحة جيدة؟", "region": "northern", "category": "سؤال عن الحال", "example": "إيقا كويي؟ - الحمد لله كويي شديد", "phonetic": "Iga Kooyee"},
    {"term": "إسي", "meaning": "الماء بالنوبية (الرطانة النوبية الأصيلة)", "region": "northern", "category": "كلمات نوبية", "example": "جيب لينا إسي بارد نشرب", "phonetic": "Essi"},
    {"term": "إكسي", "meaning": "الطعام والعيش بالنوبية", "region": "northern", "category": "كلمات نوبية", "example": "الإكسي جاهز اتفضلوا للغداء", "phonetic": "Eksi"},
    {"term": "أميتي", "meaning": "الأخ أو الصديق العزيز بالنوبية", "region": "northern", "category": "علاقات ومحبة", "example": "يا أميتي حبابك في بلدك", "phonetic": "Ameeti"},
    {"term": "تربل", "meaning": "المزارع الصبور الفلاح المرتبط بالأرض والنيل في الشمال", "region": "northern", "category": "مهن وتراث", "example": "التربل يصحى من الفجر لخدمة الجرف والنخيل", "phonetic": "Turbal"},
    {"term": "السَّاقِيَة", "meaning": "الآلة المائية التاريخية المبتكرة في الشمال لرفع مياه النيل لزراعة الضفاف", "region": "northern", "category": "تراث وزراعة", "example": "صوت الساقية نغم سوداني أصيل في الشمال", "phonetic": "Al Saqia"},
    {"term": "الجُرُفْ", "meaning": "الأرض الخصبة المتاخمة لشاطئ النيل التي تظهر بعد انحسار الفيضان", "region": "northern", "category": "زراعة وجغرافيا", "example": "زرعنا اللوبيا والخضار في الجرف", "phonetic": "Al Juruf"},
    {"term": "الكُولِي", "meaning": "المفتاح الخشبي التقليدي للبيوت النوبية القديمة", "region": "northern", "category": "أدوات وتراث", "example": "قفل الباب بالكولي الخشبي القديم", "phonetic": "Al Kooli"},
    {"term": "القَرَقُورْ", "meaning": "الزورق والقارب الصغير المخصص للتنقل والصيد بالنيل", "region": "northern", "category": "ملاحة نيلية", "example": "عبرنا النيل بالقرقور للضفة الغربية", "phonetic": "Al Qaragoor"},
    {"term": "التَّقَرّابْ", "meaning": "جلسة التآنس السمرية الشتوية حول النار والنخيل", "region": "northern", "category": "تقاليد اجتماعية", "example": "اتجمعنا في التقّراب مع كبار السن نحكي القصص", "phonetic": "Al Taqarrab"}
]

def build_lexicon_database():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    llm_dir = Path("data/processed/llm_dataset")
    llm_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "sudanese_lexicon.db"
    json_path = output_dir / "sudanese_dictionary_database.json"
    csv_path = llm_dir / "sudanese_lexicon.csv"

    print("🚀 Constructing Sudanese Lexicon & Dictionary Database...")

    # 1. Build SQLite Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lexicon (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT NOT NULL,
            meaning TEXT NOT NULL,
            region TEXT NOT NULL,
            category TEXT NOT NULL,
            example TEXT,
            phonetic TEXT
        )
    """)

    cursor.execute("DELETE FROM lexicon") # Fresh populate

    for entry in SUDANESE_LEXICON_ENTRIES:
        cursor.execute("""
            INSERT INTO lexicon (term, meaning, region, category, example, phonetic)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry["term"],
            entry["meaning"],
            entry["region"],
            entry["category"],
            entry.get("example", ""),
            entry.get("phonetic", "")
        ))

    conn.commit()
    conn.close()
    print(f"✅ SQLite Lexicon DB Created: {db_path} ({len(SUDANESE_LEXICON_ENTRIES)} terms)")

    # 2. Save JSON Database
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(SUDANESE_LEXICON_ENTRIES, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON Dictionary DB Created: {json_path}")

    # 3. Save CSV Database
    fieldnames = ["term", "meaning", "region", "category", "example", "phonetic"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(SUDANESE_LEXICON_ENTRIES)
    print(f"✅ CSV Lexicon Exported: {csv_path}")

if __name__ == "__main__":
    build_lexicon_database()
