# تنفيذ Task الـ Multilingual RAG - نسخة Starter عملية

هذا المشروع يعطيك **نسخة أولى شغالة** لتطبيق التاسك المطلوبة في الملف، مع تغطية:

- Phase 1: preprocessing + chunking + metadata + embeddings + FAISS + retrieval
- Phase 2 (نواة): query handling + history + ranking + fallback generation
- Phase 3 (نواة Bonus): FastAPI endpoints `/health`, `/ask-question`, `/evaluate`

## 1) لماذا هذا الهيكل مناسب؟

- **Natural Questions** أصلًا مكوّن من أسئلة حقيقية وإجابات قصيرة وطويلة من ويكيبيديا، وهو مناسب لبناء pipeline سؤال/جواب واسترجاع سياقي. citeturn252790search0turn166895search1turn166895search9
- **Sentence Transformers** توفر embeddings جاهزة للبحث الدلالي، والنموذج `paraphrase-multilingual-MiniLM-L12-v2` موصوف كنموذج متعدد اللغات ومُدرّب على أكثر من 50 لغة ويولّد متجهات 384 بعدًا؛ وهذا مناسب جدًا كنقطة بداية لبحث multilingual. citeturn252790search1turn166895search2turn166895search0
- **FAISS** مخصص للبحث بالكفاءة العالية داخل المتجهات dense vectors، وهو مناسب لبناء vector store محلي سريع. citeturn252790search2turn252790search10turn252790search22
- **FastAPI** إطار API عالي الأداء، ويدعم بناء REST endpoints بسهولة، كما أن توثيقه يوضح إمكان استخدام قواعد بيانات مدعومة عبر SQLAlchemy/SQLModel مثل SQLite وMySQL وPostgreSQL. citeturn252790search7turn252790search15turn252790search3turn252790search11

## 2) تثبيت المشروع

```bash
cd multilingual_rag_task
python -m venv .venv
source .venv/bin/activate   # على ويندوز: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## 3) بناء الفهرس لأول مرة

استخدم الداتا التجريبية المرفقة أولًا:

```bash
PYTHONPATH=. python scripts/build_index.py --input data/raw/sample_nq.csv
```

هذا الأمر ينفذ:
1. تحميل الداتا
2. تنظيف النصوص
3. استخراج السؤال + short answer + long answer
4. chunking
5. إضافة metadata
6. حساب embeddings
7. بناء FAISS index
8. حفظ metadata في JSONL

## 4) تجربة الـ CLI

```bash
PYTHONPATH=. python scripts/chat_cli.py
```

أمثلة:
- `Who wrote Hamlet?`
- `ما هي عاصمة مصر؟`
- `When did World War II end?`

## 5) تشغيل الـ API

```bash
PYTHONPATH=. uvicorn src.api:app --reload
```

ثم افتح:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### مثال request

```bash
curl -X POST "http://127.0.0.1:8000/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما هي عاصمة مصر؟",
    "top_k": 3
  }'
```

## 6) كيف تربطه مع LLM API مثل Groq؟

في ملف `.env` ضع:

```env
LLM_API_KEY=your_key_here
LLM_API_BASE=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
```

إذا لم تضف API key، النظام سيستخدم **fallback extractive answer** من أفضل chunk مسترجع.

## 7) كيف تحوّله إلى تنفيذ مطابق للتاسك أكثر؟

### Phase 1
- استبدل `sample_nq.csv` بداتا أكبر من Kaggle/JSONL
- أضف deduplication
- أضف language detection
- أضف metadata إضافية مثل source_url وquestion_type
- جرّب chunk sizes مختلفة

### Phase 2
- أضف query expansion بالمرادفات
- أضف reranker بعد FAISS
- خزّن history في قاعدة بيانات
- أضف caching
- أضف monitoring وlatency logs

### Phase 3
- أضف SQLAlchemy models لـ `queries`, `responses`, `sessions`
- خزّن analytics
- أضف evaluation dataset حقيقي لحساب `precision@k` و`recall@k`
- أضف BLEU/ROUGE لاحقًا إن احتجت تقييمًا نصيًا

## 8) أهم نقطة في العرض أو المناقشة

لا تقل فقط: “أنا استخدمت RAG”.
بل اشرح:
- كيف نظفت الداتا
- لماذا اخترت chunk size معين
- لماذا هذا الـ embedding model مناسب متعدد اللغات
- كيف استخدمت FAISS
- كيف تتعامل مع low-confidence answers
- كيف ستقيس retrieval quality

## 9) ما الذي تحتاج إضافته الآن لو كنت ستسلمه؟

أقرب نسخة قوية للتسليم:
1. داتا أكبر حقيقية
2. تقرير preprocessing واضح
3. مقارنة بين أكثر من chunking strategy
4. تقييم retrieval على subset معروف
5. فيديو أو لقطات API + CLI

