# ============================================================
#  🤖 RAG System — نظام الذكاء الاصطناعي للأسئلة والأجوبة
#  
#  RAG = Retrieval Augmented Generation
#  الفكرة البسيطة:
#    1. ارفع ملفاتك (PDF / Excel / CSV)
#    2. النظام يقرأها ويقسمها لـ "قطع صغيرة" (Chunks)
#    3. لما تسأل سؤال، النظام يلاقي القطع الأقرب لسؤالك
#    4. يبعتها لـ Gemini مع سؤالك → يجيب إجابة ذكية
# ============================================================

import streamlit as st          # إطار عمل الواجهة
import google.generativeai as genai  # مكتبة Gemini
import pandas as pd              # قراءة Excel و CSV
import pdfplumber                # قراءة PDF
import numpy as np               # عمليات رياضية على الـ Vectors
import io, os, json, time, re
from collections import Counter


# ══════════════════════════════════════════════════════════
#  🔄 Retry helper — يعيد المحاولة عند تجاوز الحصة
# ══════════════════════════════════════════════════════════

def call_gemini_with_retry(func, max_retries=5, base_wait=10):
    """
    يُعيد الاستدعاء تلقائياً لو حصل ResourceExhausted (Rate Limit)
    base_wait: ثواني الانتظار قبل كل إعادة محاولة (يتضاعف تدريجياً)
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err = str(e)
            is_quota = "ResourceExhausted" in err or "quota" in err.lower() or "429" in err
            if is_quota and attempt < max_retries - 1:
                wait_time = base_wait * (2 ** attempt)   # 10s → 20s → 40s → 80s
                st.warning(f"⏳ تجاوز الحصة — انتظار {wait_time} ثانية ثم إعادة المحاولة ({attempt+1}/{max_retries-1})...")
                time.sleep(wait_time)
            else:
                raise   # إذا مش quota error، أو انتهت المحاولات → أطلق الخطأ

# ══════════════════════════════════════════════════════════
#  إعداد الصفحة
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Agent — نظام الأسئلة الذكي",
    page_icon="🤖",
    layout="wide",
)

# ── CSS (ثيم داكن + عربي) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');

:root {
  --bg: #0a0a1a; --surface: #12122a; --card: #1a1a35;
  --border: #2a2a55; --accent: #6366f1; --accent2: #22d3ee;
  --text: #e2e8f0; --muted: #64748b; --green: #34d399; --red: #f87171;
}
html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--surface); border-left: 1px solid var(--border); }

/* Chat bubbles */
.msg-user {
  background: linear-gradient(135deg, #4f46e5, #6366f1);
  color: white; border-radius: 18px 18px 4px 18px;
  padding: 12px 18px; margin: 8px 0 8px 60px;
  text-align: right; direction: rtl; line-height: 1.7;
}
.msg-ai {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 18px 18px 18px 4px;
  padding: 14px 18px; margin: 8px 60px 8px 0;
  text-align: right; direction: rtl; line-height: 1.9;
}
.msg-ai-label { color: var(--accent2); font-size: 0.75rem; font-weight: 700; margin-bottom: 6px; }
.msg-user-label { color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-bottom: 6px; text-align: left; }

/* Source cards */
.source-card {
  background: var(--surface); border: 1px solid var(--border);
  border-right: 3px solid var(--accent2);
  border-radius: 8px; padding: 10px 14px;
  margin: 4px 0; font-size: 0.8rem; color: var(--muted);
  direction: rtl; text-align: right;
}

/* KPI */
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; text-align: center; }
.kpi-val { font-size: 1.8rem; font-weight: 900; color: var(--accent2); }
.kpi-lbl { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }

/* Status */
.status-ok  { color: var(--green); font-weight: 700; }
.status-err { color: var(--red);   font-weight: 700; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #4f46e5) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  font-family: 'Cairo', sans-serif !important; font-weight: 700 !important;
  width: 100%;
}
div[data-testid="stTextInput"] input {
  background: var(--card) !important; color: var(--text) !important;
  border-color: var(--border) !important; font-family: 'Cairo', sans-serif !important;
  direction: rtl; text-align: right;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  📚 الخطوة 1: قراءة الملفات واستخراج النصوص
# ══════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes):
    text_pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                text_pages.append({
                    "content": text.strip(),
                    "source": f"صفحة {i+1}"
                })
    return text_pages


def extract_text_from_table(file_bytes, filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    chunks = []
    summary = f"""
    ملخص الجدول: {filename}
    عدد الصفوف: {len(df):,}
    عدد الأعمدة: {len(df.columns)}
    الأعمدة: {', '.join(df.columns.tolist())}
    
    إحصائيات سريعة:
    {df.describe(include='all').to_string()}
    """
    chunks.append({"content": summary, "source": f"{filename} — ملخص"})

    chunk_size = 50
    for start in range(0, len(df), chunk_size):
        chunk_df = df.iloc[start : start + chunk_size]
        chunks.append({
            "content": chunk_df.to_string(index=False),
            "source": f"{filename} — صفوف {start+1} إلى {start+len(chunk_df)}"
        })
    return chunks


# ══════════════════════════════════════════════════════════
#  ✂️ الخطوة 2: تقسيم النصوص لـ Chunks
# ══════════════════════════════════════════════════════════

def split_into_chunks(text, source, chunk_size=800, overlap=100):
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1

        if current_size >= chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append({"content": chunk_text, "source": source})
            overlap_words = current_chunk[-overlap//5:]
            current_chunk = overlap_words
            current_size = sum(len(w) + 1 for w in overlap_words)

    if current_chunk:
        chunks.append({"content": " ".join(current_chunk), "source": source})
    return chunks


# ══════════════════════════════════════════════════════════
#  🔢 الخطوة 3: تحويل النص لـ Embedding (Vector)
# ══════════════════════════════════════════════════════════

def get_embedding(text, model):
    result = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document"
    )
    return np.array(result["embedding"])


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


# ══════════════════════════════════════════════════════════
#  🔁 الخطوة 4A: Query Expansion — توسيع السؤال
# ══════════════════════════════════════════════════════════

def expand_query(question, gemini_model):
    prompt = f"""أنت متخصص في تحليل الأسئلة.
السؤال: "{question}"

اكتب 3 أسئلة مختلفة تحمل نفس المعنى أو تكمّله.
أجب فقط بـ JSON بهذا الشكل (بدون أي نص إضافي):
{{"questions": ["السؤال 1", "السؤال 2", "السؤال 3"]}}"""

    try:
        response = call_gemini_with_retry(lambda: gemini_model.generate_content(prompt))
        text = response.text.strip()
        text = re.sub(r"```json|```", "", text).strip()
        data = json.loads(text)
        expanded = data.get("questions", [])
        return [question] + expanded[:3]
    except:
        return [question]


# ══════════════════════════════════════════════════════════
#  🔍 الخطوة 4B: Hybrid Search — بحث مزدوج
# ══════════════════════════════════════════════════════════

def keyword_search(question, vector_db, top_k=8):
    question_words = set(
        w.strip("؟!.,،") for w in question.split()
        if len(w) > 2
    )
    if not question_words:
        return []

    scored = []
    for chunk in vector_db:
        chunk_words = set(chunk["content"].split())
        common = question_words & chunk_words
        score = len(common) / len(question_words) if question_words else 0
        if score > 0:
            scored.append({**chunk, "score": score, "search_type": "keyword"})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def hybrid_search(questions, vector_db, embed_model, top_k=8):
    seen_ids  = set()
    all_results = []

    for q in questions:
        try:
            q_vec = call_gemini_with_retry(lambda: genai.embed_content(
                model=embed_model,
                content=q,
                task_type="retrieval_query"
            )["embedding"])
            q_vec = np.array(q_vec)

            for chunk in vector_db:
                chunk_id = chunk["content"][:50]
                if chunk_id not in seen_ids:
                    score = cosine_similarity(q_vec, np.array(chunk["embedding"]))
                    all_results.append({**chunk, "score": score, "search_type": "vector"})
                    seen_ids.add(chunk_id)
        except:
            pass

        kw_results = keyword_search(q, vector_db, top_k=top_k)
        for r in kw_results:
            chunk_id = r["content"][:50]
            if chunk_id not in seen_ids:
                all_results.append(r)
                seen_ids.add(chunk_id)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


# ══════════════════════════════════════════════════════════
#  🏆 الخطوة 4C: Reranker — إعادة الترتيب
# ══════════════════════════════════════════════════════════

def rerank_chunks(question, chunks, gemini_model, top_k=3):
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\n[{i}] المصدر: {chunk['source']}\n{chunk['content'][:300]}\n"

    prompt = f"""السؤال: "{question}"
القطع النصية التالية مسترجعة من قاعدة البيانات:
{chunks_text}

رتّب أرقام القطع من الأكثر صلة للأقل صلة بالسؤال.
أجب فقط بـ JSON (بدون أي نص إضافي):
{{"ranked": [أرقام مرتبة, مثل: 2, 0, 5, 1, ...]}}"""

    try:
        response = call_gemini_with_retry(lambda: gemini_model.generate_content(prompt))
        text = response.text.strip()
        text = re.sub(r"```json|```", "", text).strip()
        data = json.loads(text)
        ranked_indices = data.get("ranked", [])

        reranked = []
        for idx in ranked_indices:
            if isinstance(idx, int) and 0 <= idx < len(chunks):
                reranked.append({**chunks[idx], "rerank_pos": len(reranked) + 1})

        ranked_set = set(ranked_indices)
        for i, chunk in enumerate(chunks):
            if i not in ranked_set:
                reranked.append({**chunk, "rerank_pos": len(reranked) + 1})
        return reranked[:top_k]
    except:
        return chunks[:top_k]


# ══════════════════════════════════════════════════════════
#  🏗️ الخطوة 4D: Context Builder — بناء السياق
# ══════════════════════════════════════════════════════════

def build_context(reranked_chunks):
    context_parts = []
    for chunk in reranked_chunks:
        rank  = chunk.get("rerank_pos", "?")
        stype = chunk.get("search_type", "vector")
        part  = (
            f"【مصدر {rank}】 {chunk['source']} "
            f"| نوع البحث: {'متجه' if stype=='vector' else 'كلمات مفتاحية'} "
            f"| تشابه: {chunk['score']:.0%}\n"
            f"{chunk['content']}"
        )
        context_parts.append(part)
    return "\n\n" + "─" * 40 + "\n\n".join(context_parts)


# ══════════════════════════════════════════════════════════
#  💬 الخطوة 5: توليد الإجابة بـ Gemini
# ══════════════════════════════════════════════════════════

def generate_answer(question, relevant_chunks, chat_history, gemini_model):
    context = build_context(relevant_chunks)
    system_prompt = """أنت مساعد ذكي متخصص في تحليل البيانات وتقارير الأعمال.
مهمتك: الإجابة على أسئلة المستخدم بناءً على البيانات المقدمة فقط.

قواعد مهمة:
- استخدم فقط المعلومات الموجودة في السياق المقدم
- لو السؤال مش موجود في البيانات، قول ذلك بوضوح
- قدّم الأرقام بشكل منظم مع جداول لو مناسب
- الإجابة بالعربية دائماً ما لم يسأل المستخدم بلغة أخرى
- كن دقيقاً ومختصراً"""

    history_text = ""
    for msg in chat_history[-4:]:
        role = "المستخدم" if msg["role"] == "user" else "المساعد"
        history_text += f"{role}: {msg['content']}\n"

    full_prompt = f"""{system_prompt}
══ السياق المسترجع من ملفاتك ══
{context}
══ تاريخ المحادثة ══
{history_text}
══ السؤال الحالي ══
{question}
الإجابة:"""

    response = call_gemini_with_retry(lambda: gemini_model.generate_content(full_prompt))
    return response.text


# ══════════════════════════════════════════════════════════
#  🗄️ إدارة الـ Session State
# ══════════════════════════════════════════════════════════

if "vector_db"     not in st.session_state: st.session_state.vector_db     = []
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "files_loaded"  not in st.session_state: st.session_state.files_loaded  = []
if "gemini_ready"  not in st.session_state: st.session_state.gemini_ready  = False
if "api_key"       not in st.session_state: st.session_state.api_key       = ""


# ══════════════════════════════════════════════════════════
#  🎨 الواجهة — Sidebar
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0; border-bottom:1px solid #2a2a55; margin-bottom:20px;">
      <div style="font-size:2.5rem;">🤖</div>
      <div style="font-size:1.1rem; font-weight:700; color:#6366f1;">AI Agent</div>
      <div style="font-size:0.75rem; color:#64748b;">نظام الأسئلة الذكي</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔑 Google Gemini API Key")
    api_key_input = st.text_input(
        "أدخل الـ API Key",
        type="password",
        placeholder="AIza...",
        value=st.session_state.api_key,
        label_visibility="collapsed",
        key="api_key_field"
    )

    if st.button("✅ تفعيل API Key"):
        if api_key_input and api_key_input.strip():
            try:
                genai.configure(api_key=api_key_input.strip())
                # تحديث النموذج المستخدم للاختبار لمنع خطأ 404
                test_model = genai.GenerativeModel("gemini-1.5-flash")
                test_model.generate_content("test")
                st.session_state.api_key      = api_key_input.strip()
                st.session_state.gemini_ready = True
                st.success("✅ تم التفعيل بنجاح!")
                st.rerun()
            except Exception as e:
                st.session_state.gemini_ready = False
                st.error(f"❌ خطأ: {str(e)[:100]}")
        else:
            st.warning("⚠️ أدخل الـ API Key أولاً")

    # عرض حالة الاتصال
    if st.session_state.gemini_ready:
        st.markdown("<div class='status-ok'>🟢 متصل</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-err'>🔴 غير متصل</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── رفع الملفات ──
    st.markdown("### 📁 ارفع ملفاتك")
    uploaded_files = st.file_uploader(
        "PDF / Excel / CSV",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.gemini_ready:
        if st.button("🚀 معالجة الملفات"):
            # تحديث نموذج الـ Embedding للأحدث لمنع خطأ 404
            EMBED_MODEL = "models/text-embedding-004"
            progress = st.progress(0, text="جاري المعالجة...")
            all_chunks = []

            for file_idx, file in enumerate(uploaded_files):
                file_bytes = file.read()
                file_name  = file.name
                progress.progress((file_idx) / len(uploaded_files), text=f"📄 جاري قراءة: {file_name}")

                raw_chunks = []
                if file_name.endswith(".pdf"):
                    pages = extract_text_from_pdf(file_bytes)
                    for page in pages:
                        sub_chunks = split_into_chunks(page["content"], page["source"])
                        raw_chunks.extend(sub_chunks)
                else:
                    raw_chunks = extract_text_from_table(file_bytes, file_name)

                for i, chunk in enumerate(raw_chunks):
                    try:
                        # استخدام دالة الريتراي مع الـ Embeddings لمنع خطأ 429
                        embedding = call_gemini_with_retry(lambda: get_embedding(chunk["content"], EMBED_MODEL))
                        chunk["embedding"] = embedding.tolist()
                        all_chunks.append(chunk)
                        time.sleep(2)  # انتظار بسيط للتأكد من استقرار الحصة
                    except Exception as e:
                        st.warning(f"⚠️ تخطي chunk: {str(e)[:40]}")

                st.session_state.files_loaded.append(file_name)

            st.session_state.vector_db = all_chunks
            progress.progress(1.0, text="✅ اكتملت المعالجة!")
            st.rerun()

    if st.button("🗑️ مسح كل شيء"):
        for key in ["vector_db", "chat_history", "files_loaded"]:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else {}
        st.rerun()


# ══════════════════════════════════════════════════════════
#  🎨 الواجهة الرئيسية — Chat
# ══════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#12122a,#1a1a35); border:1px solid #2a2a55;
     border-radius:16px; padding:24px 32px; margin-bottom:24px; text-align:right;">
  <div style="font-size:1.8rem; font-weight:900; background:linear-gradient(90deg,#6366f1,#22d3ee);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🤖 المساعد الذكي لبياناتك
  </div>
</div>
""", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        role_class = "msg-user" if msg["role"] == "user" else "msg-ai"
        label = "أنت" if msg["role"] == "user" else "🤖 المساعد الذكي"
        st.markdown(f"""<div class="{role_class}"><div class="msg-ai-label">{label}</div>{msg["content"]}</div>""", unsafe_allow_html=True)

if "input_counter" not in st.session_state: st.session_state.input_counter = 0

col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input("سؤالك", placeholder="اسأل أي سؤال عن بياناتك...", label_visibility="collapsed", key=f"qi_{st.session_state.input_counter}")
with col_btn:
    send = st.button("إرسال ✈️")

if (send and question) or False:
    if not st.session_state.gemini_ready or not st.session_state.vector_db:
        st.error("❌ تأكد من تفعيل الـ API Key ورفع الملفات")
        st.stop()

    genai.configure(api_key=st.session_state.api_key)
    # تحديث النماذج للأحدث
    EMBED_MODEL  = "models/text-embedding-004"
    GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

    with st.spinner("🔍 جاري البحث والتحليل..."):
        try:
            expanded_questions = expand_query(question, GEMINI_MODEL)
            time.sleep(2) # انتظار لتجنب 429
            candidate_chunks = hybrid_search(expanded_questions, st.session_state.vector_db, EMBED_MODEL, top_k=8)
            time.sleep(2) # انتظار لتجنب 429
            reranked = rerank_chunks(question, candidate_chunks, GEMINI_MODEL, top_k=3)
            time.sleep(2) # انتظار لتجنب 429
            answer = generate_answer(question, reranked, st.session_state.chat_history, GEMINI_MODEL)
            
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": reranked})
            st.session_state.input_counter += 1
            st.rerun()
        except Exception as e:
            st.error(f"❌ حدث خطأ: {str(e)}")

st.markdown("<div style='text-align:center; padding:24px; color:#334155; font-size:0.78rem;'>🤖 AI Agent — Built with Streamlit & Gemini</div>", unsafe_allow_html=True)
