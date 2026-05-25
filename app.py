import streamlit as st
import joblib
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Ujaran Kebencian SARA",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main { background-color: #f8fafc; }

    .title-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .title-box h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .title-box p  { font-size: 0.9rem; opacity: 0.85; margin: 0.5rem 0 0; }

    .result-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .result-nonhate  { background: #d1fae5; color: #065f46; border-left: 5px solid #10b981; }
    .result-hate     { background: #fee2e2; color: #991b1b; border-left: 5px solid #ef4444; }
    .result-sara     { background: #fef3c7; color: #92400e; border-left: 5px solid #f59e0b; }
    .result-umum     { background: #ffe4e6; color: #9f1239; border-left: 5px solid #f43f5e; }

    .confidence-bar {
        background: #e2e8f0;
        border-radius: 8px;
        height: 10px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
    }

    .step-badge {
        display: inline-block;
        background: #1e3a5f;
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .info-small {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ───────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "input"
if "result" not in st.session_state:
    st.session_state.result = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ── Load models ──────────────────────────────────────────────────
@st.cache_resource
def load_tahap1():
    MODEL_HF = "marsya24/indobert-hate-speech"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_tahap2():
    svm   = joblib.load("models/model_tahap2_svm.pkl")
    tfidf = joblib.load("models/tfidf_tahap2.pkl")
    le    = joblib.load("models/label_encoder_tahap2.pkl")
    return svm, tfidf, le

# ── Text cleaning ────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ── Prediction functions ─────────────────────────────────────────
def predict_tahap1(text, tokenizer, model):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", max_length=128,
                       truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return pred, conf

def predict_tahap2(text, svm, tfidf, le):
    cleaned = clean_text(text)
    vec  = tfidf.transform([cleaned])
    pred = svm.predict(vec)[0]
    label = le.inverse_transform([pred])[0]
    decision = svm.decision_function(vec)[0]
    conf = float(1 / (1 + np.exp(-abs(decision))))
    return label, conf

# ── Load with spinner ────────────────────────────────────────────
with st.spinner("Memuat model... (pertama kali mungkin ~1 menit)"):
    try:
        tokenizer, model_t1 = load_tahap1()
        svm, tfidf, le      = load_tahap2()
        models_loaded = True
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

# ── HALAMAN 1 — INPUT ────────────────────────────────────────────
if st.session_state.page == "input":

    st.markdown("""
    <div class="title-box">
        <h1>🔍 Deteksi Ujaran Kebencian SARA</h1>
        <p>Sistem deteksi dua tahap: Hate Speech → Klasifikasi SARA</p>
    </div>
    """, unsafe_allow_html=True)

    if models_loaded:
        st.success("✅ Model berhasil dimuat!")

    st.markdown("### 💬 Masukkan Komentar")
    user_input = st.text_area(
        label="",
        placeholder="Contoh: orang madura emang kasar banget...",
        height=130,
        value=st.session_state.user_input
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_btn = st.button("🔍 Analisis", use_container_width=True)

    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠️ Teks tidak boleh kosong.")
        else:
            with st.spinner("Menganalisis..."):
                pred1, conf1 = predict_tahap1(user_input, tokenizer, model_t1)
                label2, conf2 = None, None
                if pred1 == 1:
                    label2, conf2 = predict_tahap2(user_input, svm, tfidf, le)

            st.session_state.user_input = user_input
            st.session_state.result = {
                "pred1": pred1,
                "conf1": conf1,
                "label2": label2,
                "conf2": conf2,
                "cleaned": clean_text(user_input)
            }
            st.session_state.page = "result"
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>"
        "Sistem Deteksi Ujaran Kebencian SARA · Skripsi Sistem Informasi"
        "</div>",
        unsafe_allow_html=True
    )

# ── HALAMAN 2 — HASIL ────────────────────────────────────────────
elif st.session_state.page == "result":

    st.markdown("""
    <div class="title-box">
        <h1>🔍 Deteksi Ujaran Kebencian SARA</h1>
        <p>Sistem deteksi dua tahap: Hate Speech → Klasifikasi SARA</p>
    </div>
    """, unsafe_allow_html=True)

    r = st.session_state.result
    pred1  = r["pred1"]
    conf1  = r["conf1"]
    label2 = r["label2"]
    conf2  = r["conf2"]

    st.markdown("---")

    st.markdown('<span class="step-badge">TAHAP 1 — Deteksi Hate Speech</span>', unsafe_allow_html=True)

    if pred1 == 0:
        st.markdown(f"""
        <div class="result-box result-nonhate">
            ✅ Non-Hate Speech
            <div class="info-small">Confidence: {conf1*100:.1f}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{conf1*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("Komentar ini tidak terdeteksi sebagai ujaran kebencian.")

    else:
        st.markdown(f"""
        <div class="result-box result-hate">
            ⚠️ Hate Speech Terdeteksi
            <div class="info-small">Confidence: {conf1*100:.1f}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{conf1*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="step-badge">TAHAP 2 — Klasifikasi SARA</span>', unsafe_allow_html=True)

        if label2 == "HS_SARA":
            st.markdown(f"""
            <div class="result-box result-sara">
                🚨 Hate Speech SARA (Suku, Agama, Ras, Antargolongan)
                <div class="info-small">Confidence: {conf2*100:.1f}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{conf2*100:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-umum">
                ⚡ Hate Speech Umum (bukan SARA)
                <div class="info-small">Confidence: {conf2*100:.1f}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{conf2*100:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("📄 Lihat teks setelah preprocessing"):
        st.code(r["cleaned"])

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Analisis Teks Lain", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>"
        "Sistem Deteksi Ujaran Kebencian SARA · Skripsi Sistem Informasi"
        "</div>",
        unsafe_allow_html=True
    )
