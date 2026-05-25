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

    .stApp {
        background: linear-gradient(-45deg, #e8f0e9, #d4ece8, #f5f0e8, #c9e4de, #eef4f0);
        background-size: 400% 400%;
        animation: gradientShift 14s ease infinite;
        min-height: 100vh;
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .bg-orbs {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(90px);
        opacity: 0.22;
        animation: floatOrb linear infinite;
    }
    .orb1 { width: 450px; height: 450px; background: #7dbfb0; top: -120px; left: -120px; animation-duration: 20s; }
    .orb2 { width: 320px; height: 320px; background: #a8d5c2; top: 35%; right: -80px; animation-duration: 26s; animation-delay: -9s; }
    .orb3 { width: 280px; height: 280px; background: #c9b46e; bottom: -60px; left: 25%; animation-duration: 18s; animation-delay: -5s; }
    .orb4 { width: 220px; height: 220px; background: #6aab9c; top: 65%; left: 8%; animation-duration: 23s; animation-delay: -13s; }
    .orb5 { width: 180px; height: 180px; background: #e8d5a3; top: 20%; left: 55%; animation-duration: 19s; animation-delay: -7s; }

    @keyframes floatOrb {
        0%   { transform: translateY(0px) translateX(0px) scale(1); }
        33%  { transform: translateY(-40px) translateX(30px) scale(1.05); }
        66%  { transform: translateY(20px) translateX(-20px) scale(0.95); }
        100% { transform: translateY(0px) translateX(0px) scale(1); }
    }

    .stars {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    .star {
        position: absolute;
        width: 3px; height: 3px;
        background: #c9b46e;
        border-radius: 50%;
        animation: twinkle ease-in-out infinite;
        opacity: 0;
    }
    @keyframes twinkle {
        0%, 100% { opacity: 0; transform: scale(1); }
        50%       { opacity: 0.5; transform: scale(1.8); }
    }

    .block-container {
        position: relative;
        z-index: 10;
        padding-top: 5rem !important;
    }

    .title-box {
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(125, 191, 176, 0.35);
        border-top: 2px solid rgba(201, 180, 110, 0.6);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        margin-bottom: 1.5rem;
        color: #2d4a3e;
        text-align: center;
        box-shadow: 0 8px 32px rgba(109, 171, 156, 0.18), inset 0 1px 0 rgba(255,255,255,0.7);
        animation: fadeSlideDown 0.6s ease;
    }
    .title-box h1 {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
        color: #2d4a3e;
        text-shadow: 0 1px 4px rgba(109, 171, 156, 0.2);
    }
    .title-box p {
        font-size: 0.88rem;
        color: #5a7a6e;
        margin: 0.5rem 0 0;
        letter-spacing: 0.3px;
    }

    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .stTextArea textarea {
        background: rgba(255,255,255,0.55) !important;
        border: 1px solid rgba(125, 191, 176, 0.4) !important;
        border-radius: 12px !important;
        color: #2d4a3e !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        backdrop-filter: blur(10px);
    }
    .stTextArea textarea::placeholder { color: rgba(90,122,110,0.5) !important; }
    .stTextArea textarea:focus {
        border-color: rgba(106, 171, 156, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(106,171,156,0.15) !important;
    }
    label, .stTextArea label { color: #3d5e52 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #8aab8e, #6aab9c) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(106,171,156,0.35) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.2px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(106,171,156,0.5) !important;
        background: linear-gradient(135deg, #7dbd92, #5c9e8f) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    .stSuccess {
        background: rgba(168, 213, 194, 0.25) !important;
        border-left-color: #6aab9c !important;
        color: #2d4a3e !important;
        border-radius: 10px !important;
    }
    .stInfo {
        background: rgba(201, 180, 110, 0.15) !important;
        border-left-color: #c9b46e !important;
        color: #4a3d1e !important;
        border-radius: 10px !important;
    }
    .stWarning {
        background: rgba(245, 200, 100, 0.2) !important;
        border-left-color: #d4a84b !important;
        color: #4a3a10 !important;
        border-radius: 10px !important;
    }

    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.4) !important;
        color: #5a7a6e !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.3) !important;
        border-radius: 0 0 10px 10px !important;
    }

    h3 { color: #2d4a3e !important; }
    hr { border-color: rgba(125, 191, 176, 0.25) !important; }

    .result-box {
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.5rem;
        margin-bottom: 2rem; 
        font-weight: 600;
        font-size: 1rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        animation: fadeSlideUp 0.4s ease;
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .result-nonhate {
        background: rgba(168, 213, 194, 0.3);
        color: #1e4a38;
        border-left: 4px solid #6aab9c;
        box-shadow: 0 4px 16px rgba(106,171,156,0.15);
    }
    .result-hate {
        background: rgba(220, 150, 130, 0.2);
        color: #5a2010;
        border-left: 4px solid #c87060;
        box-shadow: 0 4px 16px rgba(200,112,96,0.15);
    }
    .result-sara {
        background: rgba(201, 180, 110, 0.25);
        color: #4a3010;
        border-left: 4px solid #c9b46e;
        box-shadow: 0 4px 16px rgba(201,180,110,0.18);
    }
    .result-umum {
        background: rgba(210, 140, 120, 0.2);
        color: #5a2018;
        border-left: 4px solid #d4806a;
        box-shadow: 0 4px 16px rgba(212,128,106,0.15);
    }

    .confidence-bar {
        background: rgba(125, 191, 176, 0.2);
        border-radius: 8px;
        height: 8px;
        margin-top: 0.6rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #6aab9c, #c9b46e);
        box-shadow: 0 0 8px rgba(106,171,156,0.4);
    }

    .step-badge {
        display: inline-block;
        background: rgba(45, 74, 62, 0.75);
        border: 1px solid rgba(201, 180, 110, 0.5);
        color: #e8d5a3;
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        margin-top: 0rem;        
        letter-spacing: 0.3px;
    }

    .info-small {
        font-size: 0.8rem;
        opacity: 0.72;
        margin-top: 0.3rem;
    }

    /* ── Notification box ── */
    .notif-box {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        animation: fadeSlideDown 0.35s ease;
        margin-bottom: 0.25rem;
    }
    .notif-ready {
        background: rgba(168, 213, 194, 0.3);
        border-left: 4px solid #6aab9c;
        color: #1e4a38;
    }
    .notif-empty {
        background: rgba(245, 200, 100, 0.2);
        border-left: 4px solid #d4a84b;
        color: #4a3a10;
    }
    .notif-analyzing {
        background: rgba(125, 191, 176, 0.2);
        border-left: 4px solid #7dbfb0;
        color: #1e3d34;
    }
    .notif-spinner {
        display: inline-block;
        width: 0.875rem;
        height: 0.875rem;
        border: 2px solid rgba(106,171,156,0.3);
        border-top-color: #6aab9c;
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
        flex-shrink: 0;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>

<!-- Floating orbs -->
<div class="bg-orbs">
    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="orb orb3"></div>
    <div class="orb orb4"></div>
    <div class="orb orb5"></div>
</div>

<!-- Gold shimmer particles -->
<div class="stars" id="stars"></div>

<script>
(function() {
    const container = document.getElementById('stars');
    if (!container) return;
    for (let i = 0; i < 60; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.left   = Math.random() * 100 + '%';
        star.style.top    = Math.random() * 100 + '%';
        star.style.animationDuration  = (3 + Math.random() * 5) + 's';
        star.style.animationDelay     = (Math.random() * 6) + 's';
        star.style.width  = star.style.height = (1.5 + Math.random() * 2) + 'px';
        container.appendChild(star);
    }
})();
</script>
""", unsafe_allow_html=True)

# ── Session state init ───────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "input"
if "result" not in st.session_state:
    st.session_state.result = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "notif_state" not in st.session_state:
    st.session_state.notif_state = "ready"

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

    # ── Notifikasi dinamis ──────────────────────────────────────
    notif = st.session_state.notif_state
    if notif == "ready":
        st.markdown("""
        <div class="notif-box notif-ready">
            ✅ &nbsp;Model berhasil dimuat! Silahkan masukkan komentar untuk dianalisis.
        </div>
        """, unsafe_allow_html=True)
    elif notif == "empty":
        st.markdown("""
        <div class="notif-box notif-empty">
            ⚠️ &nbsp;Teks tidak boleh kosong.
        </div>
        """, unsafe_allow_html=True)
    elif notif == "analyzing":
        st.markdown("""
        <div class="notif-box notif-analyzing">
            <span class="notif-spinner"></span>
            &nbsp;Menganalisis, silahkan tunggu...
        </div>
        """, unsafe_allow_html=True)

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
            st.session_state.notif_state = "empty"
            st.rerun()
        else:
            st.session_state.user_input = user_input
            st.session_state.notif_state = "analyzing"
            st.rerun()

    # Jalankan analisis jika state = analyzing
    if st.session_state.notif_state == "analyzing" and st.session_state.user_input.strip():
        with st.spinner(""):
            pred1, conf1 = predict_tahap1(st.session_state.user_input, tokenizer, model_t1)
            label2, conf2 = None, None
            if pred1 == 1:
                label2, conf2 = predict_tahap2(st.session_state.user_input, svm, tfidf, le)

        st.session_state.result = {
            "pred1": pred1,
            "conf1": conf1,
            "label2": label2,
            "conf2": conf2,
            "cleaned": clean_text(st.session_state.user_input)
        }
        st.session_state.notif_state = "ready"
        st.session_state.page = "result"
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8aaa96; font-size:0.8rem;'>"
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
        "<div style='text-align:center; color:#8aaa96; font-size:0.8rem;'>"
        "Sistem Deteksi Ujaran Kebencian SARA · Skripsi Sistem Informasi"
        "</div>",
        unsafe_allow_html=True
    )
