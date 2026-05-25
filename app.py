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

    /* ── Animated background ── */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #1a1a4e, #0d2137, #1b3a5c, #0f2027);
        background-size: 400% 400%;
        animation: gradientShift 12s ease infinite;
        min-height: 100vh;
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Floating orbs ── */
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
        filter: blur(80px);
        opacity: 0.18;
        animation: floatOrb linear infinite;
    }
    .orb1 { width: 400px; height: 400px; background: #3b82f6; top: -100px; left: -100px; animation-duration: 20s; }
    .orb2 { width: 300px; height: 300px; background: #06b6d4; top: 40%; right: -80px; animation-duration: 25s; animation-delay: -8s; }
    .orb3 { width: 250px; height: 250px; background: #6366f1; bottom: -60px; left: 30%; animation-duration: 18s; animation-delay: -4s; }
    .orb4 { width: 200px; height: 200px; background: #0ea5e9; top: 60%; left: 10%; animation-duration: 22s; animation-delay: -12s; }

    @keyframes floatOrb {
        0%   { transform: translateY(0px) translateX(0px) scale(1); }
        33%  { transform: translateY(-40px) translateX(30px) scale(1.05); }
        66%  { transform: translateY(20px) translateX(-20px) scale(0.95); }
        100% { transform: translateY(0px) translateX(0px) scale(1); }
    }

    /* ── Stars / particles ── */
    .stars {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    .star {
        position: absolute;
        width: 2px; height: 2px;
        background: white;
        border-radius: 50%;
        animation: twinkle ease-in-out infinite;
        opacity: 0;
    }
    @keyframes twinkle {
        0%, 100% { opacity: 0; transform: scale(1); }
        50%       { opacity: 0.7; transform: scale(1.5); }
    }

    /* ── Glass card overlay ── */
    .block-container {
        position: relative;
        z-index: 10;
        padding-top: 2rem !important;
    }

    /* ── Title box (glassmorphism) ── */
    .title-box {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        animation: fadeSlideDown 0.6s ease;
    }
    .title-box h1 {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
        text-shadow: 0 2px 12px rgba(59, 130, 246, 0.5);
    }
    .title-box p {
        font-size: 0.88rem;
        opacity: 0.75;
        margin: 0.5rem 0 0;
        letter-spacing: 0.3px;
    }

    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Streamlit elements on dark bg ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        backdrop-filter: blur(10px);
    }
    .stTextArea textarea::placeholder { color: rgba(255,255,255,0.35) !important; }
    .stTextArea textarea:focus {
        border-color: rgba(59,130,246,0.6) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }
    label, .stTextArea label { color: #cbd5e1 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.2px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(59,130,246,0.55) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(8px) !important;
        color: #e2e8f0 !important;
    }
    .stSuccess { border-left-color: #10b981 !important; }
    .stInfo    { border-left-color: #3b82f6 !important; }
    .stWarning { border-left-color: #f59e0b !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* Markdown headings */
    h3 { color: #e2e8f0 !important; }
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* ── Result boxes (glassmorphism) ── */
    .result-box {
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
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
    .result-nonhate { background: rgba(16,185,129,0.15); color: #6ee7b7; border-left: 4px solid #10b981; }
    .result-hate    { background: rgba(239,68,68,0.15);  color: #fca5a5; border-left: 4px solid #ef4444; }
    .result-sara    { background: rgba(245,158,11,0.15); color: #fcd34d; border-left: 4px solid #f59e0b; }
    .result-umum    { background: rgba(244,63,94,0.15);  color: #fda4af; border-left: 4px solid #f43f5e; }

    .confidence-bar {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        height: 8px;
        margin-top: 0.6rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        box-shadow: 0 0 8px rgba(96,165,250,0.6);
    }

    .step-badge {
        display: inline-block;
        background: rgba(30,58,95,0.8);
        border: 1px solid rgba(59,130,246,0.4);
        color: #93c5fd;
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 0.3px;
    }

    .info-small {
        font-size: 0.8rem;
        opacity: 0.75;
        margin-top: 0.3rem;
    }
</style>

<!-- Floating orbs -->
<div class="bg-orbs">
    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="orb orb3"></div>
    <div class="orb orb4"></div>
</div>

<!-- Twinkling stars -->
<div class="stars" id="stars"></div>

<script>
(function() {
    const container = document.getElementById('stars');
    if (!container) return;
    for (let i = 0; i < 80; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.left   = Math.random() * 100 + '%';
        star.style.top    = Math.random() * 100 + '%';
        star.style.animationDuration  = (2 + Math.random() * 4) + 's';
        star.style.animationDelay     = (Math.random() * 5) + 's';
        star.style.width  = star.style.height = (1 + Math.random() * 2) + 'px';
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
        "<div style='text-align:center; color:rgba(148,163,184,0.6); font-size:0.8rem;'>"
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
        "<div style='text-align:center; color:rgba(148,163,184,0.6); font-size:0.8rem;'>"
        "Sistem Deteksi Ujaran Kebencian SARA · Skripsi Sistem Informasi"
        "</div>",
        unsafe_allow_html=True
    )
