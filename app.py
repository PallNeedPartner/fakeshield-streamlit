import streamlit as st
import joblib
import re
import sqlite3
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="FakeShield AI",
    page_icon="🛡️",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────
INDOBERT_REPO = "Pall654321/fakeshield-indobert"
DB_PATH = "database/news.db"

# ── Database setup ───────────────────────────────────────
def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT,
            prediction TEXT,
            confidence REAL,
            model_used TEXT,
            tanggal TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_history(text, prediction, confidence, model_used):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO history (news_text, prediction, confidence, model_used, tanggal) VALUES (?, ?, ?, ?, ?)",
        (text, prediction, confidence, model_used, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

init_db()

# ── Text cleaning ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Load models (cached so it only loads once) ────────────
@st.cache_resource
def load_tfidf():
    model = joblib.load("model/tfidf_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def load_indobert():
    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(INDOBERT_REPO)
    model.eval()
    return tokenizer, model

# ── Prediction functions ───────────────────────────────────
def predict_tfidf(text):
    model, vectorizer = load_tfidf()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba)
    label = "HOAX" if pred == 1 else "NON-HOAX"
    return label, confidence

def predict_indobert(text):
    tokenizer, model = load_indobert()
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        proba = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pred = torch.argmax(proba).item()
    confidence = proba[pred].item()
    label = "HOAX" if pred == 1 else "NON-HOAX"
    return label, confidence

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("🛡️ FakeShield AI")
st.sidebar.markdown("Sistem deteksi berita hoax berbahasa Indonesia")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model yang digunakan:**\n"
    "- NLP: TF-IDF + Logistic Regression\n"
    "- LLM: IndoBERT (fine-tuned)"
)
st.sidebar.markdown("---")
st.sidebar.caption("Gunakan menu di atas untuk melihat History dan Dashboard.")

# ── Main page ─────────────────────────────────────────────
st.title("🛡️ FakeShield AI")
st.caption("Deteksi berita hoax berbahasa Indonesia menggunakan pendekatan NLP dan LLM")

st.markdown("### Masukkan teks berita")
news_text = st.text_area(
    "Tempel judul dan isi berita di sini",
    height=200,
    placeholder="Contoh: Pemerintah resmi mengumumkan kebijakan baru terkait subsidi BBM mulai bulan depan..."
)

model_choice = st.radio(
    "Pilih model deteksi",
    ["NLP — TF-IDF + Logistic Regression", "LLM — IndoBERT (Fine-tuned)"],
    horizontal=True
)

predict_clicked = st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True)

if predict_clicked:
    if not news_text.strip():
        st.warning("Masukkan teks berita terlebih dahulu.")
    else:
        with st.spinner("Menganalisis teks..."):
            if "NLP" in model_choice:
                label, confidence = predict_tfidf(news_text)
                model_used = "NLP (TF-IDF)"
            else:
                label, confidence = predict_indobert(news_text)
                model_used = "LLM (IndoBERT)"

            save_history(news_text, label, confidence, model_used)

        st.markdown("---")
        if label == "HOAX":
            st.error(f"### ⚠️ Terindikasi HOAX")
        else:
            st.success(f"### ✅ Terindikasi NON-HOAX (Berita Valid)")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tingkat Keyakinan", f"{confidence*100:.2f}%")
        with col2:
            st.metric("Model Digunakan", model_used)

        st.progress(confidence)
