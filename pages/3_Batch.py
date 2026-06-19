import streamlit as st
import pandas as pd
import joblib
import re
import sqlite3
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Batch Prediction - FakeShield AI", page_icon="📁", layout="centered")

INDOBERT_REPO = "Pall654321/fakeshield-indobert"
DB_PATH = "database/news.db"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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

def predict_tfidf_batch(texts):
    model, vectorizer = load_tfidf()
    cleaned = [clean_text(t) for t in texts]
    vecs = vectorizer.transform(cleaned)
    preds = model.predict(vecs)
    probas = model.predict_proba(vecs)
    labels = ["HOAX" if p == 1 else "NON-HOAX" for p in preds]
    confidences = [max(p) for p in probas]
    return labels, confidences

def predict_indobert_batch(texts):
    tokenizer, model = load_indobert()
    labels, confidences = [], []
    progress = st.progress(0, text="Memproses dengan IndoBERT...")
    for i, t in enumerate(texts):
        cleaned = clean_text(t)
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            proba = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        pred = torch.argmax(proba).item()
        labels.append("HOAX" if pred == 1 else "NON-HOAX")
        confidences.append(proba[pred].item())
        progress.progress((i + 1) / len(texts), text=f"Memproses {i+1}/{len(texts)}...")
    progress.empty()
    return labels, confidences

def save_batch_to_history(texts, labels, confidences, model_used):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for text, label, conf in zip(texts, labels, confidences):
        conn.execute(
            "INSERT INTO history (news_text, prediction, confidence, model_used, tanggal) VALUES (?, ?, ?, ?, ?)",
            (text, label, conf, model_used, now)
        )
    conn.commit()
    conn.close()

st.title("📁 Batch Prediction")
st.caption("Upload file CSV berisi banyak berita untuk dideteksi sekaligus")

st.markdown(
    "**Format CSV yang diperlukan:**\n"
    "- Harus memiliki kolom bernama `text` yang berisi judul + isi berita\n"
    "- Setiap baris dianggap satu berita terpisah"
)

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

model_choice = st.radio(
    "Pilih model deteksi",
    ["NLP — TF-IDF + Logistic Regression", "LLM — IndoBERT (Fine-tuned)"],
    horizontal=True
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        df = None

    if df is not None:
        if "text" not in df.columns:
            st.error("File CSV harus memiliki kolom bernama 'text'.")
        else:
            st.success(f"File berhasil dimuat: {len(df)} baris ditemukan.")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🔍 Jalankan Batch Prediction", type="primary", use_container_width=True):
                texts = df["text"].astype(str).tolist()

                with st.spinner("Memproses prediksi..."):
                    if "NLP" in model_choice:
                        labels, confidences = predict_tfidf_batch(texts)
                        model_used = "NLP (TF-IDF)"
                    else:
                        labels, confidences = predict_indobert_batch(texts)
                        model_used = "LLM (IndoBERT)"

                    save_batch_to_history(texts, labels, confidences, model_used)

                result_df = df.copy()
                result_df["prediction"] = labels
                result_df["confidence"] = [f"{c*100:.2f}%" for c in confidences]

                st.markdown("---")
                st.subheader("Hasil Prediksi")
                st.dataframe(result_df, use_container_width=True)

                hoax_total = labels.count("HOAX")
                nonhoax_total = labels.count("NON-HOAX")
                c1, c2 = st.columns(2)
                c1.metric("Terdeteksi HOAX", hoax_total)
                c2.metric("Terdeteksi NON-HOAX", nonhoax_total)

                csv_download = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Hasil (CSV)",
                    data=csv_download,
                    file_name="hasil_prediksi_fakeshield.csv",
                    mime="text/csv",
                    use_container_width=True
                )
