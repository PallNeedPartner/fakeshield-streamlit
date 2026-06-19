import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard - FakeShield AI", page_icon="📊", layout="centered")

DB_PATH = "database/news.db"

def get_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history", conn)
    conn.close()
    return df

st.title("📊 Dashboard Analytics")
st.caption("Statistik penggunaan FakeShield AI")

df = get_history()

if df.empty:
    st.info("Belum ada data untuk ditampilkan. Coba deteksi berita di halaman utama dulu.")
else:
    hoax_count = (df['prediction'] == "HOAX").sum()
    nonhoax_count = (df['prediction'] == "NON-HOAX").sum()
    nlp_count = (df['model_used'] == "NLP (TF-IDF)").sum()
    llm_count = (df['model_used'] == "LLM (IndoBERT)").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deteksi", len(df))
    col2.metric("Hoax Terdeteksi", hoax_count)
    col3.metric("Non-Hoax Terdeteksi", nonhoax_count)
    col4.metric("Rata-rata Keyakinan", f"{df['confidence'].mean()*100:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribusi Prediksi")
        fig1 = px.pie(
            names=["HOAX", "NON-HOAX"],
            values=[hoax_count, nonhoax_count],
            color=["HOAX", "NON-HOAX"],
            color_discrete_map={"HOAX": "#DC3545", "NON-HOAX": "#198754"},
            hole=0.4
        )
        fig1.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Penggunaan Model")
        fig2 = px.bar(
            x=["NLP (TF-IDF)", "LLM (IndoBERT)"],
            y=[nlp_count, llm_count],
            color=["NLP (TF-IDF)", "LLM (IndoBERT)"],
            color_discrete_map={"NLP (TF-IDF)": "#0D6EFD", "LLM (IndoBERT)": "#6F42C1"},
            labels={"x": "Model", "y": "Jumlah Penggunaan"}
        )
        fig2.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Riwayat Confidence Score")
    df_sorted = df.sort_values("id")
    fig3 = px.line(
        df_sorted,
        x="id",
        y="confidence",
        color="model_used",
        markers=True,
        labels={"id": "Urutan Deteksi", "confidence": "Confidence Score"}
    )
    fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig3, use_container_width=True)
