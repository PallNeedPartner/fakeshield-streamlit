import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="History - FakeShield AI", page_icon="📜", layout="centered")

DB_PATH = "database/news.db"

def get_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df

def delete_item(item_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()

def clear_all():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()

st.title("📜 Riwayat Deteksi")
st.caption("Semua hasil prediksi yang pernah dilakukan")

df = get_history()

if df.empty:
    st.info("Belum ada riwayat deteksi. Coba deteksi berita di halaman utama dulu.")
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Total riwayat: {len(df)}**")
    with col2:
        if st.button("🗑️ Hapus Semua", use_container_width=True):
            clear_all()
            st.rerun()

    st.markdown("---")

    for _, row in df.iterrows():
        with st.container(border=True):
            preview = row['news_text'][:120] + "..." if len(row['news_text']) > 120 else row['news_text']
            badge = "🔴 HOAX" if row['prediction'] == "HOAX" else "🟢 NON-HOAX"

            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{badge}** · {row['model_used']} · {row['confidence']*100:.1f}% · _{row['tanggal']}_")
                st.caption(preview)
            with c2:
                if st.button("🗑️", key=f"del_{row['id']}", help="Hapus item ini"):
                    delete_item(row['id'])
                    st.rerun()

            with st.expander("Lihat teks lengkap"):
                st.write(row['news_text'])
