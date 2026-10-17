import sys
import streamlit as st
from core import config

st.set_page_config(
    page_title="Analisis Review Aplikasi Kripto",
    page_icon=":material/analytics:",
    layout="wide",
)

from views.common import DEVICE, load_resources


def _check_python_version() -> None:
    if sys.version_info[:2] != config.REQUIRED_PYTHON:
        req = ".".join(map(str, config.REQUIRED_PYTHON))
        st.error(
            f"Aplikasi berjalan di Python {sys.version.split()[0]}, padahal model topik "
            f"dibuat di Python {req} (Kaggle) dan hanya bisa di-load di versi yang sama.",
            icon=":material/error:",
        )
        st.info(f"Buat ulang venv dengan Python {req}. Lihat README bagian *Cara menjalankan*.")
        st.stop()


def _check_files() -> None:
    missing = config.missing_required_files()
    if not missing:
        return
    st.error("File berikut belum ditemukan:", icon=":material/error:")
    st.markdown("\n".join(f"- `{p.relative_to(config.BASE_DIR)}`" for p in missing))
    st.info("File ini harus ikut di repo. Lihat README.md bagian *Struktur folder*.")
    st.stop()


def _load_models():
    try:
        with st.spinner(
            "Memuat model... (pertama kali bisa beberapa menit karena mengunduh model ±1 GB)",
            show_time=True,
        ):
            return load_resources()
    except Exception as e:  # tampilkan error yang jelas, bukan traceback mentah
        st.error(f"Gagal memuat model: {type(e).__name__}: {e}", icon=":material/error:")
        st.info(
            "Penyebab umum:\n"
            "- Tidak ada koneksi internet saat model perlu diunduh dari HuggingFace.\n"
            f"- Repo HF salah/belum ada/private: `{config.HF_SA_REPO}` dan `{config.HF_TM_REPO}`.\n"
            "- Versi library berbeda dengan Kaggle, atau file model rusak/tidak lengkap."
        )
        with st.expander("Detail error"):
            st.exception(e)
        st.stop()


def _render_sidebar(res) -> None:
    with st.sidebar:
        st.subheader("Pengaturan", divider="gray")
        st.slider(
            "Ambang keyakinan sentimen",
            min_value=0.40, max_value=0.90, step=0.05,
            value=config.DEFAULT_CONFIDENCE_THRESHOLD,
            key="conf_threshold",
            format="%.2f",
            help="Prediksi dengan keyakinan di bawah nilai ini akan diberi tanda peringatan.",
        )
        st.toggle(
            "Tampilkan teks hasil preprocessing",
            value=True,
            key="show_clean",
        )

        st.subheader("Ringkasan model", divider="gray")
        sa_f1, sa_std = config.SA_INFO["metrics"]["Test F1-Macro"]
        st.markdown(
            "**Sentimen** — IndoBERTweet (fine-tuned)  \n"
            f"F1-Macro test: `{sa_f1}` {sa_std}"
        )
        st.markdown(
            f"**Topik** — {config.TM_INFO['pipeline']}  \n"
            f"{len(res.topic.topic_ids)} topik · Cv `{config.TM_INFO['metrics']['Coherence (Cv)']}`"
        )

        with st.expander("Kamus topik", icon=":material/menu_book:"):
            for tid in res.topic.topic_ids:
                kw = ", ".join(res.topic.keywords(tid, config.TOP_KEYWORDS))
                st.markdown(f"**T{tid} · {res.topic.name(tid)}**  \n:gray[{kw}]")

        st.caption(
            f"Perangkat: {DEVICE.upper()}  \n"
            f"Model sentimen: {res.sa_location.source} (`{res.sa_location.label}`)  \n"
            f"Model topik: {res.tm_location.source} (`{res.tm_location.label}`)"
        )


pages = st.navigation([
    st.Page("views/single.py", title="Analisis Tunggal", icon=":material/edit_note:", default=True),
    st.Page("views/batch.py", title="Analisis Batch", icon=":material/table_view:"),
    st.Page("views/about.py", title="Tentang Model", icon=":material/info:"),
])

_check_python_version()
_check_files()
resources = _load_models()
_render_sidebar(resources)
pages.run()
