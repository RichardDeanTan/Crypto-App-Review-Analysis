"""Halaman Tentang Model: pipeline, hyperparameter, dan metrik evaluasi."""

import pandas as pd
import streamlit as st

from core import config
from views.common import DEVICE, get_resources

PIPELINE_DOT = f"""
digraph {{
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", color="#3E63DD",
        fontname="Helvetica", fontsize=11];
  edge [color="#8B8D98"];

  raw   [label="Review mentah"];
  clean [label="Preprocessing\\n(11 langkah + kamus)"];

  tok   [label="Tokenizer\\nIndoBERTweet"];
  sa    [label="IndoBERTweet\\n(fine-tuned)"];
  sent  [label="Sentimen\\nnegatif / netral / positif", fillcolor="#E6F6EB", color="#30A46C"];

  emb   [label="Embedding\\nIndoBERT (mean pooling)"];
  red   [label="{config.TM_INFO['reducer_label']}\\n{config.TM_INFO['embedding_dim']} → {config.TM_INFO['n_components']} dimensi"];
  km    [label="K-Means\\n{config.TM_INFO['n_clusters']} klaster"];
  topic [label="Topik +\\nkata kunci (c-TF-IDF)", fillcolor="#E6F6EB", color="#30A46C"];

  raw -> tok [label="teks asli", fontsize=9, fontcolor="#8B8D98"];
  tok -> sa -> sent;
  raw -> clean -> emb -> red -> km -> topic;
}}
"""


def _kv_table(data: dict) -> pd.DataFrame:
    return pd.DataFrame({"Parameter": list(data.keys()), "Nilai": list(data.values())})


def main() -> None:
    res = get_resources()

    st.title("Tentang Model")
    st.write(
        "Aplikasi ini menggabungkan dua model hasil penelitian: klasifikasi **sentimen** "
        "dan pemodelan **topik** untuk ulasan aplikasi kripto (Pintu, Indodax, Tokocrypto)."
    )

    st.subheader("Alur pemrosesan")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")

    st.markdown(
        f"- **Sentimen:** {config.SA_INFO['input']}. IndoBERTweet dilatih dari teks mentah, "
        "sehingga emoji, tanda baca, dan kata tidak baku tetap menjadi sinyal bagi model.\n"
        f"- **Topik:** {config.TM_INFO['input']}. Teks dibersihkan agar kata kunci c-TF-IDF "
        "tidak dipenuhi variasi ejaan dan simbol."
    )

    with st.expander("Langkah preprocessing (input model topik)", icon=":material/cleaning_services:"):
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(config.PREPROCESSING_STEPS, 1)))
        st.caption(
            "Langkah yang sama dengan pembuatan dataset_clean.csv, sehingga input model topik "
            "konsisten dengan data training."
        )

    st.divider()

    # ── Sentimen ──
    st.header("Analisis Sentimen — IndoBERTweet")
    st.caption(f"Base model: `{config.SA_INFO['base_model']}`")

    cols = st.columns(len(config.SA_INFO["metrics"]))
    for col, (name, (value, std)) in zip(cols, config.SA_INFO["metrics"].items()):
        col.metric(name, value, delta=std, delta_color="off")

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Hyperparameter**")
        st.dataframe(_kv_table(config.SA_INFO["hyperparameters"]), hide_index=True, width="stretch")
    with right:
        st.markdown("**Data**")
        st.dataframe(_kv_table(config.SA_INFO["data"]), hide_index=True, width="stretch")

    st.markdown("**Classification report — Test set (rata-rata 3 seed: 42, 123, 2026)**")
    st.dataframe(pd.DataFrame(config.SA_INFO["report"]), hide_index=True, width="stretch")
    st.caption(
        "Kelas *netral* paling sulit dikenali (F1 ≈ 0.68) karena jumlah datanya paling sedikit "
        "dan sering bercampur dengan opini positif/negatif. Karena itu aplikasi menandai prediksi "
        "dengan keyakinan rendah."
    )

    st.markdown("**Baseline vs Fine-tuned**")
    st.dataframe(pd.DataFrame(config.SA_INFO["comparison"]), hide_index=True, width="stretch")
    st.caption(config.SA_INFO["comparison_note"])

    st.divider()

    # ── Topik ──
    st.header("Topic Modelling — BERTopic")
    st.caption(config.TM_INFO["pipeline"])

    cols = st.columns(len(config.TM_INFO["metrics"]))
    for col, (name, value) in zip(cols, config.TM_INFO["metrics"].items()):
        col.metric(name, value)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Hyperparameter**")
        st.dataframe(_kv_table(config.TM_INFO["hyperparameters"]), hide_index=True, width="stretch")
    with right:
        st.markdown("**Data**")
        st.dataframe(_kv_table(config.TM_INFO["data"]), hide_index=True, width="stretch")
        st.caption("Hybrid Score = 0.5 × Coherence (Cv) + 0.5 × KL Score")

    st.markdown("**Pemilihan model final**")
    st.dataframe(pd.DataFrame(config.TM_INFO["selection"]), hide_index=True, width="stretch")
    st.caption(config.TM_INFO["selection_note"])

    st.markdown("**Daftar topik**")
    st.dataframe(res.topic.overview(n_words=10), hide_index=True, width="stretch")
    st.caption(
        "*Kata kunci representatif* dipilih manual dari 10 kata teratas c-TF-IDF agar kata umum "
        "(misalnya *malah*, *padahal*, *kok*) tidak mendominasi tampilan. Kolom terakhir adalah "
        "kata teratas asli dari model."
    )

    st.divider()
    st.caption(f"Perangkat inferensi: **{DEVICE.upper()}**")


main()
