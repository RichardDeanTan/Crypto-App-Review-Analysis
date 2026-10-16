"""Halaman Analisis Tunggal: satu review → sentimen + topik."""

import streamlit as st

from core import config
from core.pipeline import analyze_texts
from views import charts
from views.common import confidence_threshold, get_resources, show_cleaned_text

EXAMPLES = {
    "Negatif": "Aplikasinya lemot bgt, tiap mau login selalu gagal terus 😡 tolong diperbaiki!!",
    "Positif": "WD ke bank cepet bgt, fee jg murah. Tampilannya simpel, gampang dipake pemula 👍",
    "Netral": "Baru coba download, msh belajar cara pakainya. Nanti diupdate lg reviewnya",
    "Harga": "Harga koin pas beli beda jauh sm yg di chart, market order malah kena harga atas",
}


def _set_example(text: str) -> None:
    st.session_state["single_input"] = text
    st.session_state.pop("single_result", None)


def _render_sentiment(row: dict) -> None:
    threshold = confidence_threshold()
    label = row["sentimen"]
    color = config.SENTIMENT_ST_COLORS[label]
    with st.container(border=True):
        st.caption("SENTIMEN")
        st.markdown(f"### :{color}[{label.capitalize()}]")
        st.metric("Tingkat keyakinan", f"{row['confidence']:.1%}")
        st.plotly_chart(
            charts.probability_chart({
                "negatif": row["prob_negatif"],
                "netral": row["prob_netral"],
                "positif": row["prob_positif"],
            }),
            width="stretch",
            config={"displayModeBar": False},
        )
        if row["confidence"] < threshold:
            st.warning(
                f"Keyakinan model di bawah ambang {threshold:.0%}. "
                "Hasil ini sebaiknya ditinjau manual.",
                icon=":material/warning:",
            )


def _render_topic(row: dict) -> None:
    with st.container(border=True):
        st.caption("TOPIK")
        if not isinstance(row["topik"], str):
            st.markdown("### :gray[Tidak dapat ditentukan]")
            st.info(
                "Setelah preprocessing teks menjadi kosong (misalnya hanya berisi emoji atau "
                "tanda baca), jadi topiknya tidak bisa ditentukan. Sentimen tetap dianalisis "
                "dari teks asli.",
                icon=":material/info:",
            )
            return
        st.markdown(f"### {row['topik']}")
        st.markdown(f"Topik **{row['topik_id']}**")
        st.markdown("**Kata kunci topik**")
        keywords = [k.strip() for k in (row["kata_kunci"] or "").split(",") if k.strip()]
        st.markdown(" ".join(f":blue-badge[{k}]" for k in keywords))
        st.caption(
            "Topik ditentukan oleh K-Means (setiap review masuk ke tepat satu klaster), "
            "sehingga tidak ada skor probabilitas topik."
        )


def _render_result(row: dict) -> None:
    col_sa, col_tm = st.columns(2, gap="medium")
    with col_sa:
        _render_sentiment(row)
    with col_tm:
        _render_topic(row)

    if show_cleaned_text():
        with st.expander("Lihat teks yang masuk ke masing-masing model", icon=":material/cleaning_services:"):
            st.markdown("**Input model sentimen (IndoBERTweet)** — teks asli, tanpa cleaning")
            st.text(row["review"])
            st.markdown("**Input model topik (BERTopic)** — teks setelah preprocessing")
            st.text(row["review_bersih"] or "(kosong)")
            st.caption(
                "Kedua model dilatih dengan data yang berbeda: IndoBERTweet dari teks mentah, "
                "BERTopic dari teks yang sudah dibersihkan. Karena itu inputnya juga dibedakan."
            )


def main() -> None:
    res = get_resources()

    st.title("Analisis Tunggal")
    st.write(
        "Masukkan satu ulasan aplikasi kripto. Sentimen dianalisis dari teks asli, "
        "sedangkan topik dianalisis dari teks yang sudah dibersihkan."
    )

    st.caption("Coba contoh:")
    example_cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(example_cols, EXAMPLES.items()):
        col.button(name, on_click=_set_example, args=(text,), width="stretch")

    with st.form("single_form", border=False):
        text = st.text_area(
            "Teks review",
            key="single_input",
            height=140,
            max_chars=config.SINGLE_MAX_CHARS,
            placeholder="Contoh: aplikasinya bagus, tapi withdraw ke bank lama banget...",
        )
        submitted = st.form_submit_button("Analisis", type="primary", icon=":material/search:")

    if submitted:
        if not text.strip():
            st.warning("Teks review masih kosong.")
            st.session_state.pop("single_result", None)
        else:
            with st.spinner("Menganalisis..."):
                df = analyze_texts([text], res.slang_dict, res.sentiment, res.topic)
            st.session_state["single_result"] = df.iloc[0].to_dict()

    result = st.session_state.get("single_result")
    if result is None:
        return

    st.divider()
    _render_result(result)


main()
