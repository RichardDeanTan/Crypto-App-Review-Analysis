"""Halaman Analisis Batch: upload CSV (kolom `review`, maks 100 baris)."""

import io

import pandas as pd
import streamlit as st

from core import config
from core.pipeline import STATUS_OK, analyze_texts
from views import charts
from views.common import confidence_threshold, get_resources, show_cleaned_text


# ── Membaca & memvalidasi CSV ───────────────────────────────────────────────

def _read_csv(uploaded) -> pd.DataFrame | None:
    raw = uploaded.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except UnicodeDecodeError:
        st.error(
            "File tidak bisa dibaca sebagai UTF-8. Simpan ulang file sebagai "
            "**CSV UTF-8** (di Excel: *Save As → CSV UTF-8*).",
            icon=":material/error:",
        )
    except pd.errors.EmptyDataError:
        st.error("File CSV kosong.", icon=":material/error:")
    except pd.errors.ParserError as e:
        st.error(f"Format CSV tidak valid: {e}", icon=":material/error:")
    return None


def _validate(df: pd.DataFrame) -> list[str] | None:
    """Kembalikan daftar review siap proses, atau None kalau tidak valid."""
    columns = {c.strip().lower(): c for c in df.columns}
    if "review" not in columns:
        st.error(
            f"Kolom `review` tidak ditemukan. Kolom yang ada: {', '.join(map(str, df.columns))}. "
            "Unduh template di atas untuk contoh format.",
            icon=":material/error:",
        )
        return None

    extra = [c for c in df.columns if c != columns["review"]]
    if extra:
        st.info(f"Kolom lain diabaikan: {', '.join(map(str, extra))}", icon=":material/info:")

    series = df[columns["review"]]
    series = series[series.notna()].astype(str)
    series = series[series.str.strip() != ""]
    skipped = len(df) - len(series)
    if skipped:
        st.info(f"{skipped} baris kosong dilewati.", icon=":material/info:")

    if series.empty:
        st.error("Tidak ada review yang bisa diproses.", icon=":material/error:")
        return None

    if len(series) > config.BATCH_LIMIT:
        st.warning(
            f"File berisi {len(series)} review. Hanya **{config.BATCH_LIMIT} baris pertama** "
            "yang akan diproses.",
            icon=":material/warning:",
        )
        series = series.iloc[: config.BATCH_LIMIT]

    return series.tolist()


# ── Tampilan hasil ──────────────────────────────────────────────────────────

def _render_metrics(df: pd.DataFrame, with_sentiment: pd.DataFrame, with_topic: pd.DataFrame) -> None:
    threshold = confidence_threshold()
    low_conf = int((with_sentiment["confidence"] < threshold).sum())
    no_topic = len(with_sentiment) - len(with_topic)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Review diproses", len(df))
    c2.metric(
        "Topik terdeteksi", len(with_topic),
        delta=f"{no_topic} tanpa topik" if no_topic else None, delta_color="off",
    )
    c3.metric("Rata-rata keyakinan", f"{with_sentiment['confidence'].mean():.1%}")
    c4.metric(f"Keyakinan < {threshold:.0%}", low_conf)


def _insights(with_sentiment: pd.DataFrame, with_topic: pd.DataFrame) -> list[str]:
    notes = []
    dominant = with_sentiment["sentimen"].value_counts()
    notes.append(
        f"Sentimen dominan: **{dominant.index[0]}** "
        f"({dominant.iloc[0] / len(with_sentiment):.0%} dari {len(with_sentiment)} review)."
    )
    if with_topic.empty:
        return notes
    valid = with_topic
    top_topic = valid["topik"].value_counts()
    notes.append(
        f"Topik paling sering dibahas: **{top_topic.index[0]}** "
        f"({top_topic.iloc[0]} dari {len(valid)} review)."
    )

    per_topic = valid.groupby("topik")["sentimen"].agg(
        n="size",
        neg=lambda s: (s == "negatif").mean(),
        pos=lambda s: (s == "positif").mean(),
    )
    eligible = per_topic[per_topic["n"] >= 3]
    if not eligible.empty:
        worst = eligible["neg"].idxmax()
        if eligible.loc[worst, "neg"] > 0:
            notes.append(
                f"Topik dengan proporsi negatif tertinggi: **{worst}** "
                f"({eligible.loc[worst, 'neg']:.0%} dari {int(eligible.loc[worst, 'n'])} review)."
            )
        best = eligible["pos"].idxmax()
        if eligible.loc[best, "pos"] > 0:
            notes.append(
                f"Topik dengan proporsi positif tertinggi: **{best}** "
                f"({eligible.loc[best, 'pos']:.0%} dari {int(eligible.loc[best, 'n'])} review)."
            )
    return notes


def _render_visuals(with_sentiment: pd.DataFrame, with_topic: pd.DataFrame) -> None:
    st.markdown("\n".join(f"- {n}" for n in _insights(with_sentiment, with_topic)))

    plot_cfg = {"displayModeBar": False}
    left, right = st.columns(2, gap="medium")
    with left:
        st.subheader("Distribusi sentimen")
        st.plotly_chart(charts.sentiment_distribution_chart(with_sentiment), width="stretch", config=plot_cfg)

    if with_topic.empty:
        with right:
            st.info("Tidak ada review yang bisa ditentukan topiknya.", icon=":material/info:")
        return

    valid = with_topic
    with right:
        st.subheader("Distribusi topik")
        st.plotly_chart(charts.topic_distribution_chart(valid), width="stretch", config=plot_cfg)

    if len(with_sentiment) > len(with_topic):
        st.caption(
            f"Grafik topik dan Sentimen × Topik hanya memakai {len(with_topic)} review yang "
            "topiknya terdeteksi."
        )

    left, right = st.columns(2, gap="medium")
    with left:
        st.subheader("Sentimen × Topik (jumlah)")
        st.plotly_chart(charts.sentiment_topic_heatmap(valid), width="stretch", config=plot_cfg)
    with right:
        st.subheader("Komposisi sentimen per topik")
        st.plotly_chart(charts.sentiment_share_per_topic_chart(valid), width="stretch", config=plot_cfg)


def _render_table(df: pd.DataFrame) -> None:
    threshold = confidence_threshold()
    f1, f2, f3 = st.columns([1, 2, 1])
    sentiments = f1.multiselect("Filter sentimen", config.SENTIMENT_ORDER, placeholder="Semua")
    topics = f2.multiselect("Filter topik", sorted(df["topik"].dropna().unique()), placeholder="Semua")
    only_low = f3.toggle(f"Hanya keyakinan < {threshold:.0%}")

    view = df
    if sentiments:
        view = view[view["sentimen"].isin(sentiments)]
    if topics:
        view = view[view["topik"].isin(topics)]
    if only_low:
        view = view[view["confidence"] < threshold]

    columns = ["review"]
    if show_cleaned_text():
        columns.append("review_bersih")  # input model topik
    columns += ["sentimen", "confidence", "topik", "kata_kunci", "status"]

    st.caption(f"Menampilkan {len(view)} dari {len(df)} baris")
    st.dataframe(
        view[columns],
        width="stretch",
        hide_index=True,
        column_config={
            "review": st.column_config.TextColumn("Review", width="large"),
            "review_bersih": st.column_config.TextColumn(
                "Review bersih", width="large", help="Teks setelah preprocessing, dipakai model topik."
            ),
            "sentimen": st.column_config.TextColumn("Sentimen"),
            "confidence": st.column_config.ProgressColumn(
                "Keyakinan", min_value=0.0, max_value=1.0, format="percent"
            ),
            "topik": st.column_config.TextColumn("Topik", width="medium"),
            "kata_kunci": st.column_config.TextColumn(
                "Kata kunci", help="Kata kunci representatif topik (dipilih dari 10 kata teratas c-TF-IDF)."
            ),
            "status": st.column_config.TextColumn("Status"),
        },
    )


# ── Halaman ─────────────────────────────────────────────────────────────────

def main() -> None:
    res = get_resources()

    st.title("Analisis Batch")
    st.write(
        f"Upload file CSV dengan satu kolom bernama **`review`** (maksimal "
        f"**{config.BATCH_LIMIT} baris**). Hasil bisa diunduh sebagai CSV."
    )

    st.download_button(
        "Unduh template CSV",
        data=config.TEMPLATE_BATCH_PATH.read_bytes(),
        file_name="template_batch.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is None:
        st.session_state.pop("batch_result", None)
        return

    signature = (uploaded.name, uploaded.size)
    if st.session_state.get("batch_signature") != signature:
        st.session_state["batch_signature"] = signature
        st.session_state.pop("batch_result", None)

    df_input = _read_csv(uploaded)
    if df_input is None:
        return
    reviews = _validate(df_input)
    if reviews is None:
        return

    with st.expander(f"Pratinjau data ({len(reviews)} review)", icon=":material/table:"):
        st.dataframe(pd.DataFrame({"review": reviews}), width="stretch", hide_index=True)

    if st.button("Proses semua review", type="primary", icon=":material/play_arrow:"):
        bar = st.progress(0.0, text="Memulai...")
        df_result = analyze_texts(
            reviews, res.slang_dict, res.sentiment, res.topic,
            progress=lambda f, msg: bar.progress(f, text=msg),
        )
        bar.empty()
        st.session_state["batch_result"] = df_result

    df_result = st.session_state.get("batch_result")
    if df_result is None:
        return

    with_sentiment = df_result[df_result["sentimen"].notna()]
    with_topic = df_result[df_result["status"] == STATUS_OK]
    st.divider()

    if with_sentiment.empty:
        st.error("Tidak ada review yang bisa dianalisis.", icon=":material/error:")
        return

    _render_metrics(df_result, with_sentiment, with_topic)

    tab_visual, tab_table = st.tabs([":material/insights: Ringkasan visual", ":material/table_view: Tabel hasil"])
    with tab_visual:
        _render_visuals(with_sentiment, with_topic)
    with tab_table:
        _render_table(df_result)

    st.download_button(
        "Unduh hasil (CSV)",
        data=df_result.to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_analisis_review.csv",
        mime="text/csv",
        type="primary",
        icon=":material/download:",
    )


main()
