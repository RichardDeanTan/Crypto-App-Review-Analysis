"""
Pipeline lengkap. Kedua model menerima input yang BERBEDA, mengikuti data training masing-masing:

    review mentah ──┬── strip() ──────────────────→ IndoBERTweet → sentimen   (dilatih dari dataset_raw.csv)
                    └── clean_text() + kamus ─────→ BERTopic   → topik      (dilatih dari dataset_clean.csv)

Dipakai oleh Single Mode (1 teks) dan Batch Mode (banyak teks).
"""

from typing import Callable

import pandas as pd

from core import config
from core.preprocessing import clean_text

STATUS_OK = "OK"
STATUS_NO_TOPIC = "Topik tidak dapat ditentukan (teks kosong setelah preprocessing)"
STATUS_EMPTY = "Teks kosong"

RESULT_COLUMNS = [
    "review", "review_bersih", "sentimen", "confidence",
    "prob_negatif", "prob_netral", "prob_positif",
    "topik_id", "topik", "kata_kunci", "status",
]


def analyze_texts(
    texts: list[str],
    slang_dict: dict,
    sentiment_model,
    topic_model,
    progress: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    def report(fraction: float, message: str) -> None:
        if progress:
            progress(min(max(fraction, 0.0), 1.0), message)

    report(0.0, "Preprocessing teks...")
    raw = [t.strip() if isinstance(t, str) else "" for t in texts]
    cleaned = [clean_text(t, slang_dict) for t in raw]

    rows = []
    for r, c in zip(raw, cleaned):
        if not r:
            status = STATUS_EMPTY
        elif not c:
            status = STATUS_NO_TOPIC
        else:
            status = STATUS_OK
        rows.append({col: None for col in RESULT_COLUMNS} | {
            "review": r, "review_bersih": c, "status": status,
        })

    # ── Sentimen: teks asli (hanya di-strip), sama seperti training IndoBERTweet ──
    sa_idx = [i for i, r in enumerate(raw) if r]
    if sa_idx:
        report(0.1, "Memprediksi sentimen...")
        sa_results = sentiment_model.predict(
            [raw[i] for i in sa_idx],
            batch_size=config.SA_BATCH_SIZE,
            progress=lambda f: report(0.1 + 0.5 * f, "Memprediksi sentimen..."),
        )
        for i, sa in zip(sa_idx, sa_results):
            rows[i].update({
                "sentimen": sa["label"],
                "confidence": round(sa["confidence"], 4),
                "prob_negatif": round(sa["probabilities"].get("negatif", 0.0), 4),
                "prob_netral": round(sa["probabilities"].get("netral", 0.0), 4),
                "prob_positif": round(sa["probabilities"].get("positif", 0.0), 4),
            })

    # ── Topik: teks bersih, sama seperti training BERTopic ──
    tm_idx = [i for i, c in enumerate(cleaned) if c]
    if tm_idx:
        report(0.6, "Menentukan topik...")
        tm_results = topic_model.predict([cleaned[i] for i in tm_idx], n_keywords=config.TOP_KEYWORDS)
        for i, tm in zip(tm_idx, tm_results):
            rows[i].update({
                "topik_id": tm["topic_id"],
                "topik": tm["topic_name"],
                "kata_kunci": ", ".join(tm["keywords"]),
            })

    report(1.0, "Selesai")
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df["topik_id"] = df["topik_id"].astype("Int64")
    for col in ["confidence", "prob_negatif", "prob_netral", "prob_positif"]:
        df[col] = df[col].astype(float)
    return df
