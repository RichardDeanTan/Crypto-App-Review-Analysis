"""
Konfigurasi global aplikasi.

Semua path, hyperparameter inferensi, dan ringkasan metrik model dikumpulkan di sini
supaya mudah diubah tanpa menyentuh kode lain.
"""

import os
from pathlib import Path

# Versi Python saat model dilatih di Kaggle. Pickle BERTopic hanya bisa di-load di versi ini.
REQUIRED_PYTHON = (3, 13)

# ── Path ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
KAMUS_PATH = ASSETS_DIR / "kamus_kata_baku.csv"
TOPIC_LABELS_PATH = ASSETS_DIR / "topic_labels.json"
TEMPLATE_BATCH_PATH = ASSETS_DIR / "template_batch.csv"

SA_MODEL_DIR = BASE_DIR / "sa_model"
TM_MODEL_PATH = BASE_DIR / "tm_model" / "bertopic_model.pkl"

# Sumber model sentimen: default folder lokal. Saat deploy, bisa diganti ke repo
# HuggingFace Hub (misal "username/indobertweet-sentimen") lewat environment variable.
SA_MODEL_SOURCE = os.getenv("SA_MODEL_SOURCE", str(SA_MODEL_DIR))

# Embedding model untuk BERTopic (tidak di-fine-tune, diunduh dari HuggingFace).
# Bisa diganti ke folder lokal lewat environment variable kalau ingin jalan offline.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "indobenchmark/indobert-base-p1")

# ── Inferensi ───────────────────────────────────────────────────────────────
SA_MAX_LENGTH = 64          # sama dengan TR_MAX_LENGTH saat training
SA_BATCH_SIZE = 16
EMBED_BATCH_SIZE = 32

BATCH_LIMIT = 10000           # maksimum baris untuk Batch Mode
SINGLE_MAX_CHARS = 1000     # maksimum karakter input Single Mode
TOP_KEYWORDS = 5            # jumlah kata kunci topik yang ditampilkan

DEFAULT_CONFIDENCE_THRESHOLD = 0.60

# ── Label & warna ───────────────────────────────────────────────────────────
SENTIMENT_ORDER = ["negatif", "netral", "positif"]
SENTIMENT_COLORS = {          # dipakai di grafik Plotly
    "negatif": "#E5484D",
    "netral": "#8B8D98",
    "positif": "#30A46C",
}
SENTIMENT_ST_COLORS = {       # nama warna markdown Streamlit
    "negatif": "red",
    "netral": "gray",
    "positif": "green",
}
SENTIMENT_ICONS = {
    "negatif": ":material/sentiment_dissatisfied:",
    "netral": ":material/sentiment_neutral:",
    "positif": ":material/sentiment_satisfied:",
}

# ── Ringkasan model (dari notebook Kaggle: SKRIPSI_Best_SA_Model & SKRIPSI_Best_TM_Model) ──
SA_INFO = {
    "base_model": "indolem/indobertweet-base-uncased",
    "input": "Teks mentah (dataset_raw.csv), tanpa cleaning",
    "hyperparameters": {
        "Batch size": "8",
        "Learning rate": "5e-5",
        "Epoch": "3 (dipilih dari 5-fold CV, maks. 4)",
        "Max length": "64 token",
        "Weight decay": "0.01",
    },
    "data": {
        "Total data": "4.127 review",
        "Tuning pool (train)": "3.507 review",
        "Test (held-out)": "620 review",
        "Split": "85% / 15%, stratified, seed 42",
        "Seed model final": "42, 123, 2026",
    },
    "metrics": {
        "CV F1-Macro (5-fold)": ("0.8397", "± 0.0083"),
        "Test F1-Macro": ("0.8616", "± 0.0061"),
        "Test Accuracy": ("0.9172", "± 0.0059"),
        "Train F1-Macro": ("0.9915", "± 0.0013"),
    },
    # Classification report Test, rata-rata 3 seed (42, 123, 2026)
    "report": [
        {"Kelas": "negatif", "Precision": "0.9306 ± 0.0027", "Recall": "0.9477 ± 0.0166", "F1-Score": "0.9390 ± 0.0077", "Support": 344},
        {"Kelas": "netral", "Precision": "0.7130 ± 0.0458", "Recall": "0.6453 ± 0.0160", "F1-Score": "0.6762 ± 0.0148", "Support": 78},
        {"Kelas": "positif", "Precision": "0.9683 ± 0.0125", "Recall": "0.9714 ± 0.0086", "F1-Score": "0.9698 ± 0.0053", "Support": 198},
        {"Kelas": "accuracy", "Precision": "", "Recall": "", "F1-Score": "0.9172 ± 0.0059", "Support": 620},
        {"Kelas": "macro avg", "Precision": "0.8706 ± 0.0136", "Recall": "0.8548 ± 0.0020", "F1-Score": "0.8616 ± 0.0061", "Support": 620},
        {"Kelas": "weighted avg", "Precision": "0.9153 ± 0.0044", "Recall": "0.9172 ± 0.0059", "F1-Score": "0.9157 ± 0.0048", "Support": 620},
    ],
    # Perbandingan dengan Baseline v3 (protokol sama, multi-seed)
    "comparison": [
        {"Model": "Baseline", "Batch size": 16, "Learning rate": "2e-5", "Epoch": 4,
         "CV F1-Macro": "0.8359 ± 0.0120", "Test F1-Macro": "0.8637 ± 0.0100"},
        {"Model": "Fine-tuned (dipakai di aplikasi)", "Batch size": 8, "Learning rate": "5e-5", "Epoch": 3,
         "CV F1-Macro": "0.8397 ± 0.0083", "Test F1-Macro": "0.8616 ± 0.0061"},
    ],
    "comparison_note": (
        "Selisih Test F1-Macro −0.0021 (−0.24%), lebih kecil dari std terbesar (0.0100), "
        "sehingga keduanya dinilai SERI. Konfigurasi fine-tuned dipilih karena CV F1-Macro-nya "
        "lebih tinggi dan variansinya lebih kecil; pemilihan model tidak boleh memakai skor Test."
    ),
}

TM_INFO = {
    "pipeline": "BERTopic (IndoBERT + UMAP + K-Means)",
    "input": "Teks bersih (dataset_clean.csv)",
    "hyperparameters": {
        "Embedding": "indobenchmark/indobert-base-p1 (mean pooling)",
        "UMAP n_neighbors": "30",
        "UMAP n_components": "10",
        "UMAP metric": "cosine",
        "K-Means n_clusters": "8",
        "K-Means n_init": "10",
        "c-TF-IDF vectorizer": "max_features=5000, min_df=2, max_df=0.95",
    },
    "data": {
        "Jumlah dokumen": "4.117 review (setelah cleaning & hapus duplikat)",
    },
    "metrics": {
        "Coherence (Cv)": "0.6660",
        "KL Divergence": "0.0431",
        "KL Score": "0.9793",
        "Hybrid Score": "0.8227",
    },
}

# Langkah text_cleaning.ipynb (hanya untuk input model topik)
PREPROCESSING_STEPS = [
    "Perbaiki artefak encoding (misal `Â²` → `²`)",
    "Hapus pola username (`username: xxx`)",
    "Hapus @mention dan #hashtag",
    "Ganti emoji dengan spasi",
    "Normalisasi reduplikasi `²` (`lain²` → `lain-lain`)",
    "Ganti karakter spesial dengan spasi (sisakan huruf, angka, spasi, `-`)",
    "Normalisasi reduplikasi angka (`lain2` → `lain-lain`)",
    "Ubah ke huruf kecil",
    "Normalisasi kata tidak baku (kamus kata baku)",
    "Hapus tanda `-` yang berdiri sendiri (`hati-hati` tetap aman)",
    "Rapikan spasi",
]


def missing_required_files() -> list[Path]:
    """Daftar file wajib yang belum ada (untuk pesan error yang jelas saat start)."""
    required = [KAMUS_PATH, TOPIC_LABELS_PATH, TM_MODEL_PATH]
    # Model sentimen hanya dicek kalau sumbernya folder lokal.
    if SA_MODEL_SOURCE == str(SA_MODEL_DIR):
        required += [SA_MODEL_DIR / "config.json", SA_MODEL_DIR / "model.safetensors"]
    return [p for p in required if not p.exists()]
