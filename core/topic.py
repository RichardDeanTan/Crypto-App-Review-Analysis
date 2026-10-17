"""
Model topik: BERTopic (IndoBERT + KernelPCA + K-Means) dari `tm_model/bertopic_model.pkl`.

File pickle disimpan dengan `save_embedding_model=False`, jadi IndoBERT di-load terpisah
dari HuggingFace lalu diberikan ke BERTopic. Prediksi topik berjalan lewat
KernelPCA.transform → KMeans.predict, sama seperti pipeline saat training.
(Di dalam objek BERTopic, KernelPCA tersimpan di atribut `umap_model` dan K-Means di
`hdbscan_model` — itu hanya nama argumen bawaan BERTopic.)
"""

import json
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

try:  # path baru (sentence-transformers >= 5.x)
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
except ImportError:  # fallback versi lama
    from sentence_transformers.models import Pooling, Transformer


def _load_transformer(model_name: str) -> Transformer:
    """
    Repo indobenchmark/indobert-base-p1 hanya punya `pytorch_model.bin` di branch utama, sehingga
    transformers mengunduh .bin LALU mengunduh lagi `model.safetensors` hasil konversi otomatis
    di background (total ±1 GB). Dengan use_safetensors=True hanya versi safetensors (±500 MB)
    yang diunduh. Bobotnya identik. Kalau gagal (misal folder lokal yang hanya berisi .bin),
    kembali ke cara default.
    """
    try:
        return Transformer(model_name, model_kwargs={"use_safetensors": True})
    except Exception:
        return Transformer(model_name)


def build_indobert_embedder(model_name: str, device: str = "cpu") -> SentenceTransformer:
    """Bangun embedder persis seperti notebook: Transformer + mean pooling."""
    word_embedding = _load_transformer(model_name)
    pooling = Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[word_embedding, pooling], device=device)


def load_topic_labels(path: str | Path) -> dict[int, dict]:
    """
    Baca `topic_labels.json`. Format per topik:
        {"name": "Nama Topik", "keywords": ["kata", "kunci", "representatif"]}
    Format lama (hanya string nama) tetap didukung.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    labels = {}
    for key, value in raw.items():
        if isinstance(value, str):
            value = {"name": value}
        labels[int(key)] = {"name": value.get("name"), "keywords": list(value.get("keywords") or [])}
    return labels


class TopicModel:
    def __init__(
        self,
        model_path: str | Path,
        embedding_model_name: str,
        labels: dict[int, dict],
        device: str = "cpu",
        embed_batch_size: int = 32,
    ):
        self.embedder = build_indobert_embedder(embedding_model_name, device=device)
        self.embed_batch_size = embed_batch_size
        self.model = BERTopic.load(str(model_path), embedding_model=self.embedder)
        self.labels = labels

        info = self.model.get_topic_info()
        self.topic_ids = [int(t) for t in info["Topic"] if int(t) != -1]
        self.topic_sizes = {int(r.Topic): int(r.Count) for r in info.itertuples()}
        self.topic_words = {
            tid: [w for w, _ in (self.model.get_topic(tid) or []) if w] for tid in self.topic_ids
        }

    def name(self, topic_id: int) -> str:
        label = self.labels.get(topic_id, {})
        return label.get("name") or f"Topik {topic_id}"

    def model_words(self, topic_id: int, n: int = 10) -> list[str]:
        """Kata teratas c-TF-IDF langsung dari model (tanpa kurasi)."""
        return self.topic_words.get(topic_id, [])[:n]

    def keywords(self, topic_id: int, n: int = 3) -> list[str]:
        """
        Kata kunci yang ditampilkan di UI: kata kunci representatif dari topic_labels.json
        (dipilih manual dari 10 kata teratas c-TF-IDF). Kalau tidak ada, pakai kata teratas model.
        """
        curated = self.labels.get(topic_id, {}).get("keywords") or []
        return curated[:n] if curated else self.model_words(topic_id, n)

    def predict(self, texts: list[str], n_keywords: int = 3) -> list[dict]:
        """Prediksi topik untuk teks yang SUDAH di-cleaning."""
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.embed_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        topics, _ = self.model.transform(texts, embeddings=embeddings)
        return [
            {
                "topic_id": int(t),
                "topic_name": self.name(int(t)),
                "keywords": self.keywords(int(t), n_keywords),
            }
            for t in topics
        ]

    def warmup(self) -> None:
        """Satu prediksi awal supaya IndoBERT dan model siap sebelum request pertama user."""
        self.predict(["aplikasi ini sangat bagus dan mudah digunakan"])

    def overview(self, n_words: int = 10) -> pd.DataFrame:
        """Tabel ringkas semua topik (untuk halaman Tentang Model)."""
        rows = [
            {
                "ID": tid,
                "Nama Topik": self.name(tid),
                "Jumlah Dokumen (training)": self.topic_sizes.get(tid, 0),
                "Kata Kunci Representatif": ", ".join(self.keywords(tid, 5)),
                f"{n_words} Kata Teratas (c-TF-IDF)": ", ".join(self.model_words(tid, n_words)),
            }
            for tid in self.topic_ids
        ]
        return pd.DataFrame(rows)
