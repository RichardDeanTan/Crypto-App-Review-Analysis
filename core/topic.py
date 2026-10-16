"""
Model topik: BERTopic (IndoBERT + UMAP + K-Means) dari `tm_model/bertopic_model.pkl`.

File pickle disimpan dengan `save_embedding_model=False`, jadi IndoBERT di-load terpisah
dari HuggingFace lalu diberikan ke BERTopic. Prediksi topik berjalan lewat
UMAP.transform → KMeans.predict, sama seperti pipeline saat training.
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


def build_indobert_embedder(model_name: str, device: str = "cpu") -> SentenceTransformer:
    """Bangun embedder persis seperti notebook: Transformer + mean pooling."""
    word_embedding = Transformer(model_name)
    pooling = Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[word_embedding, pooling], device=device)


def load_topic_labels(path: str | Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


class TopicModel:
    def __init__(
        self,
        model_path: str | Path,
        embedding_model_name: str,
        labels: dict[int, str],
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
        self.topic_keywords = {
            tid: [w for w, _ in (self.model.get_topic(tid) or [])] for tid in self.topic_ids
        }

    def name(self, topic_id: int) -> str:
        return self.labels.get(topic_id, f"Topik {topic_id}")

    def keywords(self, topic_id: int, n: int = 2) -> list[str]:
        return self.topic_keywords.get(topic_id, [])[:n]

    def predict(self, texts: list[str], n_keywords: int = 2) -> list[dict]:
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
        """Panggilan pertama UMAP memicu kompilasi numba (lama); jalankan sekali di awal."""
        self.predict(["aplikasi ini sangat bagus dan mudah digunakan"])

    def overview(self, n_keywords: int = 10) -> pd.DataFrame:
        """Tabel ringkas semua topik (untuk halaman Tentang Model)."""
        rows = [
            {
                "ID": tid,
                "Nama Topik": self.name(tid),
                "Jumlah Dokumen (training)": self.topic_sizes.get(tid, 0),
                "Kata Kunci": ", ".join(self.keywords(tid, n_keywords)),
            }
            for tid in self.topic_ids
        ]
        return pd.DataFrame(rows)
