"""Model sentimen: IndoBERTweet hasil fine-tuning (folder `sa_model/`)."""

from typing import Callable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentModel:
    def __init__(self, source: str, max_length: int = 64, device: str = "cpu"):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.to(device)
        self.model.eval()
        # config.json menyimpan key sebagai string ("0"), ubah ke int.
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.labels = [self.id2label[i] for i in sorted(self.id2label)]

    @torch.inference_mode()
    def predict(
        self,
        texts: list[str],
        batch_size: int = 16,
        progress: Callable[[float], None] | None = None,
    ) -> list[dict]:
        """
        Prediksi sentimen untuk teks yang SUDAH di-cleaning.

        Return per teks: {"label": str, "confidence": float, "probabilities": {label: float}}
        """
        results = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()

            for row in probs:
                idx = int(row.argmax())
                results.append({
                    "label": self.id2label[idx],
                    "confidence": float(row[idx]),
                    "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(row)},
                })
            if progress:
                progress(min(start + batch_size, total) / total)
        return results
