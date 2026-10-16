"""
Resource bersama untuk semua halaman: loader model (di-cache) dan pengaturan sidebar.
"""

from dataclasses import dataclass

import streamlit as st
import torch

from core import config
from core.preprocessing import load_slang_dict
from core.sentiment import SentimentModel
from core.topic import TopicModel, load_topic_labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Resources:
    slang_dict: dict
    sentiment: SentimentModel
    topic: TopicModel


@st.cache_resource(show_spinner=False)
def _load_slang_dict() -> dict:
    return load_slang_dict(config.KAMUS_PATH)


@st.cache_resource(show_spinner=False)
def _load_sentiment_model() -> SentimentModel:
    return SentimentModel(config.SA_MODEL_SOURCE, max_length=config.SA_MAX_LENGTH, device=DEVICE)


@st.cache_resource(show_spinner=False)
def _load_topic_model() -> TopicModel:
    model = TopicModel(
        config.TM_MODEL_PATH,
        config.EMBEDDING_MODEL_NAME,
        labels=load_topic_labels(config.TOPIC_LABELS_PATH),
        device=DEVICE,
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    model.warmup()
    return model


def load_resources() -> Resources:
    """Muat semua model sekali (di-cache oleh Streamlit selama server hidup)."""
    return Resources(
        slang_dict=_load_slang_dict(),
        sentiment=_load_sentiment_model(),
        topic=_load_topic_model(),
    )


def get_resources() -> Resources:
    """Dipanggil dari halaman. Model sudah di-load di app.py, jadi ini instan."""
    return load_resources()


# ── Pengaturan dari sidebar ─────────────────────────────────────────────────

def confidence_threshold() -> float:
    return st.session_state.get("conf_threshold", config.DEFAULT_CONFIDENCE_THRESHOLD)


def show_cleaned_text() -> bool:
    return st.session_state.get("show_clean", True)
