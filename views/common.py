"""
Resource bersama untuk semua halaman: loader model (di-cache) dan pengaturan sidebar.
"""

from dataclasses import dataclass

import streamlit as st
import torch

from core import config
from core.model_source import ModelLocation, sentiment_model_location, topic_model_file
from core.preprocessing import load_slang_dict
from core.sentiment import SentimentModel
from core.topic import TopicModel, load_topic_labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Resources:
    slang_dict: dict
    sentiment: SentimentModel
    topic: TopicModel
    sa_location: ModelLocation
    tm_location: ModelLocation


@st.cache_resource(show_spinner=False)
def _load_slang_dict() -> dict:
    return load_slang_dict(config.KAMUS_PATH)


@st.cache_resource(show_spinner=False)
def _load_sentiment_model() -> tuple[SentimentModel, ModelLocation]:
    loc = sentiment_model_location()
    model = SentimentModel(loc.location, max_length=config.SA_MAX_LENGTH, device=DEVICE)
    return model, loc


@st.cache_resource(show_spinner=False)
def _load_topic_model() -> tuple[TopicModel, ModelLocation]:
    loc = topic_model_file()
    model = TopicModel(
        loc.location,
        config.EMBEDDING_MODEL_NAME,
        labels=load_topic_labels(config.TOPIC_LABELS_PATH),
        device=DEVICE,
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    model.warmup()
    return model, loc


def load_resources() -> Resources:
    """Muat semua model sekali (di-cache oleh Streamlit selama server hidup)."""
    sentiment, sa_loc = _load_sentiment_model()
    topic, tm_loc = _load_topic_model()
    return Resources(
        slang_dict=_load_slang_dict(),
        sentiment=sentiment,
        topic=topic,
        sa_location=sa_loc,
        tm_location=tm_loc,
    )


def get_resources() -> Resources:
    """Dipanggil dari halaman. Model sudah di-load di app.py, jadi ini instan."""
    return load_resources()


# ── Pengaturan dari sidebar ─────────────────────────────────────────────────

def confidence_threshold() -> float:
    return st.session_state.get("conf_threshold", config.DEFAULT_CONFIDENCE_THRESHOLD)


def show_cleaned_text() -> bool:
    return st.session_state.get("show_clean", True)
