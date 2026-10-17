"""
Menentukan dari mana file model diambil: folder lokal atau HuggingFace Hub.

- Di laptop (folder sa_model/ dan tm_model/ ada) → pakai file lokal, tidak perlu internet
  untuk kedua model ini.
- Di Streamlit Cloud (folder model tidak ikut di-push ke GitHub) → unduh dari repo HF Hub
  yang diatur di core/config.py (HF_SA_REPO, HF_TM_REPO).

Repo HF yang public tidak butuh token. Kalau repo dibuat private, isi HF_TOKEN di
Streamlit secrets; huggingface_hub otomatis membacanya dari environment variable.
"""

from dataclasses import dataclass
from pathlib import Path

from core import config

LOCAL = "lokal"
HF_HUB = "HuggingFace Hub"


@dataclass
class ModelLocation:
    source: str      # LOCAL atau HF_HUB
    location: str    # path folder/file lokal, atau repo id HF
    label: str       # teks singkat untuk ditampilkan di UI


def _sa_local_available() -> bool:
    d = config.SA_MODEL_DIR
    return (d / "config.json").exists() and (d / "model.safetensors").exists()


def sentiment_model_location() -> ModelLocation:
    """Lokasi model IndoBERTweet. Hasilnya bisa langsung diberikan ke `from_pretrained()`
    (path folder lokal maupun repo id HF sama-sama diterima)."""
    if not config.FORCE_HF_HUB and _sa_local_available():
        return ModelLocation(LOCAL, str(config.SA_MODEL_DIR), "sa_model/")
    return ModelLocation(HF_HUB, config.HF_SA_REPO, config.HF_SA_REPO)


def topic_model_file() -> ModelLocation:
    """Path file bertopic_model.pkl. Kalau tidak ada di lokal, diunduh dulu dari HF Hub
    (disimpan di cache HuggingFace, jadi run berikutnya tidak mengunduh ulang)."""
    if not config.FORCE_HF_HUB and config.TM_MODEL_PATH.exists():
        return ModelLocation(LOCAL, str(config.TM_MODEL_PATH), "tm_model/bertopic_model.pkl")

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=config.HF_TM_REPO, filename=config.TM_MODEL_FILENAME)
    return ModelLocation(HF_HUB, str(Path(path)), f"{config.HF_TM_REPO}/{config.TM_MODEL_FILENAME}")
