"""
Pipeline cleaning teks review (input untuk model TOPIK saja).

Disalin dari notebook `text_cleaning.ipynb` dengan urutan dan regex yang SAMA PERSIS,
supaya input user diproses sama seperti data `dataset_clean.csv` yang dipakai melatih BERTopic.

Catatan: model SENTIMEN (IndoBERTweet) dilatih dari `dataset_raw.csv`, jadi TIDAK memakai
cleaning ini. Lihat core/pipeline.py.
"""

import re
from pathlib import Path

import emoji
import pandas as pd


def load_slang_dict(path: str | Path) -> dict[str, str]:
    """Muat kamus kata tidak baku → kata baku (kolom: kata_tidak_baku, kata_baku).

    Sama dengan notebook: kalau satu kata muncul lebih dari sekali, entri TERAKHIR yang dipakai.
    """
    kamus = pd.read_csv(path, encoding="utf-8")
    kamus = kamus.dropna(subset=["kata_tidak_baku", "kata_baku"])
    return dict(zip(
        kamus["kata_tidak_baku"].str.lower().str.strip(),
        kamus["kata_baku"].str.lower().str.strip(),
    ))


# ── Langkah-langkah cleaning (identik dengan text_cleaning.ipynb) ───────────

def fix_encoding_artifacts(text: str) -> str:
    """'Â²' → '²', 'ðŸ˜' → emoji asli. Teks normal dibiarkan apa adanya."""
    if not isinstance(text, str):
        return text
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text


def remove_username_patterns(text: str) -> str:
    """'username:aple', 'USERNAME : aple', 'user name: aple'."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\buser\s*name\s*:\s*\S*", "", text, flags=re.IGNORECASE)


def remove_mentions_hashtags(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return text


def remove_emojis(text: str) -> str:
    """Emoji diganti SPASI supaya kata yang menempel tidak tergabung ('bangett🙏ok')."""
    if not isinstance(text, str):
        return text
    return emoji.replace_emoji(text, replace=" ")


def normalize_squared_reduplication(text: str) -> str:
    """'lain²' → 'lain-lain'"""
    if not isinstance(text, str):
        return text
    return re.sub(r"(\w+)²", lambda m: f"{m.group(1)}-{m.group(1)}", text)


def remove_special_characters(text: str) -> str:
    """Karakter spesial diganti SPASI ('passkey/verifikasi' → 'passkey verifikasi').
    Dipertahankan: huruf, angka, spasi, tanda hubung (-)."""
    if not isinstance(text, str):
        return text
    return re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)


def normalize_digit_reduplication(text: str) -> str:
    """'lain2' → 'lain-lain' (angka standalone seperti '2022' tidak disentuh)."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\b([a-zA-Z]\w*)2\b", lambda m: f"{m.group(1)}-{m.group(1)}", text)


def to_lowercase(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.lower()


def normalize_slang(text: str, slang_dict: dict) -> str:
    """Satu kali lewat kamus per kata (tidak berantai), sama dengan notebook."""
    if not isinstance(text, str):
        return text
    return " ".join(slang_dict.get(w, w) for w in text.split())


def remove_standalone_dash(text: str) -> str:
    """Hapus strip yang TIDAK diapit huruf/angka di kedua sisi.
    'harga - ui' → 'harga   ui', '- aplikasi' → ' aplikasi'. Aman: 'hati-hati', 'user-friendly'."""
    if not isinstance(text, str):
        return text
    return re.sub(r"(?<![a-z0-9])-+|-+(?![a-z0-9])", " ", text)


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, slang_dict: dict) -> str:
    """Jalankan seluruh langkah cleaning. Mengembalikan '' untuk input non-teks."""
    if not isinstance(text, str):
        return ""
    text = fix_encoding_artifacts(text)
    text = remove_username_patterns(text)
    text = remove_mentions_hashtags(text)
    text = remove_emojis(text)
    text = normalize_squared_reduplication(text)
    text = remove_special_characters(text)
    text = normalize_digit_reduplication(text)
    text = to_lowercase(text)
    text = normalize_slang(text, slang_dict)
    text = remove_standalone_dash(text)
    text = normalize_whitespace(text)
    return text
