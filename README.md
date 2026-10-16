# Analisis Review Aplikasi Crypto — Streamlit App

Aplikasi untuk menjalankan dua model skripsi:

- **Analisis Sentimen** — IndoBERTweet hasil fine-tuning (negatif / netral / positif)
- **Topic Modelling** — BERTopic (IndoBERT + UMAP + K-Means, 8 topik)

Kedua model menerima input berbeda, mengikuti data training masing-masing:

```
review mentah ──┬── strip() ─────────────→ IndoBERTweet → sentimen   (dilatih dari dataset_raw.csv)
                └── cleaning + kamus ────→ BERTopic   → topik      (dilatih dari dataset_clean.csv)
```

Review yang kosong setelah cleaning (misal hanya emoji) tetap mendapat sentimen, tapi topiknya
"tidak dapat ditentukan".

Fitur: **Analisis Tunggal**, **Analisis Batch** (CSV, maks 100 baris, unduh hasil), dan **Tentang Model**.

---

## Struktur folder

```
app/
├── app.py                      # entry point (navigasi + sidebar + load model)
├── views/
│   ├── single.py               # halaman Analisis Tunggal
│   ├── batch.py                # halaman Analisis Batch
│   ├── about.py                # halaman Tentang Model
│   ├── common.py               # loader model (cache) + pengaturan sidebar
│   └── charts.py               # grafik Plotly
├── core/
│   ├── config.py               # path, parameter, metrik model
│   ├── preprocessing.py        # cleaning (sama dengan text_cleaning.ipynb), untuk model topik
│   ├── sentiment.py            # IndoBERTweet
│   ├── topic.py                # BERTopic
│   └── pipeline.py             # teks mentah → cleaning → sentimen + topik
├── assets/
│   ├── kamus_kata_baku.csv     # kamus terbaru (sudah termasuk di paket)
│   ├── topic_labels.json       # nama tiap topik (boleh diedit)
│   └── template_batch.csv
├── sa_model/                   # ⚠️ hasil extract model sentimen
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── tm_model/
│   └── bertopic_model.pkl      # ⚠️ hasil save pickle BERTopic
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

File bertanda ⚠️ tidak ikut di paket kode, jadi harus kamu taruh sendiri.

---

## Cara menjalankan (Windows, PowerShell)

### 0. Prasyarat

- **Python 3.13** (sama dengan Kaggle). Cek dengan:
  ```powershell
  py -3.13 --version
  ```
  Kalau belum ada, install dari https://www.python.org/downloads/ (centang *Add python.exe to PATH*).
- Koneksi internet (untuk install library dan unduh IndoBERT ±500MB saat pertama kali run).
- Ruang disk kosong ±4 GB.

### 1. Masuk ke folder app

```powershell
cd "C:\path\ke\folder\app"
```

### 2. Buat virtual environment

```powershell
py -3.13 -m venv venv
```

### 3. Aktifkan virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

Kalau muncul error *"running scripts is disabled on this system"*, jalankan sekali:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

lalu ulangi perintah aktivasi. Kalau berhasil, baris terminal diawali `(venv)`.

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install library

```powershell
pip install -r requirements.txt
```

Proses ini 5–15 menit (torch cukup besar). Di Windows, torch dari PyPI otomatis versi CPU.

### 6. Taruh file model & kamus

- `sa_model/` → isi hasil extract model sentimen
- `tm_model/bertopic_model.pkl` → file pickle BERTopic

### 7. Jalankan aplikasi

```powershell
streamlit run app.py
```

Browser akan terbuka otomatis di http://localhost:8501.
Run pertama butuh 1–3 menit karena mengunduh IndoBERT; run berikutnya jauh lebih cepat (sudah di-cache).

### 8. Menghentikan & menjalankan lagi

- Stop: tekan `Ctrl + C` di terminal.
- Run lagi lain waktu:
  ```powershell
  cd "C:\path\ke\folder\app"
  .\venv\Scripts\Activate.ps1
  streamlit run app.py
  ```

---

## Format CSV untuk Analisis Batch

```csv
review
"Aplikasinya lemot bgt, tiap buka harus nunggu lama"
"WD ke bank cepet banget, fee juga murah"
```

- Wajib ada kolom `review` (kolom lain diabaikan).
- Maksimal 100 baris (sisanya dipotong).
- Simpan sebagai **CSV UTF-8**.

Kolom hasil unduhan: `review` (input sentimen), `review_bersih` (input topik), `sentimen`, `confidence`, `prob_negatif`,
`prob_netral`, `prob_positif`, `topik_id`, `topik`, `kata_kunci`, `status`.

---

## Mengubah sesuatu

| Ingin mengubah | File |
|---|---|
| Nama topik | `assets/topic_labels.json` |
| Batas baris batch, ambang keyakinan default, jumlah kata kunci | `core/config.py` |
| Angka metrik di halaman Tentang Model | `core/config.py` (`SA_INFO`, `TM_INFO`) |
| Contoh review di Analisis Tunggal | `views/single.py` (`EXAMPLES`) |
| Warna tema | `.streamlit/config.toml` |

---

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `File berikut belum ditemukan` | Cek lokasi file sesuai *Struktur folder* di atas. |
| Gagal unduh IndoBERT / `OSError ... indobenchmark` | Pastikan internet aktif saat run pertama. |
| Error saat load `bertopic_model.pkl` | Pastikan versi library sama dengan `requirements.txt`, dan file disimpan dengan `save_embedding_model=False`. |
| `SystemError: no locals found when setting up annotations` | Venv dibuat dengan Python selain 3.13. Hapus folder `venv`, buat ulang dengan `py -3.13 -m venv venv`. |
| `py -3.13` tidak dikenal | Install Python 3.13, atau ganti dengan `python` kalau versi default kamu sudah 3.13. |
| Warning `huggingface_hub` / `HF_TOKEN` di terminal | Aman diabaikan. |
