"""
scripts/rebuild_faiss_indobert.py — Rebuild FAISS Index dengan IndoBERT
Smart-Kos Hybrid Engine — Tahap 2

Menggantikan kos_tfidf.index (Tahap 1) dengan kos_indobert.index (Tahap 2).

══════════════════════════════════════════════════════════════════
⚠  PERINGATAN PERFORMA:

  Encoding 909 dokumen via IndoBERT (BERT-base, 768-dim):
  ┌─────────────────────────────┬──────────────┬────────────────┐
  │ Environment                 │ batch_size   │ Estimasi Waktu │
  ├─────────────────────────────┼──────────────┼────────────────┤
  │ CPU lokal (Intel i5/i7)     │ 8 – 16       │ 30–60 menit    │
  │ GPU T4 Google Colab (FREE)  │ 64 – 128     │ 2–5 menit      │
  │ GPU A100 Kaggle (FREE)      │ 128 – 256    │ 1–2 menit      │
  └─────────────────────────────┴──────────────┴────────────────┘

  REKOMENDASI:
    Jalankan script ini di Google Colab T4, lalu download hasilnya:
      faiss_index/kos_indobert.index
      faiss_index/id_map_indobert.pkl
    ke folder lokal untuk digunakan oleh FastAPI server.

  INSTRUKSI COLAB (copy ke cell):
    !git clone https://github.com/USERNAME/smart-kos-backend.git
    %cd smart-kos-backend
    !pip install -q faiss-cpu transformers torch sentencepiece PySastrawi
    !python scripts/rebuild_faiss_indobert.py
    # Download kedua file di panel Files kiri sebelum session mati
══════════════════════════════════════════════════════════════════
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from stki.embedder import IndoBERTEmbedder
from stki.indexer import KosVectorIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def rebuild_faiss_indobert(
    processed_csv_path: str,
    index_path: str,
    id_map_path: str,
    model_name: str = "indobenchmark/indobert-base-p1",
    max_length: int = 128,
    batch_size: int | None = None,
) -> None:
    """
    Bangun FAISS IndoBERT index dari dataset processed CSV.

    Args:
        processed_csv_path : Path ke data_kos_processed.csv.
        index_path         : Path output FAISS index (.index).
        id_map_path        : Path output id_map (.pkl).
        model_name         : HuggingFace model identifier.
        max_length         : Panjang token maksimum (128 cukup untuk deskripsi kos).
        batch_size         : Batch size encoding. None = otomatis (GPU: 64, CPU: 8).

    Raises:
        FileNotFoundError: Jika CSV processed tidak ada.
        ValueError       : Jika kolom 'deskripsi_clean' tidak ditemukan.
    """
    if not Path(processed_csv_path).exists():
        raise FileNotFoundError(
            f"CSV processed tidak ditemukan: {processed_csv_path}\n"
            "Solusi: python scripts/preprocess_dataset.py"
        )

    logger.info(f"Membaca dataset: {processed_csv_path}")
    df: pd.DataFrame = pd.read_csv(processed_csv_path, encoding="utf-8")

    if "deskripsi_clean" not in df.columns:
        raise ValueError(
            "Kolom 'deskripsi_clean' tidak ada. "
            "Jalankan scripts/preprocess_dataset.py terlebih dahulu."
        )

    df_valid = df[df["deskripsi_clean"].str.strip().str.len() > 0].copy()
    logger.info(f"Dokumen valid: {len(df_valid)}/{len(df)}")

    documents: list[str] = df_valid["deskripsi_clean"].tolist()
    kos_ids: list[int] = df_valid["id_kos"].astype(int).tolist()

    # Tentukan batch_size optimal berdasarkan device
    is_gpu = torch.cuda.is_available()
    if batch_size is None:
        batch_size = 64 if is_gpu else 8
        logger.info(
            f"Auto batch_size: {batch_size} "
            f"({'GPU' if is_gpu else 'CPU'})"
        )

    if not is_gpu:
        logger.warning(
            "⚠  GPU tidak terdeteksi. Encoding akan berjalan di CPU.\n"
            "   Estimasi waktu: 30-60 menit untuk 909 dokumen.\n"
            "   Sangat disarankan menggunakan Google Colab T4 (gratis)."
        )

    # Inisialisasi IndoBERT embedder
    logger.info(f"Memuat IndoBERT: {model_name}...")
    embedder = IndoBERTEmbedder(
        model_name=model_name,
        device="auto",
        max_length=max_length,
    )

    # Bangun FAISS index
    indexer = KosVectorIndexer(
        index_path=index_path,
        id_map_path=id_map_path,
        embedder=embedder,
    )

    indexer.build_index(
        documents=documents,
        kos_ids=kos_ids,
        encode_batch_size=batch_size,
    )

    logger.info("✅ IndoBERT FAISS index berhasil dibangun!")
    logger.info(f"   Index   : {index_path}")
    logger.info(f"   ID Map  : {id_map_path}")
    logger.info(f"   Docs    : {len(df_valid)}")
    logger.info(f"   Dimensi : {IndoBERTEmbedder.EMBEDDING_DIM}")
    if not is_gpu:
        logger.info(
            "   Jika berjalan di Colab, download file index sekarang "
            "sebelum session Google Colab berakhir!"
        )


if __name__ == "__main__":
    rebuild_faiss_indobert(
        processed_csv_path=settings.data_processed_path,
        index_path=settings.faiss_indobert_index_path,
        id_map_path=settings.faiss_indobert_id_map_path,
        model_name=settings.indobert_model_name,
        max_length=settings.indobert_max_length,
        batch_size=settings.indobert_batch_size if settings.indobert_batch_size != 16
                   else None,  # None = auto-detect GPU/CPU
    )