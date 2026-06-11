"""
Script pembangunan FAISS vector index untuk Smart-Kos Backend.
Membaca kolom 'deskripsi_clean' dari CSV processed, lalu membangun
FAISS IndexFlatIP menggunakan TF-IDF vectorizer (Tahap 1 placeholder).

Prasyarat: python scripts/preprocess_dataset.py sudah dijalankan sukses.
Output   : faiss_index/kos_tfidf.index + id_map.pkl + tfidf_vectorizer.pkl
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from stki.indexer import KosVectorIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_faiss_index(
    processed_csv_path: str,
    index_path: str,
    id_map_path: str,
    vectorizer_path: str,
) -> None:
    """
    Memuat dataset processed dan membangun FAISS vector index.

    Args:
        processed_csv_path : Path ke CSV processed (wajib ada kolom deskripsi_clean).
        index_path         : Path output file FAISS index (.index).
        id_map_path        : Path output file mapping posisi FAISS → id_kos (.pkl).
        vectorizer_path    : Path output TF-IDF vectorizer (.pkl).

    Raises:
        FileNotFoundError: Jika CSV processed tidak ditemukan.
        ValueError       : Jika kolom 'deskripsi_clean' tidak ada di CSV.
    """
    if not Path(processed_csv_path).exists():
        raise FileNotFoundError(
            f"CSV processed tidak ditemukan: {processed_csv_path}\n"
            "Solusi: Jalankan python scripts/preprocess_dataset.py terlebih dahulu."
        )

    logger.info(f"Membaca CSV processed: {processed_csv_path}")
    df: pd.DataFrame = pd.read_csv(processed_csv_path, encoding="utf-8")

    if "deskripsi_clean" not in df.columns:
        raise ValueError(
            "Kolom 'deskripsi_clean' tidak ditemukan. "
            "Pastikan preprocess_dataset.py sudah dijalankan."
        )

    # Filter baris dengan deskripsi_clean valid (tidak kosong setelah strip)
    df_valid = df[df["deskripsi_clean"].str.strip().str.len() > 0].copy()
    skipped = len(df) - len(df_valid)
    if skipped > 0:
        logger.warning(
            f"⚠  {skipped} baris dilewati karena deskripsi_clean kosong."
        )

    logger.info(f"Dokumen valid untuk diindeks: {len(df_valid)}/{len(df)}")

    documents: list[str] = df_valid["deskripsi_clean"].tolist()
    kos_ids: list[int] = df_valid["id_kos"].astype(int).tolist()

    # Bangun FAISS index menggunakan KosVectorIndexer
    indexer = KosVectorIndexer(
        index_path=index_path,
        id_map_path=id_map_path,
        vectorizer_path=vectorizer_path,
    )
    indexer.build_index(documents=documents, kos_ids=kos_ids)

    logger.info("✅ FAISS index berhasil dibangun!")
    logger.info(f"   Index      : {index_path}")
    logger.info(f"   ID Map     : {id_map_path}")
    logger.info(f"   Vectorizer : {vectorizer_path}")
    logger.info("")
    logger.info("   ⚡ Catatan migrasi Tahap 2:")
    logger.info("   Index saat ini menggunakan TF-IDF (placeholder).")
    logger.info("   Pada Tahap 2, ganti _vectorize() di stki/indexer.py")
    logger.info("   dengan IndoBERT sentence encoder (768-dim, IndexFlatIP tetap).")


if __name__ == "__main__":
    build_faiss_index(
        processed_csv_path=settings.data_processed_path,
        index_path=settings.faiss_index_path,
        id_map_path=settings.faiss_id_map_path,
        vectorizer_path=settings.faiss_vectorizer_path,
    )