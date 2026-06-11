"""
Script batch text preprocessing dataset kos.
Membaca CSV augmented, menjalankan pipeline 4-langkah Sastrawi,
dan menyimpan hasilnya ke kolom 'deskripsi_clean' di CSV processed.

Strategi penggabungan teks:
    deskripsi_promosi + ulasan_penghuni → _raw_combined → deskripsi_clean
    Menggabungkan dua sumber teks meningkatkan kekayaan konteks semantik
    untuk FAISS vector index (lebih banyak informasi per dokumen).
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from stki.preprocessor import IndonesianTextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_dataset(
    input_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Menjalankan pipeline preprocessing teks pada dataset augmented.

    Langkah eksekusi:
        1. Baca CSV augmented
        2. Gabungkan kolom deskripsi_promosi + ulasan_penghuni
        3. Jalankan IndonesianTextPreprocessor.preprocess_batch()
           (normalize → clean → stopword removal → stemming)
        4. Simpan hasil ke kolom 'deskripsi_clean'
        5. Export ke CSV processed

    Args:
        input_path : Path ke CSV augmented (output augment_dataset.py).
        output_path: Path output CSV processed.

    Returns:
        DataFrame dengan kolom 'deskripsi_clean' tambahan.

    Raises:
        FileNotFoundError: Jika CSV augmented tidak ditemukan.
        ValueError        : Jika kolom teks yang diperlukan tidak ada.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"CSV augmented tidak ditemukan: {input_path}\n"
            "Solusi: Jalankan python scripts/augment_dataset.py terlebih dahulu."
        )

    logger.info(f"Membaca CSV augmented: {input_path}")
    df: pd.DataFrame = pd.read_csv(input_path, encoding="utf-8")
    logger.info(f"Dataset dimuat: {len(df)} baris.")

    # Validasi kolom teks yang diperlukan
    for col in ("deskripsi_promosi", "ulasan_penghuni"):
        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ada. Pastikan augment_dataset.py sudah dijalankan."
            )

    # --- Gabungkan dua sumber teks (pisahkan dengan spasi) ---
    logger.info("Menggabungkan deskripsi_promosi + ulasan_penghuni...")
    df["_raw_combined"] = (
        df["deskripsi_promosi"].fillna("") + " " +
        df["ulasan_penghuni"].fillna("")
    )

    # --- Inisialisasi preprocessor (sekali saja — loading kamus Sastrawi mahal) ---
    preprocessor = IndonesianTextPreprocessor()

    # --- Jalankan pipeline Sastrawi pada seluruh dokumen ---
    logger.info(f"Menjalankan Sastrawi pipeline pada {len(df)} dokumen...")
    df["deskripsi_clean"] = preprocessor.preprocess_batch(
        df["_raw_combined"].tolist()
    )

    # Hapus kolom intermediate yang tidak perlu disimpan
    df.drop(columns=["_raw_combined"], inplace=True)

    # --- Validasi kualitas output ---
    empty_count = (df["deskripsi_clean"].str.strip().str.len() == 0).sum()
    if empty_count > 0:
        logger.warning(
            f"⚠  {empty_count} baris menghasilkan deskripsi_clean kosong. "
            "Periksa kualitas data pada baris tersebut."
        )

    # --- Simpan hasil ke CSV processed ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"✅ Preprocessing selesai. Output: {output_path}")
    logger.info("Contoh output (3 baris pertama):")
    sample = df[["id_kos", "deskripsi_clean"]].head(3)
    for _, r in sample.iterrows():
        logger.info(f"  id_kos={r['id_kos']} | clean='{r['deskripsi_clean'][:80]}...'")

    return df


if __name__ == "__main__":
    preprocess_dataset(
        input_path=settings.data_augmented_path,
        output_path=settings.data_processed_path,
    )