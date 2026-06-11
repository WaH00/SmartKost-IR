"""
Script inisialisasi PostgreSQL + PostGIS untuk Smart-Kos Backend.

Proses berurutan:
    1. Buat koneksi sinkron ke PostgreSQL via psycopg2
    2. Eksekusi file migrasi SQL (DDL: tabel, trigger, index, view)
    3. Baca CSV processed, hitung derived features yang belum ada
    4. Batch insert semua data ke tabel kos_listings (idempoten via ON CONFLICT)

Catatan: Script ini menggunakan psycopg2 sinkron (bukan asyncpg) karena
hanya dijalankan sekali sebagai admin script, bukan sebagai FastAPI endpoint.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Label encoding — HARUS konsisten dengan label_encoders.pkl dari dataset asli
# Urutan alfabetis digunakan sebagai konvensi (sama dengan LabelEncoder sklearn)
# =============================================================================
KOTA_ENCODING: dict[str, int] = {
    "Jakarta Barat":   0,
    "Jakarta Pusat":   1,
    "Jakarta Selatan": 2,
    "Jakarta Timur":   3,
    "Jakarta Utara":   4,
}

TIPE_KOS_ENCODING: dict[str, int] = {
    "Campur": 0,
    "Putra":  1,
    "Putri":  2,
}


def get_connection() -> psycopg2.extensions.connection:
    """
    Membuat koneksi sinkron ke PostgreSQL menggunakan DATABASE_DSN dari config.
    Digunakan hanya untuk admin scripts — bukan untuk FastAPI runtime.
    """
    logger.info("Menghubungkan ke PostgreSQL...")
    conn = psycopg2.connect(settings.database_dsn)
    logger.info("Koneksi PostgreSQL berhasil.")
    return conn


def run_migration(
    conn: psycopg2.extensions.connection,
    migration_path: str,
) -> None:
    """
    Mengeksekusi file SQL migrasi ke database.

    Args:
        conn           : Koneksi psycopg2 aktif.
        migration_path : Path absolut ke file .sql migrasi.

    Raises:
        FileNotFoundError: Jika file .sql tidak ditemukan.
    """
    if not Path(migration_path).exists():
        raise FileNotFoundError(f"File migrasi tidak ditemukan: {migration_path}")

    logger.info(f"Menjalankan migrasi SQL: {Path(migration_path).name}")
    with open(migration_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    with conn.cursor() as cur:
        cur.execute(sql_script)
    conn.commit()
    logger.info("Migrasi SQL berhasil dieksekusi.")


def load_data_to_postgis(
    conn: psycopg2.extensions.connection,
    csv_path: str,
    batch_size: int = 100,
) -> None:
    """
    Memuat dataset processed CSV ke tabel kos_listings via batch insert.
    Insert bersifat idempoten: ON CONFLICT (id_kos) DO UPDATE memastikan
    script dapat dijalankan ulang tanpa duplikasi data.

    Args:
        conn       : Koneksi psycopg2 aktif.
        csv_path   : Path ke CSV processed (output preprocess_dataset.py).
        batch_size : Jumlah baris per satu batch insert (default: 100).

    Raises:
        FileNotFoundError: Jika CSV processed tidak ditemukan.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV processed tidak ditemukan: {csv_path}\n"
            "Solusi: Jalankan python scripts/preprocess_dataset.py terlebih dahulu."
        )

    logger.info(f"Membaca CSV processed: {csv_path}")
    df: pd.DataFrame = pd.read_csv(csv_path, encoding="utf-8")

    # Hitung derived features yang belum ada di CSV
    fasilitas_cols = [
        "ac", "kamar_mandi_dalam", "wifi", "listrik_include",
        "parkir", "dapur", "laundry", "security_24jam",
    ]
    df["total_fasilitas"] = df[fasilitas_cols].sum(axis=1).astype(int)
    df["jarak_kampus_dekat"] = (df["jarak_ke_kampus_km"] < 2.0).astype(int)
    df["jarak_transportasi_dekat"] = (df["jarak_ke_transportasi_km"] < 1.0).astype(int)
    df["kota_encoded"] = df["kota"].map(KOTA_ENCODING).fillna(0).astype(int)
    df["tipe_kos_encoded"] = df["tipe_kos"].map(TIPE_KOS_ENCODING).fillna(0).astype(int)

    # Template INSERT idempoten dengan ON CONFLICT DO UPDATE
    # Trigger trg_sync_geom di PostgreSQL akan otomatis mengisi kolom geom
    # dari latitude + longitude setiap kali baris di-insert atau di-update.
    insert_sql = """
        INSERT INTO kos_listings (
            id_kos, nama_kos, kota, wilayah, harga_per_bulan,
            tipe_kos, ukuran_kamar, jumlah_kamar,
            ac, kamar_mandi_dalam, wifi, listrik_include,
            parkir, dapur, laundry, security_24jam,
            total_fasilitas, jarak_ke_kampus_km, jarak_ke_transportasi_km,
            jarak_kampus_dekat, jarak_transportasi_dekat,
            rating, jumlah_review,
            deskripsi_promosi, ulasan_penghuni, deskripsi_clean, foto_path,
            latitude, longitude,
            kota_encoded, tipe_kos_encoded
        ) VALUES %s
        ON CONFLICT (id_kos) DO UPDATE SET
            deskripsi_clean   = EXCLUDED.deskripsi_clean,
            deskripsi_promosi = EXCLUDED.deskripsi_promosi,
            ulasan_penghuni   = EXCLUDED.ulasan_penghuni,
            latitude          = EXCLUDED.latitude,
            longitude         = EXCLUDED.longitude,
            updated_at        = NOW();
    """

    records: list[tuple[Any, ...]] = []
    total_inserted = 0
    total_rows = len(df)

    logger.info(f"Mulai batch insert {total_rows} baris (batch_size={batch_size})...")

    for _, row in df.iterrows():
        records.append((
            int(row["id_kos"]),
            str(row.get("nama_kos", f"Kos-{row['id_kos']}")),
            str(row["kota"]),
            str(row.get("wilayah", "")),
            int(row["harga_per_bulan"]),
            str(row.get("tipe_kos", "Campur")),
            int(row.get("ukuran_kamar", 9)),
            int(row.get("jumlah_kamar", 1)),
            int(row.get("ac", 0)),
            int(row.get("kamar_mandi_dalam", 0)),
            int(row.get("wifi", 0)),
            int(row.get("listrik_include", 0)),
            int(row.get("parkir", 0)),
            int(row.get("dapur", 0)),
            int(row.get("laundry", 0)),
            int(row.get("security_24jam", 0)),
            int(row["total_fasilitas"]),
            float(row.get("jarak_ke_kampus_km", 0.0)),
            float(row.get("jarak_ke_transportasi_km", 0.0)),
            int(row["jarak_kampus_dekat"]),
            int(row["jarak_transportasi_dekat"]),
            float(row.get("rating", 0.0)),
            int(row.get("jumlah_review", 0)),
            str(row.get("deskripsi_promosi", "")),
            str(row.get("ulasan_penghuni", "")),
            str(row.get("deskripsi_clean", "")),
            str(row.get("foto_path", "")),
            float(row.get("latitude", -6.2)),
            float(row.get("longitude", 106.8)),
            int(row.get("kota_encoded", 0)),
            int(row.get("tipe_kos_encoded", 0)),
        ))

        # Eksekusi batch insert setiap batch_size baris terkumpul
        if len(records) >= batch_size:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, insert_sql, records)
            conn.commit()
            total_inserted += len(records)
            logger.info(f"Progress: {total_inserted}/{total_rows} baris.")
            records = []

    # Insert sisa baris yang belum di-flush
    if records:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, insert_sql, records)
        conn.commit()
        total_inserted += len(records)

    logger.info(
        f"✅ {total_inserted} baris berhasil dimuat ke PostgreSQL + PostGIS."
    )
    logger.info(
        "   Kolom 'geom' diisi otomatis oleh trigger trg_sync_geom "
        "berdasarkan latitude + longitude."
    )


def main() -> None:
    """Entry point script init_postgis."""
    conn = None
    try:
        conn = get_connection()

        # Langkah 1: Jalankan DDL migration
        migration_path = str(
            Path(__file__).parent.parent / "migrations" / "001_create_kos_table.sql"
        )
        run_migration(conn, migration_path)

        # Langkah 2: Load data ke PostGIS
        load_data_to_postgis(
            conn=conn,
            csv_path=settings.data_processed_path,
            batch_size=100,
        )

        logger.info("🎉 Inisialisasi PostgreSQL + PostGIS selesai dengan sukses!")

    except Exception as exc:
        logger.error(f"❌ Terjadi kesalahan: {exc}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Koneksi PostgreSQL ditutup.")


if __name__ == "__main__":
    main()