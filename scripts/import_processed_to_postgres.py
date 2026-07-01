import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import create_engine, text

from app.core.config import settings


def main():
    sync_url = settings.database_url.replace(
        "postgresql+asyncpg://",
        "postgresql+psycopg2://",
    )

    print(f"Membaca CSV: {settings.data_processed_path}")
    df = pd.read_csv(settings.data_processed_path)

    df["id_kos"] = (
        df["id_kos"]
        .astype(str)
        .str.replace("KOS-", "", regex=False)
        .astype(int)
    )

    fasilitas_cols = [
        "ac",
        "kamar_mandi_dalam",
        "wifi",
        "listrik_include",
        "parkir",
        "dapur",
        "laundry",
        "security_24jam",
    ]

    for col in fasilitas_cols:
        df[col] = df[col].fillna(0).astype(int)

    df["total_fasilitas"] = df[fasilitas_cols].sum(axis=1)
    df["jarak_kampus_dekat"] = (df["jarak_ke_kampus_km"] <= 2).astype(int)
    df["jarak_transportasi_dekat"] = (
        df["jarak_ke_transportasi_km"] <= 1
    ).astype(int)

    df["kota_encoded"] = df["kota"].astype("category").cat.codes.astype(int)
    df["tipe_kos_encoded"] = df["tipe_kos"].astype("category").cat.codes.astype(int)

    required_cols = [
        "id_kos",
        "nama_kos",
        "kota",
        "wilayah",
        "harga_per_bulan",
        "tipe_kos",
        "ukuran_kamar",
        "ac",
        "kamar_mandi_dalam",
        "wifi",
        "listrik_include",
        "parkir",
        "dapur",
        "laundry",
        "security_24jam",
        "total_fasilitas",
        "jarak_ke_kampus_km",
        "jarak_ke_transportasi_km",
        "jarak_kampus_dekat",
        "jarak_transportasi_dekat",
        "kota_encoded",
        "tipe_kos_encoded",
        "foto_path",
        "rating",
        "deskripsi_clean",
        "latitude",
        "longitude",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di CSV: {missing}")

    df = df[required_cols].copy()
    df = df.where(pd.notnull(df), None)

    records = df.to_dict(orient="records")

    engine = create_engine(sync_url)

    create_table_sql = """
    DROP TABLE IF EXISTS kos_listings;

    CREATE TABLE kos_listings (
        id_kos INTEGER PRIMARY KEY,
        nama_kos TEXT,
        kota TEXT,
        wilayah TEXT,
        harga_per_bulan INTEGER,
        tipe_kos TEXT,
        ukuran_kamar INTEGER,

        ac INTEGER,
        kamar_mandi_dalam INTEGER,
        wifi INTEGER,
        listrik_include INTEGER,
        parkir INTEGER,
        dapur INTEGER,
        laundry INTEGER,
        security_24jam INTEGER,

        total_fasilitas INTEGER,
        jarak_ke_kampus_km DOUBLE PRECISION,
        jarak_ke_transportasi_km DOUBLE PRECISION,
        jarak_kampus_dekat INTEGER,
        jarak_transportasi_dekat INTEGER,

        kota_encoded INTEGER,
        tipe_kos_encoded INTEGER,

        foto_path TEXT,
        rating DOUBLE PRECISION,
        deskripsi_clean TEXT,

        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        geom GEOGRAPHY(Point, 4326)
    );
    """

    insert_sql = text("""
        INSERT INTO kos_listings (
            id_kos,
            nama_kos,
            kota,
            wilayah,
            harga_per_bulan,
            tipe_kos,
            ukuran_kamar,
            ac,
            kamar_mandi_dalam,
            wifi,
            listrik_include,
            parkir,
            dapur,
            laundry,
            security_24jam,
            total_fasilitas,
            jarak_ke_kampus_km,
            jarak_ke_transportasi_km,
            jarak_kampus_dekat,
            jarak_transportasi_dekat,
            kota_encoded,
            tipe_kos_encoded,
            foto_path,
            rating,
            deskripsi_clean,
            latitude,
            longitude,
            geom
        )
        VALUES (
            :id_kos,
            :nama_kos,
            :kota,
            :wilayah,
            :harga_per_bulan,
            :tipe_kos,
            :ukuran_kamar,
            :ac,
            :kamar_mandi_dalam,
            :wifi,
            :listrik_include,
            :parkir,
            :dapur,
            :laundry,
            :security_24jam,
            :total_fasilitas,
            :jarak_ke_kampus_km,
            :jarak_ke_transportasi_km,
            :jarak_kampus_dekat,
            :jarak_transportasi_dekat,
            :kota_encoded,
            :tipe_kos_encoded,
            :foto_path,
            :rating,
            :deskripsi_clean,
            :latitude,
            :longitude,
            ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326)::geography
        )
    """)

    print("Mengaktifkan PostGIS extension...")
    print("Membuat ulang tabel kos_listings...")
    print("Mengimport data...")

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.execute(text(create_table_sql))
        conn.execute(insert_sql, records)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS kos_listings_geom_idx
            ON kos_listings USING GIST (geom);
        """))

    print(f"Berhasil import {len(records)} baris ke tabel kos_listings.")


if __name__ == "__main__":
    main()