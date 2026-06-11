from sqlalchemy import Column, Integer, Numeric, SmallInteger, String, Text, TIMESTAMP
from sqlalchemy.sql import func

from app.core.database import Base


class KosListing(Base):
    """
    SQLAlchemy ORM model untuk tabel kos_listings di PostgreSQL.

    Catatan arsitektur:
    Kolom 'geom' (GEOGRAPHY PostGIS) TIDAK didefinisikan di sini secara eksplisit
    agar tidak perlu dependency GeoAlchemy2 yang berat. Kolom geom dikelola
    langsung oleh trigger fn_sync_geom_from_latlon() di level database —
    setiap kali latitude/longitude di-insert atau di-update, PostGIS
    otomatis men-generate kolom geom yang benar.

    Query geospatial (ST_DWithin, ST_Distance) tetap bisa dieksekusi via
    text() atau raw SQL di service layer tanpa ORM.
    """

    __tablename__ = "kos_listings"

    # --- Primary Key ---
    id = Column(Integer, primary_key=True, autoincrement=True)
    id_kos = Column(Integer, unique=True, nullable=False, index=True)

    # --- Identitas ---
    nama_kos = Column(String(255), nullable=False)

    # --- Lokasi Tekstual ---
    kota = Column(String(50), nullable=False, index=True)
    wilayah = Column(String(100))

    # --- Harga ---
    harga_per_bulan = Column(Integer, nullable=False, index=True)

    # --- Karakteristik Fisik ---
    tipe_kos = Column(String(20))
    ukuran_kamar = Column(Integer)
    jumlah_kamar = Column(Integer)

    # --- Fasilitas (binary: 0 = tidak ada, 1 = ada) ---
    ac = Column(SmallInteger, default=0)
    kamar_mandi_dalam = Column(SmallInteger, default=0)
    wifi = Column(SmallInteger, default=0)
    listrik_include = Column(SmallInteger, default=0)
    parkir = Column(SmallInteger, default=0)
    dapur = Column(SmallInteger, default=0)
    laundry = Column(SmallInteger, default=0)
    security_24jam = Column(SmallInteger, default=0)

    # --- Derived Features (dihitung saat ingest, digunakan langsung oleh model ML) ---
    total_fasilitas = Column(SmallInteger, default=0)
    jarak_ke_kampus_km = Column(Numeric(5, 2))
    jarak_ke_transportasi_km = Column(Numeric(5, 2))
    jarak_kampus_dekat = Column(SmallInteger, default=0)        # 1 jika < 2 km
    jarak_transportasi_dekat = Column(SmallInteger, default=0)  # 1 jika < 1 km

    # --- Review ---
    rating = Column(Numeric(3, 2))
    jumlah_review = Column(Integer, default=0)

    # --- Konten Teks ---
    deskripsi_promosi = Column(Text)
    ulasan_penghuni = Column(Text)
    deskripsi_clean = Column(Text)  # Output pipeline Sastrawi — input FAISS indexer

    # --- Media ---
    foto_path = Column(String(500))

    # --- Geospatial (latitude/longitude disinkronkan ke geom via trigger PostGIS) ---
    latitude = Column(Numeric(10, 6))
    longitude = Column(Numeric(10, 6))
    # Kolom geom (GEOGRAPHY) dikelola via trigger — tidak perlu ORM mapping

    # --- Encoded Features untuk inferensi ML langsung dari DB ---
    kota_encoded = Column(SmallInteger)
    tipe_kos_encoded = Column(SmallInteger)

    # --- Metadata ---
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )