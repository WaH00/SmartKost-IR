-- migrations/001_create_kos_table.sql
-- Skema database Smart-Kos Hybrid Engine v1.0
-- Membutuhkan: PostgreSQL 13+ dengan ekstensi PostGIS
-- Idempoten: Aman dijalankan berulang kali (gunakan IF NOT EXISTS)

-- Aktifkan ekstensi PostGIS
-- Diperlukan untuk tipe data GEOGRAPHY dan fungsi ST_* (ST_DWithin, ST_Distance, dll.)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- ==============================================================================
-- TABEL UTAMA: kos_listings
-- ==============================================================================
CREATE TABLE IF NOT EXISTS kos_listings (

    -- Primary Key
    id                          SERIAL PRIMARY KEY,
    id_kos                      INTEGER UNIQUE NOT NULL,

    -- Identitas kos
    nama_kos                    VARCHAR(255) NOT NULL,

    -- Lokasi tekstual
    kota                        VARCHAR(50)  NOT NULL,
    wilayah                     VARCHAR(100),

    -- Harga (satuan: Rupiah)
    harga_per_bulan             INTEGER NOT NULL,

    -- Karakteristik fisik
    tipe_kos                    VARCHAR(20) CHECK (tipe_kos IN ('Putra', 'Putri', 'Campur')),
    ukuran_kamar                INTEGER,
    jumlah_kamar                INTEGER,

    -- Fasilitas binary (0 = tidak ada, 1 = ada)
    ac                          SMALLINT DEFAULT 0 CHECK (ac IN (0, 1)),
    kamar_mandi_dalam           SMALLINT DEFAULT 0 CHECK (kamar_mandi_dalam IN (0, 1)),
    wifi                        SMALLINT DEFAULT 0 CHECK (wifi IN (0, 1)),
    listrik_include             SMALLINT DEFAULT 0 CHECK (listrik_include IN (0, 1)),
    parkir                      SMALLINT DEFAULT 0 CHECK (parkir IN (0, 1)),
    dapur                       SMALLINT DEFAULT 0 CHECK (dapur IN (0, 1)),
    laundry                     SMALLINT DEFAULT 0 CHECK (laundry IN (0, 1)),
    security_24jam              SMALLINT DEFAULT 0 CHECK (security_24jam IN (0, 1)),

    -- Derived features (dihitung saat ingest; siap digunakan langsung oleh model ML)
    total_fasilitas             SMALLINT DEFAULT 0,
    jarak_ke_kampus_km          NUMERIC(5, 2),
    jarak_ke_transportasi_km    NUMERIC(5, 2),
    jarak_kampus_dekat          SMALLINT DEFAULT 0,  -- 1 jika jarak < 2 km dari kampus
    jarak_transportasi_dekat    SMALLINT DEFAULT 0,  -- 1 jika jarak < 1 km dari transportasi umum

    -- Review penghuni
    rating                      NUMERIC(3, 2),
    jumlah_review               INTEGER DEFAULT 0,

    -- Konten teks
    deskripsi_promosi           TEXT,
    ulasan_penghuni             TEXT,
    deskripsi_clean             TEXT,  -- Output pipeline Sastrawi — input FAISS indexer

    -- Media
    foto_path                   VARCHAR(500),

    -- === KOLOM GEOSPATIAL (PostGIS) ===
    -- Tipe GEOGRAPHY dipilih karena menggunakan kalkulasi geodesik (ellipsoidal earth)
    -- yang lebih akurat untuk ST_DWithin/ST_Distance dibanding GEOMETRY (flat-earth).
    -- SRID 4326 = WGS84, standar koordinat GPS yang universal.
    -- Nilai geom di-generate OTOMATIS oleh trigger fn_sync_geom_from_latlon()
    -- setiap kali kolom latitude/longitude di-insert atau di-update.
    latitude                    NUMERIC(10, 6),
    longitude                   NUMERIC(10, 6),
    geom                        GEOGRAPHY(POINT, 4326),

    -- Encoded features untuk inferensi model ML langsung dari DB (tanpa re-encode)
    kota_encoded                SMALLINT,
    tipe_kos_encoded            SMALLINT,

    -- Metadata audit trail
    created_at                  TIMESTAMPTZ DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ DEFAULT NOW()
);

-- ==============================================================================
-- TRIGGER FUNCTION: Sinkronisasi otomatis kolom geom dari latitude + longitude
-- Dipanggil setiap BEFORE INSERT OR UPDATE pada kolom latitude/longitude
-- ==============================================================================
CREATE OR REPLACE FUNCTION fn_sync_geom_from_latlon()
RETURNS TRIGGER AS $$
BEGIN
    -- Sinkronisasi hanya jika latitude dan longitude keduanya tersedia
    IF NEW.latitude IS NOT NULL AND NEW.longitude IS NOT NULL THEN
        -- ST_MakePoint(longitude, latitude) — URUTAN: lon dulu, lat kemudian
        -- Ini adalah konvensi OGC/PostGIS (x=lon, y=lat), berbeda dengan GPS (lat, lon)
        -- ST_SetSRID(..., 4326) — tentukan Spatial Reference ID = WGS84
        -- Cast ke GEOGRAPHY agar ST_DWithin menggunakan satuan meter
        NEW.geom = ST_SetSRID(
            ST_MakePoint(NEW.longitude, NEW.latitude),
            4326
        )::GEOGRAPHY;
    END IF;
    -- Update timestamp setiap ada perubahan data
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Pasang trigger — aktif pada INSERT dan UPDATE (khususnya bila lat/lon berubah)
DROP TRIGGER IF EXISTS trg_sync_geom ON kos_listings;
CREATE TRIGGER trg_sync_geom
    BEFORE INSERT OR UPDATE OF latitude, longitude
    ON kos_listings
    FOR EACH ROW
    EXECUTE FUNCTION fn_sync_geom_from_latlon();

-- ==============================================================================
-- INDEX: Optimasi performa query
-- ==============================================================================

-- Indeks GIST untuk query radius PostGIS (ST_DWithin, ST_DWithin)
-- INI WAJIB ADA. Tanpa indeks GIST, query radius = full table scan = sangat lambat.
-- Dengan GIST, kompleksitas turun dari O(N) ke O(log N).
CREATE INDEX IF NOT EXISTS idx_kos_geom
    ON kos_listings USING GIST (geom);

-- Indeks B-tree untuk kolom yang sering muncul di WHERE dan ORDER BY
CREATE INDEX IF NOT EXISTS idx_kos_kota    ON kos_listings (kota);
CREATE INDEX IF NOT EXISTS idx_kos_harga   ON kos_listings (harga_per_bulan);
CREATE INDEX IF NOT EXISTS idx_kos_tipe    ON kos_listings (tipe_kos);
CREATE INDEX IF NOT EXISTS idx_kos_rating  ON kos_listings (rating DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_kos_id_kos  ON kos_listings (id_kos);

-- ==============================================================================
-- VIEW: Statistik ringkasan harga kos per kota
-- Berguna untuk analytics dashboard dan validasi distribusi data
-- ==============================================================================
CREATE OR REPLACE VIEW v_kos_summary_per_kota AS
SELECT
    kota,
    COUNT(*)                                   AS total_kos,
    ROUND(AVG(harga_per_bulan))                AS avg_harga,
    MIN(harga_per_bulan)                       AS min_harga,
    MAX(harga_per_bulan)                       AS max_harga,
    ROUND(AVG(rating)::NUMERIC, 2)             AS avg_rating,
    ROUND(AVG(total_fasilitas)::NUMERIC, 1)    AS avg_fasilitas
FROM kos_listings
GROUP BY kota
ORDER BY avg_harga DESC;

-- Konfirmasi eksekusi migrasi
DO $$
BEGIN
    RAISE NOTICE '=== Migrasi 001_create_kos_table.sql BERHASIL ===';
    RAISE NOTICE 'Tabel    : kos_listings';
    RAISE NOTICE 'Trigger  : trg_sync_geom (auto-sync kolom geom dari lat/lon)';
    RAISE NOTICE 'Index    : GIST spatial + B-tree (kota, harga, tipe, rating)';
    RAISE NOTICE 'View     : v_kos_summary_per_kota';
END $$;