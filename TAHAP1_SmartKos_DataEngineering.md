# TAHAP 1 — Smart-Kos Hybrid Engine
## Data Engineering & Fondasi Arsitektur Database

---

### Struktur Folder Backend

```
smart-kos-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI entry point & lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic Settings (env vars)
│   │   └── database.py              # Async SQLAlchemy engine
│   ├── models/
│   │   ├── __init__.py
│   │   └── kos.py                   # SQLAlchemy ORM model
│   ├── api/v1/
│   │   └── router.py                # (Diisi Tahap 2+)
│   └── services/
│       └── (Diisi Tahap 2+)
├── stki/
│   ├── __init__.py
│   ├── preprocessor.py              # Pipeline Sastrawi (4 langkah)
│   └── indexer.py                   # FAISS index manager (TF-IDF Tahap 1)
├── scripts/
│   ├── augment_dataset.py           # Augmentasi CSV dataset
│   ├── preprocess_dataset.py        # Batch preprocessing teks
│   ├── init_postgis.py              # Load data ke PostgreSQL + PostGIS
│   └── build_faiss_index.py         # Bangun FAISS TF-IDF index
├── migrations/
│   └── 001_create_kos_table.sql     # DDL: tabel, trigger, index, view
├── data/
│   ├── raw/                         # <- Taruh data_kos_jakarta.csv di sini
│   ├── augmented/
│   └── processed/
├── models_ml/                       # <- Taruh model_rf.pkl & scaler.pkl
├── faiss_index/
├── .env.example
├── requirements.txt
└── setup.sh
```

---

### Urutan Eksekusi Wajib

```
1.  cp .env.example .env                     → sesuaikan DB credentials
2.  Taruh data_kos_jakarta.csv               → ke data/raw/
3.  Taruh model_rf.pkl, scaler.pkl           → ke models_ml/
4.  bash setup.sh                            → install deps + buat semua folder
5.  python scripts/augment_dataset.py        → tambah 5 kolom baru (GPS, teks, foto)
6.  python scripts/preprocess_dataset.py     → pipeline Sastrawi → deskripsi_clean
7.  createdb smart_kos_db                    → buat database PostgreSQL
8.  python scripts/init_postgis.py           → DDL migration + batch insert PostGIS
9.  python scripts/build_faiss_index.py      → bangun FAISS TF-IDF index
10. uvicorn app.main:app --reload            → jalankan API (http://localhost:8000)
```

---

## FILE 1: `requirements.txt`

```text
# Smart-Kos Hybrid Engine — Python dependencies
# Python 3.10+

# === Web Framework ===
fastapi==0.111.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9

# === Data Validation ===
pydantic==2.7.1
pydantic-settings==2.3.0

# === Database ===
sqlalchemy==2.0.30
asyncpg==0.29.0           # Async PostgreSQL driver (FastAPI runtime)
psycopg2-binary==2.9.9    # Sync PostgreSQL driver (scripts admin/migration)

# === Data Processing ===
pandas==2.2.2
numpy==1.26.4

# === Machine Learning ===
scikit-learn==1.5.0        # TF-IDF, RandomForest, StandardScaler

# === Vector Search ===
faiss-cpu==1.7.4           # CPU FAISS (ganti faiss-gpu jika ada GPU)

# === Indonesian NLP ===
PySastrawi==1.2.0          # Stemmer + Stopword Remover Bahasa Indonesia

# === Utilities ===
python-dotenv==1.0.1
regex==2024.5.15
```

---

## FILE 2: `.env.example`

```env
# Smart-Kos Hybrid Engine — Environment Variables
# Salin file ini: cp .env.example .env, lalu sesuaikan nilainya

# === PostgreSQL ===
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/smart_kos_db
DATABASE_DSN=host=localhost port=5432 dbname=smart_kos_db user=postgres password=password

# === FAISS Paths ===
FAISS_INDEX_PATH=./faiss_index/kos_tfidf.index
FAISS_ID_MAP_PATH=./faiss_index/id_map.pkl
FAISS_VECTORIZER_PATH=./faiss_index/tfidf_vectorizer.pkl

# === Data Paths ===
DATA_RAW_PATH=./data/raw/data_kos_jakarta.csv
DATA_AUGMENTED_PATH=./data/augmented/data_kos_augmented.csv
DATA_PROCESSED_PATH=./data/processed/data_kos_processed.csv

# === ML Model Paths ===
MODEL_RF_PATH=./models_ml/model_rf.pkl
SCALER_PATH=./models_ml/scaler.pkl
LABEL_ENCODER_PATH=./models_ml/label_encoders.pkl

# === App Config ===
APP_NAME=Smart-Kos Hybrid Engine API
APP_VERSION=1.0.0
DEBUG=true
```

---

## FILE 3: `setup.sh`

```bash
#!/bin/bash
# setup.sh — Script persiapan environment Smart-Kos Backend (jalankan sekali)
set -e
echo "🔧 Menyiapkan Smart-Kos Hybrid Engine Backend..."

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Buat seluruh struktur direktori
mkdir -p data/raw data/augmented data/processed
mkdir -p models_ml faiss_index
mkdir -p app/core app/api/v1 app/models app/services
mkdir -p stki scripts migrations assets/kos_photos

# Buat file __init__.py untuk setiap package Python
touch app/__init__.py app/core/__init__.py
touch app/api/__init__.py app/api/v1/__init__.py
touch app/models/__init__.py app/services/__init__.py
touch stki/__init__.py

# Salin template .env jika belum ada
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠  File .env dibuat. Sesuaikan konfigurasi database Anda sebelum lanjut."
fi

echo "✅ Setup selesai! Ikuti Urutan Eksekusi di README_TAHAP1."
```

---

## FILE 4: `app/core/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Konfigurasi terpusat Smart-Kos Backend menggunakan Pydantic-Settings.
    Seluruh nilai dibaca otomatis dari file .env via pola Singleton.
    Menambahkan variabel baru cukup di sini — tidak perlu ubah modul lain.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Konfigurasi Database PostgreSQL ---
    database_url: str = (
        "postgresql+asyncpg://postgres:password@localhost:5432/smart_kos_db"
    )
    # DATABASE_DSN digunakan oleh psycopg2 di scripts admin (bukan FastAPI runtime)
    database_dsn: str = (
        "host=localhost port=5432 dbname=smart_kos_db user=postgres password=password"
    )

    # --- Konfigurasi FAISS Vector Index ---
    faiss_index_path: str = "./faiss_index/kos_tfidf.index"
    faiss_id_map_path: str = "./faiss_index/id_map.pkl"
    faiss_vectorizer_path: str = "./faiss_index/tfidf_vectorizer.pkl"

    # --- Konfigurasi Path File Data ---
    data_raw_path: str = "./data/raw/data_kos_jakarta.csv"
    data_augmented_path: str = "./data/augmented/data_kos_augmented.csv"
    data_processed_path: str = "./data/processed/data_kos_processed.csv"

    # --- Konfigurasi Model Machine Learning ---
    model_rf_path: str = "./models_ml/model_rf.pkl"
    scaler_path: str = "./models_ml/scaler.pkl"
    label_encoder_path: str = "./models_ml/label_encoders.pkl"

    # --- Konfigurasi Aplikasi ---
    app_name: str = "Smart-Kos Hybrid Engine API"
    app_version: str = "1.0.0"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"


# Instans singleton — diimpor oleh seluruh modul lain dengan:
#   from app.core.config import settings
settings = Settings()
```

---

## FILE 5: `app/core/database.py`

```python
import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Engine async untuk operasi CRUD via FastAPI ---
# pool_pre_ping=True: cek koneksi mati sebelum digunakan (cegah idle timeout error)
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,   # Log SQL query di mode debug (matikan di production)
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# --- Session factory ---
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Kelas dasar untuk seluruh SQLAlchemy ORM model aplikasi."""
    pass


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency injector untuk async database session.
    Menjamin session di-commit, di-rollback jika error, dan ditutup
    setelah setiap request HTTP selesai (Resource cleanup otomatis).

    Penggunaan di route handler:
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(KosListing))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(f"Database session error: {exc}")
            raise
        finally:
            await session.close()
```

---

## FILE 6: `app/main.py`

```python
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manajemen siklus hidup aplikasi FastAPI.
    Titik injeksi untuk startup/shutdown resources berat (FAISS, ML model).

    Tahap 2: Tambahkan load FAISS index dan TF-IDF vectorizer.
    Tahap 4: Tambahkan load model Random Forest dan StandardScaler.
    """
    # === STARTUP ===
    logger.info(f"🚀 Memulai {settings.app_name} v{settings.app_version}...")

    # [Tahap 2] Uncomment setelah KosVectorIndexer selesai:
    # from stki.indexer import KosVectorIndexer
    # indexer = KosVectorIndexer(...)
    # indexer.load_index()
    # app.state.faiss_indexer = indexer

    # [Tahap 4] Uncomment setelah PricingService selesai:
    # from app.services.pricing import PricingService
    # app.state.pricing_service = PricingService(settings.model_rf_path, ...)

    logger.info("✅ Startup selesai. API siap menerima request.")

    yield  # Aplikasi aktif di antara startup dan shutdown

    # === SHUTDOWN ===
    logger.info("🛑 Menghentikan aplikasi...")
    # [Tahap 2+] Bersihkan resource di sini jika diperlukan
    logger.info("Cleanup selesai.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "**Smart-Kos Hybrid Engine** — Platform PropTech 4 Pilar Kecerdasan:\n\n"
        "1. 🧠 **Semantic IR**: IndoBERT + FAISS vector search\n"
        "2. 🔄 **Auto-Correction**: TF-IDF Pseudo-Relevance Feedback (PRF)\n"
        "3. 📍 **Geospatial**: PostGIS ST_DWithin radius search\n"
        "4. 💰 **Predictive Pricing**: Random Forest price classification"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware — izinkan Flutter frontend mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain Flutter spesifik di environment production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root() -> dict:
    """Endpoint root — konfirmasi API aktif."""
    return {
        "status": "online",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Detailed health check endpoint.
    Status setiap komponen akan diperbarui seiring bertambahnya tahap.
    """
    return {
        "status": "healthy",
        "database": "postgresql+postgis",
        "vector_index": "not_loaded",   # Diisi Tahap 2
        "ml_model": "not_loaded",        # Diisi Tahap 4
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
```

---

## FILE 7: `app/models/kos.py`

```python
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
```

---

## FILE 8: `stki/preprocessor.py`

```python
import logging
import re
from typing import Optional

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

logger = logging.getLogger(__name__)


class IndonesianTextPreprocessor:
    """
    Pipeline preprocessing teks Bahasa Indonesia menggunakan library Sastrawi.

    Urutan 4 langkah wajib (TIDAK boleh diubah urutannya):
        1. Normalisasi  → lowercase, hapus URL dan emoji Unicode
        2. Pembersihan  → hapus angka, tanda baca, karakter non-alfabet
        3. Stopword removal → hapus kata hubung (kamus 757+ kata Sastrawi)
        4. Stemming     → reduksi morfologi ke kata dasar (algoritma ECS)

    Alasan urutan: pembersihan HARUS sebelum stemming agar stemmer tidak
    menerima token berisi karakter non-alfabet yang merusak kalkulasi
    algoritma Enhanced Confix Stripping (ECS) Nazief-Adriani.

    Contoh transformasi lengkap:
        Input : "Kos putri dengan kamar mandi dalam & AC, wifi kencang!"
        Step 1: "kos putri dengan kamar mandi dalam  ac wifi kencang"
        Step 2: "kos putri dengan kamar mandi dalam  ac wifi kencang"
        Step 3: "kos putri kamar mandi ac wifi kencang" (hapus 'dengan')
        Output: "kos putri kamar mandi ac wifi kencang" (sudah terstemming)
    """

    def __init__(self) -> None:
        """
        Inisialisasi komponen Sastrawi.
        Dilakukan SEKALI saja karena loading kamus dari disk cukup mahal (~1-2 detik).
        Instansiasi ulang di setiap request/baris = pemborosan resource.
        """
        logger.info("Menginisialisasi IndonesianTextPreprocessor...")

        # Stemmer menggunakan algoritma Enhanced Confix Stripping (ECS)
        stemmer_factory = StemmerFactory()
        self._stemmer = stemmer_factory.create_stemmer()

        # Stopword remover dengan kamus 757+ kata Bahasa Indonesia
        stopword_factory = StopWordRemoverFactory()
        self._stopword_remover = stopword_factory.create_stop_word_remover()

        # Pola regex dikompilasi sekali untuk efisiensi batch processing
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+", re.IGNORECASE
        )
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Simbol & piktogram
            "\U0001F680-\U0001F6FF"  # Transport & peta
            "\U0001F1E0-\U0001F1FF"  # Bendera negara
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        self._non_alpha_pattern = re.compile(r"[^a-zA-Z\s]")
        self._whitespace_pattern = re.compile(r"\s+")

        logger.info("IndonesianTextPreprocessor siap digunakan.")

    def _normalize(self, text: str) -> str:
        """
        Langkah 1 — Normalisasi.
        Mengubah ke huruf kecil, menghapus URL dan emoji.
        URL harus dihapus sebelum langkah lain karena '://' dan '.' akan
        menjadi token yang merusak kualitas stemming.
        """
        text = text.lower()
        text = self._url_pattern.sub(" ", text)
        text = self._emoji_pattern.sub(" ", text)
        return text

    def _clean(self, text: str) -> str:
        """
        Langkah 2 — Pembersihan karakter.
        Menghapus angka, tanda baca, dan karakter non-alfabet.
        Menormalisasi spasi ganda, tab, dan newline menjadi satu spasi.
        """
        text = self._non_alpha_pattern.sub(" ", text)
        text = self._whitespace_pattern.sub(" ", text).strip()
        return text

    def _remove_stopwords(self, text: str) -> str:
        """
        Langkah 3 — Penghapusan stopwords Bahasa Indonesia.
        Kamus Sastrawi mencakup 757+ kata umum: 'yang', 'dan', 'dengan',
        'untuk', 'adalah', 'ini', 'itu', dll.
        """
        return self._stopword_remover.remove(text)

    def _stem(self, text: str) -> str:
        """
        Langkah 4 — Stemming morfologi Bahasa Indonesia.
        Algoritma ECS (Enhanced Confix Stripping, turunan Nazief-Adriani).
        Contoh: 'memberikan' → 'beri', 'pelayanan' → 'layan',
                'kebersihan' → 'bersih', 'pemrosesan' → 'proses'.
        """
        return self._stemmer.stem(text)

    def preprocess(self, text: Optional[str]) -> str:
        """
        Eksekusi pipeline preprocessing lengkap secara berurutan.

        Args:
            text: Teks mentah Bahasa Indonesia.

        Returns:
            Teks bersih setelah 4 langkah pipeline.
            String kosong jika input None, bukan string, atau gagal diproses.
        """
        if not text or not isinstance(text, str):
            return ""

        try:
            text = self._normalize(text)
            text = self._clean(text)
            text = self._remove_stopwords(text)
            text = self._stem(text)
            return text
        except Exception as exc:
            logger.warning(
                f"Gagal memproses teks '{str(text)[:50]}...' | Error: {exc}"
            )
            return ""

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """
        Preprocessing batch untuk efisiensi pemrosesan CSV berukuran besar.
        Urutan output dijamin sama dengan urutan input.

        Args:
            texts: List teks mentah Bahasa Indonesia (satu elemen per dokumen).

        Returns:
            List teks yang sudah diproses (indeks berkorespondensi dengan input).
        """
        total = len(texts)
        logger.info(f"Memulai batch preprocessing untuk {total} dokumen...")
        results = [self.preprocess(t) for t in texts]

        # Validasi kualitas output
        empty_count = sum(1 for r in results if not r)
        logger.info(
            f"Preprocessing selesai. Berhasil: {total - empty_count}/{total}. "
            f"Output kosong: {empty_count}."
        )
        return results
```

---

## FILE 9: `stki/indexer.py`

```python
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class KosVectorIndexer:
    """
    Manager FAISS Vector Index untuk pencarian semantik kos-kosan.

    =====================================================================
    CATATAN ARSITEKTUR MIGRASI (PENTING untuk Tahap 2):

    TAHAP 1 — TF-IDF Placeholder (implementasi saat ini):
        Vektorisasi menggunakan TF-IDF (sparse → dense float32).
        Dimensi vektor: dinamis, dibatasi max_features=5000.
        Cocok sebagai fondasi dan validasi pipeline FAISS.

    TAHAP 2 — IndoBERT Sentence Encoder (akan menggantikan TF-IDF):
        Ganti isi method _vectorize() dan _vectorize_query() dengan
        IndoBERT dari HuggingFace: 'indobenchmark/indobert-base-p1'.
        Dimensi vektor IndoBERT: 768 (lihat konstanta INDOBERT_DIM).
        Interface publik (build_index, search, load_index) TIDAK BERUBAH.
        Konstanta INDOBERT_DIM sudah disiapkan sebagai referensi migrasi.
    =====================================================================

    Attributes:
        index_path      : Path file FAISS binary index (.index)
        id_map_path     : Path file mapping posisi FAISS → id_kos asli
        vectorizer_path : Path file TF-IDF vectorizer (diganti IndoBERT Tahap 2)
    """

    # Dimensi vektor IndoBERT — konstanta referensi untuk migrasi Tahap 2
    INDOBERT_DIM: int = 768

    def __init__(
        self,
        index_path: str,
        id_map_path: str,
        vectorizer_path: str,
    ) -> None:
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.vectorizer_path = Path(vectorizer_path)

        # State runtime (diisi saat build_index() atau load_index() dipanggil)
        self.index: Optional[faiss.Index] = None
        self.id_map: list[int] = []
        self.vectorizer: Optional[TfidfVectorizer] = None

        # Pastikan direktori output sudah ada
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # TAHAP 1: TF-IDF vectorization (PLACEHOLDER — akan diganti Tahap 2)
    # ------------------------------------------------------------------

    def _vectorize(self, documents: list[str]) -> np.ndarray:
        """
        [TAHAP 1 PLACEHOLDER] Vektorisasi dokumen menggunakan TF-IDF.

        Pada Tahap 2, seluruh isi method ini diganti dengan:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('indobenchmark/indobert-base-p1')
            vectors = model.encode(documents, normalize_embeddings=True)
            return vectors.astype(np.float32)

        Interface input-output tetap sama: list[str] → np.ndarray float32.

        Konfigurasi TF-IDF yang dipilih:
            max_features=5000 : Batasi dimensi agar FAISS efisien di CPU
            ngram_range=(1,2) : Unigram + bigram untuk konteks lebih kaya
            min_df=2          : Abaikan term yang hanya muncul di 1 dokumen
            sublinear_tf=True : Gunakan log(1+tf) untuk normalisasi TF
                                (mengurangi dominasi term frekuensi sangat tinggi)

        Returns:
            Matrix float32 (N, D) yang sudah dinormalisasi L2.
            Normalisasi L2 + IndexFlatIP = pencarian cosine similarity.
        """
        logger.info("[Tahap 1] Menjalankan TF-IDF vectorization...")

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )

        # Fit + transform sekaligus (efisien untuk pipeline offline)
        sparse_matrix = self.vectorizer.fit_transform(documents)

        # Konversi sparse → dense float32 karena FAISS tidak menerima sparse matrix
        vectors: np.ndarray = sparse_matrix.toarray().astype(np.float32)

        # Normalisasi L2: setiap vektor dijadikan unit vector
        # Tujuan: IndexFlatIP (inner product) = cosine similarity setelah L2-norm
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Hindari division by zero untuk dokumen kosong
        vectors = vectors / norms

        logger.info(f"Vektorisasi selesai. Shape matrix: {vectors.shape}")
        return vectors

    def _vectorize_query(self, query_text: str) -> np.ndarray:
        """
        Vektorisasi satu teks kueri menggunakan vectorizer yang sudah di-fit.
        Pastikan load_index() sudah dipanggil sebelum method ini.

        Args:
            query_text: Teks kueri yang sudah dipreproses oleh Sastrawi pipeline.

        Returns:
            Array float32 shape (1, D) yang sudah dinormalisasi L2.
        """
        if self.vectorizer is None:
            raise RuntimeError(
                "Vectorizer belum dimuat. Panggil load_index() terlebih dahulu."
            )

        query_vec = (
            self.vectorizer.transform([query_text]).toarray().astype(np.float32)
        )
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        return query_vec.reshape(1, -1)

    # ------------------------------------------------------------------
    # Interface Publik — TIDAK BERUBAH antara Tahap 1 dan Tahap 2
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: list[str],
        kos_ids: list[int],
    ) -> None:
        """
        Membangun FAISS index dari list dokumen teks yang sudah dipreproses.

        Proses berurutan:
            1. Vektorisasi dokumen (TF-IDF Tahap 1 / IndoBERT Tahap 2)
            2. Normalisasi L2 setiap vektor
            3. Inisialisasi FAISS IndexFlatIP (brute-force cosine similarity)
            4. Tambahkan semua vektor ke index
            5. Simpan index, id_map, dan vectorizer ke disk

        Catatan pemilihan IndexFlatIP:
            - Brute-force, tidak ada approximation error
            - Optimal untuk dataset < 100k dokumen
            - Untuk dataset > 100k: gunakan IndexIVFFlat dengan nlist=sqrt(N)

        Args:
            documents: List string 'deskripsi_clean' dari dataset.
            kos_ids  : List id_kos berkorespondensi (indeks HARUS sinkron).

        Raises:
            ValueError: Jika panjang documents dan kos_ids tidak sama.
        """
        if len(documents) != len(kos_ids):
            raise ValueError(
                f"Panjang documents ({len(documents)}) "
                f"tidak sama dengan kos_ids ({len(kos_ids)})."
            )
        if not documents:
            raise ValueError("List dokumen tidak boleh kosong.")

        logger.info(f"Membangun FAISS index untuk {len(documents)} dokumen...")

        vectors = self._vectorize(documents)
        vector_dim = vectors.shape[1]

        self.index = faiss.IndexFlatIP(vector_dim)
        self.index.add(vectors)
        self.id_map = list(kos_ids)

        logger.info(
            f"FAISS index dibangun: {self.index.ntotal} vektor, "
            f"{vector_dim} dimensi."
        )
        self._persist()

    def search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Mencari kos dengan teks paling relevan terhadap kueri.

        Args:
            query_text: Teks kueri yang sudah dipreproses oleh Sastrawi pipeline.
            top_k     : Jumlah hasil teratas yang dikembalikan (default: 10).

        Returns:
            List dict terurut berdasarkan relevansi (tertinggi di posisi 0):
                [{"id_kos": int, "score": float}, ...]
            Score adalah cosine similarity (0.0–1.0).

        Raises:
            RuntimeError: Jika index belum dimuat (load_index() belum dipanggil).
        """
        if self.index is None:
            raise RuntimeError(
                "FAISS index belum dimuat. Panggil load_index() terlebih dahulu."
            )

        query_vector = self._vectorize_query(query_text)
        scores, indices = self.index.search(query_vector, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS mengembalikan -1 jika jumlah dokumen < top_k
                continue
            results.append({
                "id_kos": self.id_map[int(idx)],
                "score": float(score),
            })
        return results

    def load_index(self) -> None:
        """
        Memuat FAISS index, id_map, dan vectorizer dari disk ke memori.
        Dipanggil saat aplikasi FastAPI startup (di lifespan context manager).

        Raises:
            FileNotFoundError: Jika file index tidak ditemukan.
                               Artinya scripts/build_faiss_index.py belum dijalankan.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index tidak ditemukan: {self.index_path}.\n"
                "Solusi: Jalankan python scripts/build_faiss_index.py"
            )

        logger.info(f"Memuat FAISS index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        with open(self.id_map_path, "rb") as f:
            self.id_map = pickle.load(f)

        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        logger.info(f"FAISS index dimuat: {self.index.ntotal} vektor.")

    def _persist(self) -> None:
        """Simpan FAISS index, id_map, dan vectorizer ke disk."""
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"FAISS index disimpan: {self.index_path}")

        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)
        logger.info(f"ID map disimpan: {self.id_map_path}")

        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Vectorizer disimpan: {self.vectorizer_path}")
```

---

## FILE 10: `scripts/augment_dataset.py`

```python
"""
Script augmentasi dataset kos-kosan Jakarta.

Menambahkan 5 kolom baru ke CSV dataset asli:
    deskripsi_promosi : Teks promosi panjang dari pemilik kos (bahasa alami)
    ulasan_penghuni   : Teks ulasan dari penghuni sebelumnya
    foto_path         : Path relatif ke file foto kamar
    latitude          : Koordinat GPS lintang (Gaussian noise sekitar pusat kota)
    longitude         : Koordinat GPS bujur

Dijalankan SEKALI sebagai langkah pertama data engineering pipeline.
Sepenuhnya reproducible karena menggunakan numpy random seed tetap (default: 42).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# KONSTANTA: Koordinat pusat setiap kota sebagai anchor GPS
# Sumber: titik tengah area administratif Jakarta (WGS84 / EPSG:4326)
# =============================================================================
CITY_COORD_ANCHORS: dict[str, dict[str, float]] = {
    "Jakarta Pusat":   {"lat": -6.1862, "lon": 106.8348},
    "Jakarta Selatan": {"lat": -6.2615, "lon": 106.8106},
    "Jakarta Timur":   {"lat": -6.2250, "lon": 106.9004},
    "Jakarta Barat":   {"lat": -6.1744, "lon": 106.7629},
    "Jakarta Utara":   {"lat": -6.1213, "lon": 106.8649},
}

# Std dev Gaussian noise koordinat: 0.025° ≈ 2.75 km (mensimulasikan sebaran kos nyata)
GPS_SCATTER_STD: float = 0.025

# =============================================================================
# TEMPLATE TEKS: Basis generasi teks sintetis yang bervariasi per baris
# =============================================================================

DESKRIPSI_TEMPLATES: list[str] = [
    (
        "Kos {tipe} strategis di {wilayah}, {kota}. Kamar {ukuran}m² dilengkapi "
        "{fasilitas_str}. Lokasi hanya {jarak_kampus} km dari kampus dan "
        "{jarak_transport} km dari moda transportasi umum. Suasana bersih, "
        "aman, dan nyaman. Cocok untuk mahasiswa maupun pekerja muda yang "
        "menginginkan hunian berkualitas dengan harga terjangkau."
    ),
    (
        "Tersedia kamar kos {tipe} di kawasan {wilayah}, {kota}. Luas kamar "
        "{ukuran}m², fasilitas: {fasilitas_str}. Dekat kampus ({jarak_kampus} km) "
        "dan transportasi umum ({jarak_transport} km). Lingkungan tenang, "
        "pemilik kos ramah dan responsif. Harga sudah mencakup biaya perawatan "
        "fasilitas bersama. Ideal untuk pelajar yang baru merantau."
    ),
    (
        "Kos eksklusif {tipe} area {wilayah}! Fasilitas lengkap: {fasilitas_str}. "
        "Kamar {ukuran}m² dengan pencahayaan dan ventilasi optimal. Akses mudah "
        "ke pusat bisnis {kota}. Penjaga aktif 24 jam, CCTV terpasang di area "
        "umum. Hanya {jarak_kampus} km ke kampus terdekat. Daftar sekarang!"
    ),
    (
        "Hunian nyaman di jantung {kota}. Kos {tipe} dengan {fasilitas_str}. "
        "Ukuran kamar {ukuran}m², cukup untuk bekerja dan beristirahat. "
        "Bebas banjir, akses 24 jam. Jarak ke transportasi umum hanya "
        "{jarak_transport} km. Cocok untuk profesional muda yang aktif."
    ),
    (
        "Kos bersih dan terawat di {wilayah}, {kota}. Kamar {tipe} berukuran "
        "{ukuran}m² tersedia dengan fasilitas {fasilitas_str}. Pengelolaan "
        "profesional, kontrak fleksibel per bulan. Jarak {jarak_kampus} km "
        "dari kampus, {jarak_transport} km dari halte atau stasiun. "
        "Survei gratis, hubungi pemilik sekarang!"
    ),
]

ULASAN_POSITIF: list[str] = [
    (
        "Sudah {durasi} bulan di sini, sangat puas! Kamar bersih dan wangi. "
        "{catatan} Pemilik kos sigap menangani keluhan. Sangat direkomendasikan!"
    ),
    (
        "Lokasi sangat strategis, ke mana-mana mudah dijangkau. {catatan} "
        "Harga sangat worth it untuk fasilitasnya. Teman sesama penghuni juga asik."
    ),
    (
        "Kos paling nyaman selama kuliah di Jakarta. {catatan} "
        "Lingkungan bersih dan aman, tidak pernah ada masalah keamanan. "
        "Sangat recommended untuk mahasiswa baru!"
    ),
    (
        "Baru {durasi} minggu tapi sudah betah banget. {catatan} "
        "Air bersih dan lancar, listrik stabil, tidak pernah mati. Harga sepadan."
    ),
    (
        "Sudah {durasi} tahun tinggal di sini dan tidak mau pindah. {catatan} "
        "Lingkungan aman dan nyaman, penghuni dipilih secara selektif."
    ),
]

ULASAN_NETRAL: list[str] = [
    (
        "Cukup nyaman untuk kebutuhan sehari-hari. {catatan} "
        "Harga standar untuk area sini. Cocok untuk yang budget terbatas."
    ),
    (
        "Oke secara keseluruhan. {catatan} "
        "Ada plus minusnya, tapi tidak mengecewakan. Lumayan untuk jangka panjang."
    ),
    (
        "Lumayan sesuai harga yang dibayar. {catatan} "
        "Respons pemilik kadang agak lambat, tapi fasilitas tetap terjaga."
    ),
]


def _build_fasilitas_str(row: pd.Series) -> str:
    """
    Menyusun deskripsi fasilitas dalam format kalimat Bahasa Indonesia yang alami.
    Digunakan untuk mengisi placeholder {fasilitas_str} pada template deskripsi.

    Args:
        row: Satu baris DataFrame kos dengan kolom fasilitas binary.

    Returns:
        String fasilitas natural: "AC, WiFi, dan kamar mandi dalam"
    """
    items: list[str] = []
    if row.get("ac", 0) == 1:
        items.append("AC")
    if row.get("kamar_mandi_dalam", 0) == 1:
        items.append("kamar mandi dalam")
    if row.get("wifi", 0) == 1:
        items.append("WiFi")
    if row.get("listrik_include", 0) == 1:
        items.append("listrik termasuk")
    if row.get("parkir", 0) == 1:
        items.append("parkir")
    if row.get("dapur", 0) == 1:
        items.append("dapur bersama")
    if row.get("laundry", 0) == 1:
        items.append("laundry")
    if row.get("security_24jam", 0) == 1:
        items.append("security 24 jam")

    if not items:
        return "fasilitas dasar"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} dan {items[1]}"
    return ", ".join(items[:-1]) + f", dan {items[-1]}"


def _build_catatan_fasilitas(row: pd.Series) -> str:
    """Menghasilkan catatan singkat fasilitas untuk template ulasan."""
    notes: list[str] = []
    if row.get("wifi", 0) == 1:
        notes.append("WiFi kencang dan stabil.")
    if row.get("ac", 0) == 1:
        notes.append("AC dingin, tidak berisik.")
    if row.get("kamar_mandi_dalam", 0) == 1:
        notes.append("Kamar mandi pribadi sangat nyaman.")
    if row.get("security_24jam", 0) == 1:
        notes.append("Keamanan 24 jam membuat tenang.")
    return " ".join(notes[:2]) if notes else "Fasilitas standar terpenuhi."


def _generate_gps(city: str, rng: np.random.Generator) -> tuple[float, float]:
    """
    Menghasilkan koordinat GPS realistis dengan Gaussian noise di sekitar pusat kota.

    Noise Gaussian (std=0.025°) mensimulasikan sebaran lokasi kos yang sesungguhnya
    di dalam satu wilayah kota. Radius ~2.75 km dari pusat kota per 1 std dev.

    Args:
        city: Nama kota (kunci CITY_COORD_ANCHORS).
        rng : NumPy Generator untuk hasil reproducible (dari seed tetap).

    Returns:
        Tuple (latitude, longitude) presisi 6 desimal.
    """
    anchor = CITY_COORD_ANCHORS.get(city, CITY_COORD_ANCHORS["Jakarta Pusat"])
    lat = round(anchor["lat"] + rng.normal(0, GPS_SCATTER_STD), 6)
    lon = round(anchor["lon"] + rng.normal(0, GPS_SCATTER_STD), 6)
    return lat, lon


def augment_dataset(
    input_path: str,
    output_path: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Fungsi utama augmentasi dataset kos-kosan Jakarta.

    Args:
        input_path  : Path ke CSV dataset asli (data_kos_jakarta.csv).
        output_path : Path output CSV yang sudah diaugmentasi.
        random_seed : Seed untuk reproducibility (default: 42).

    Returns:
        DataFrame yang sudah diaugmentasi dengan 5 kolom baru.

    Raises:
        FileNotFoundError: Jika file input tidak ditemukan.
        ValueError        : Jika kolom wajib tidak ada di dataset.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {input_path}\n"
            "Pastikan data_kos_jakarta.csv sudah ditaruh di data/raw/"
        )

    logger.info(f"Membaca dataset: {input_path}")
    df: pd.DataFrame = pd.read_csv(input_path)

    # Validasi kolom wajib
    required_cols = ["id_kos", "kota", "wilayah", "tipe_kos", "ukuran_kamar"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di dataset: {missing}")

    logger.info(f"Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom.")

    rng = np.random.default_rng(random_seed)
    durasi_opts = ["2", "3", "6", "8", "12", "18", "24"]

    deskripsi_list: list[str] = []
    ulasan_list: list[str] = []
    foto_list: list[str] = []
    lat_list: list[float] = []
    lon_list: list[float] = []

    for idx, row in df.iterrows():
        # --- 1. Koordinat GPS (Gaussian noise di sekitar pusat kota) ---
        lat, lon = _generate_gps(str(row.get("kota", "Jakarta Pusat")), rng)
        lat_list.append(lat)
        lon_list.append(lon)

        # --- 2. Path foto kamar ---
        foto_list.append(
            f"assets/kos_photos/kos_{int(row.get('id_kos', idx + 1)):04d}.jpg"
        )

        # --- 3. Deskripsi promosi (pilih template acak, isi dengan data baris) ---
        tmpl_idx = rng.integers(0, len(DESKRIPSI_TEMPLATES))
        deskripsi_list.append(
            DESKRIPSI_TEMPLATES[tmpl_idx].format(
                tipe=str(row.get("tipe_kos", "Campur")).lower(),
                wilayah=str(row.get("wilayah", "Jakarta")),
                kota=str(row.get("kota", "Jakarta")),
                ukuran=str(row.get("ukuran_kamar", 9)),
                fasilitas_str=_build_fasilitas_str(row),
                jarak_kampus=str(
                    round(float(row.get("jarak_ke_kampus_km", 2.0)), 1)
                ),
                jarak_transport=str(
                    round(float(row.get("jarak_ke_transportasi_km", 1.0)), 1)
                ),
            )
        )

        # --- 4. Ulasan penghuni (positif jika rating >= 4.0, netral jika di bawah) ---
        rating = float(row.get("rating", 3.5))
        templates = ULASAN_POSITIF if rating >= 4.0 else ULASAN_NETRAL
        ulasan_tmpl = templates[rng.integers(0, len(templates))]
        ulasan_list.append(
            ulasan_tmpl.format(
                durasi=str(rng.choice(durasi_opts)),
                catatan=_build_catatan_fasilitas(row),
            )
        )

    # Tambahkan 5 kolom baru ke DataFrame
    df["deskripsi_promosi"] = deskripsi_list
    df["ulasan_penghuni"] = ulasan_list
    df["foto_path"] = foto_list
    df["latitude"] = lat_list
    df["longitude"] = lon_list

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"✅ Augmentasi selesai. Output: {output_path}")
    logger.info(f"   Baris: {len(df)} | Kolom: {len(df.columns)}")
    logger.info(
        f"   Kolom baru: deskripsi_promosi, ulasan_penghuni, "
        f"foto_path, latitude, longitude"
    )
    return df


if __name__ == "__main__":
    augment_dataset(
        input_path=settings.data_raw_path,
        output_path=settings.data_augmented_path,
    )
```

---

## FILE 11: `scripts/preprocess_dataset.py`

```python
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
```

---

## FILE 12: `migrations/001_create_kos_table.sql`

```sql
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
```

---

## FILE 13: `scripts/init_postgis.py`

```python
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
```

---

## FILE 14: `scripts/build_faiss_index.py`

```python
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
```

---

*End of TAHAP 1 — Fondasi Data Engineering & Dual Database Architecture selesai.*
*State yang diingat untuk Tahap 2: FAISS interface (build_index/search/load_index),*
*kolom deskripsi_clean sebagai input vectorizer, id_kos sebagai primary key linker,*
*DATABASE_DSN untuk koneksi sync admin scripts.*
