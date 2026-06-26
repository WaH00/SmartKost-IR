# TAHAP 3 — Smart-Kos Hybrid Engine
## FastAPI Bridge + Flutter Mobile Integration

> Melanjutkan state dari:
> - TAHAP1: Database (PostgreSQL + PostGIS), Sastrawi Pipeline, FAISS TF-IDF placeholder
> - TAHAP2: IndoBERT Embedder, FAISS IndexFlatIP, PRF Engine, RF PricingService,
>           FusionScorer, SearchService (11-step pipeline), SearchResponse dataclass

---

### File Baru & Diperbarui di Tahap 3

```
smart-kos-backend/
├── app/
│   ├── main.py                           ← UPDATE: lifespan loader + CORS (siap Tahap 3)
│   ├── core/
│   │   └── config.py                     ← UPDATE: tambah field baru Tahap 3
│   └── api/
│       └── v1/
│           ├── router.py                  ← NEW: agregator semua sub-router
│           ├── schemas/
│           │   ├── __init__.py            ← NEW
│           │   ├── search.py              ← NEW: Pydantic Request + Response
│           │   └── common.py             ← NEW: shared schemas (error, pagination)
│           └── endpoints/
│               ├── __init__.py            ← NEW
│               ├── search.py              ← NEW: POST /search-kos
│               └── health.py             ← NEW: GET /health (extended)
│
smart-kos-flutter/                         ← PROJECT FLUTTER (baru)
├── lib/
│   ├── main.dart                          ← NEW: entry point + MaterialApp
│   ├── core/
│   │   ├── api_client.dart               ← NEW: Dio singleton + interceptors
│   │   └── constants.dart               ← NEW: URL, timeouts, config
│   ├── models/
│   │   └── kos_result.dart              ← NEW: Dart model dari JSON response
│   ├── services/
│   │   └── search_service.dart          ← NEW: layer HTTP → Dart model
│   ├── providers/
│   │   └── search_provider.dart         ← NEW: ChangeNotifier state management
│   └── screens/
│       ├── search_screen.dart           ← NEW: UI pencarian + GPS
│       └── widgets/
│           ├── kos_card.dart            ← NEW: card kos dengan badge Super Deal
│           ├── super_deal_badge.dart    ← NEW: badge 🔥 animated
│           └── score_bars.dart         ← NEW: visualisasi skor komponen
│
├── pubspec.yaml                          ← NEW: dependencies Flutter
└── test/
    └── widget_test.dart                 ← NEW: stress test simulator
```

---

### Tambahan `app/core/config.py` (field Tahap 3)

```python
# Tambahkan ke class Settings yang ada:

    # === API Server ===
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # CORS: ganti "*" dengan IP Flutter emulator di production
    # Emulator Android mengakses host via 10.0.2.2 (bukan localhost)
    cors_origins: list[str] = ["*"]

    # === Search defaults ===
    default_top_k: int = 10
    max_top_k: int = 50
    default_n_candidates: int = 20
```

---

## BACKEND — FastAPI

---

## FILE 1: `app/api/v1/schemas/common.py`

```python
"""
app/api/v1/schemas/common.py — Shared Pydantic Schemas
Smart-Kos Hybrid Engine — Tahap 3

Skema yang digunakan bersama oleh lebih dari satu endpoint.
Menjaga DRY (Don't Repeat Yourself) di seluruh layer API.
"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Format error standar Smart-Kos API.
    Seluruh HTTPException menggunakan format ini agar Flutter
    dapat mem-parse error secara konsisten tanpa switch-case tipe.
    """
    status: str = Field("error", description="Selalu 'error' untuk respons gagal")
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Deskripsi error yang human-readable")
    detail: str | None = Field(None, description="Detail teknis (opsional)")

    model_config = {"json_schema_extra": {
        "example": {
            "status": "error",
            "code": 422,
            "message": "Parameter kueri tidak valid",
            "detail": "Field 'latitude' wajib diisi"
        }
    }}


class HealthStatus(BaseModel):
    """Respons endpoint GET /health."""
    status: str
    version: str
    components: dict[str, str]
    uptime_seconds: float | None = None
```

---

## FILE 2: `app/api/v1/schemas/search.py`

```python
"""
app/api/v1/schemas/search.py — Search Request & Response Schemas
Smart-Kos Hybrid Engine — Tahap 3

Kontrak data yang mengikat Flutter (client) dan FastAPI (server).
Setiap perubahan di sini = perubahan di Flutter model, dan sebaliknya.

Prinsip desain:
  - Response schema harus serializeable langsung dari KosScoringResult (Tahap 2)
  - Flutter hanya perlu field yang relevan untuk UI — tidak semua field backend
  - Setiap field numerik memiliki batas min/max untuk mencegah abuse request
"""

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class SearchKosRequest(BaseModel):
    """
    Body POST /api/v1/search-kos — dikirim oleh Flutter.

    Validasi otomatis dilakukan Pydantic sebelum menyentuh service layer:
    - kueri tidak boleh kosong setelah strip
    - latitude: -90 s/d +90 (valid WGS84)
    - longitude: -180 s/d +180 (valid WGS84)
    - top_k: batas atas 50 agar tidak membebani inference IndoBERT
    """
    kueri: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Teks pencarian kos dari user Flutter",
        examples=["kos dekat kampus wifi AC murah"],
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude GPS perangkat user (WGS84 / EPSG:4326)",
        examples=[-6.2088],
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude GPS perangkat user (WGS84 / EPSG:4326)",
        examples=[106.8456],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Jumlah hasil yang dikembalikan (1–50, default: 10)",
    )
    n_candidates: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Pool kandidat dari FAISS Re-rank sebelum top_k filter",
    )

    @field_validator("kueri", mode="before")
    @classmethod
    def strip_kueri(cls, v: str) -> str:
        """Normalisasi spasi berlebih di awal/akhir kueri."""
        stripped = str(v).strip()
        if not stripped:
            raise ValueError("Kueri tidak boleh berisi hanya spasi.")
        return stripped

    @model_validator(mode="after")
    def validate_top_k_lte_candidates(self) -> "SearchKosRequest":
        """
        top_k tidak boleh melebihi n_candidates.
        Jika dilanggar, clamp top_k ke n_candidates secara senyap (no error)
        agar user tidak mendapat 422 untuk kasus ini.
        """
        if self.top_k > self.n_candidates:
            self.top_k = self.n_candidates
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "kueri": "kos putri dekat kampus wifi AC kamar mandi dalam",
            "latitude": -6.1862,
            "longitude": 106.8348,
            "top_k": 10,
            "n_candidates": 20,
        }
    }}


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class FasilitasSchema(BaseModel):
    """Sub-schema fasilitas kos (binary flags)."""
    ac: bool
    kamar_mandi_dalam: bool
    wifi: bool
    listrik_include: bool
    parkir: bool
    dapur: bool
    laundry: bool
    security_24jam: bool


class KosResultItem(BaseModel):
    """
    Satu item kos dalam response list — diterima oleh Flutter dan dirender
    sebagai KosCard widget. Setiap field memiliki deskripsi eksplisit
    untuk dokumentasi Swagger yang lengkap.

    Korespondensi dengan KosScoringResult (Tahap 2):
        Semua field diambil langsung dari dataclass KosScoringResult.
        Flutter hanya perlu field ini — field internal pipeline (raw scores)
        diekspose untuk kepentingan riset & transparansi di Bab 4.
    """

    # Identitas
    id_kos: int = Field(..., description="ID unik kos dari database")
    nama_kos: str = Field(..., description="Nama lengkap kos")
    kota: str = Field(..., description="Kota (Jakarta Pusat/Selatan/Timur/Barat/Utara)")
    wilayah: str = Field(..., description="Wilayah/kecamatan kos")
    foto_path: str = Field(..., description="Path relatif file foto kamar")

    # Harga
    harga_per_bulan: int = Field(..., description="Harga sewa aktual (Rp/bulan)")
    predicted_price: float = Field(
        ..., description="Harga wajar prediksi Random Forest (Rp/bulan)"
    )

    # Lokasi
    distance_km: float = Field(
        ..., description="Jarak Haversine dari GPS user ke kos (km)"
    )

    # Skor komponen (untuk transparansi riset + Bab 4 laporan)
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fusion Score akhir [0–1]: 0.5×sem + 0.3×geo + 0.2×price_boost",
    )
    semantic_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity IndoBERT [0–1]"
    )
    geospatial_score: float = Field(
        ..., ge=0.0, le=1.0, description="Skor jarak via exp decay [0–1]"
    )
    price_boost: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Boost harga: 1.0=super_deal, 0.5=underpriced, 0.0=fair/overpriced",
    )

    # Label harga
    price_label: str = Field(
        ...,
        description="Label kelayakan harga: super_deal|underpriced|fair|overpriced",
    )
    is_super_deal: bool = Field(
        ...,
        description="True jika harga aktual < 85% dari prediksi RF → badge 🔥 di Flutter",
    )

    # Fasilitas
    rating: float = Field(..., description="Rating rata-rata dari penghuni (0–5)")
    fasilitas: FasilitasSchema = Field(..., description="Detail fasilitas kos")

    @classmethod
    def from_scoring_result(cls, r: object) -> "KosResultItem":
        """
        Factory method: konversi KosScoringResult (Tahap 2 dataclass)
        → KosResultItem (Pydantic schema).

        Memisahkan kolom fasilitas individual menjadi sub-object FasilitasSchema
        untuk struktur JSON yang lebih bersih di Flutter.
        """
        return cls(
            id_kos=r.id_kos,
            nama_kos=r.nama_kos,
            kota=r.kota,
            wilayah=r.wilayah,
            foto_path=r.foto_path,
            harga_per_bulan=r.harga_per_bulan,
            predicted_price=r.predicted_price,
            distance_km=round(r.distance_km, 3),
            final_score=round(r.final_score, 6),
            semantic_score=round(r.semantic_score, 6),
            geospatial_score=round(r.geospatial_score, 6),
            price_boost=r.price_boost,
            price_label=r.price_label,
            is_super_deal=r.is_super_deal,
            rating=r.rating,
            fasilitas=FasilitasSchema(
                ac=bool(r.ac),
                kamar_mandi_dalam=bool(r.kamar_mandi_dalam),
                wifi=bool(r.wifi),
                listrik_include=bool(r.listrik_include),
                parkir=bool(r.parkir),
                dapur=bool(r.dapur),
                laundry=bool(r.laundry),
                security_24jam=bool(r.security_24jam),
            ),
        )


class SearchKosResponse(BaseModel):
    """
    Respons lengkap POST /api/v1/search-kos — diterima oleh Flutter.

    Selain daftar kos, menyertakan metadata pipeline untuk:
    1. Transparansi akademis (laporan Bab 4: bagaimana PRF bekerja)
    2. Debugging di development (cek expansion_terms, search_time_ms)
    3. UX Flutter (tampilkan "Pencarian diperluas: +kulkas +pantry" di UI)
    """

    status: str = Field("success", description="'success' jika request berhasil")
    # Metadata pipeline (untuk riset + debugging)
    query_original: str = Field(..., description="Kueri asli dari user")
    query_preprocessed: str = Field(..., description="Kueri setelah Sastrawi pipeline")
    query_expanded: str = Field(..., description="Kueri setelah PRF expansion")
    expansion_terms: list[str] = Field(
        ..., description="Term baru yang ditambahkan PRF"
    )
    total_candidates: int = Field(..., description="Jumlah kandidat sebelum filter top_k")
    search_time_ms: float = Field(..., description="Latency end-to-end (ms)")
    # Hasil
    total_results: int = Field(..., description="Jumlah kos dalam list results")
    super_deal_count: int = Field(..., description="Jumlah kos dengan label Super Deal")
    results: list[KosResultItem] = Field(..., description="Daftar kos terurut by final_score")

    model_config = {"json_schema_extra": {
        "example": {
            "status": "success",
            "query_original": "kos putri dekat kampus wifi",
            "query_preprocessed": "kos putri dekat kampus wifi",
            "query_expanded": "kos putri dekat kampus wifi kamar bersih nyaman",
            "expansion_terms": ["kamar", "bersih", "nyaman"],
            "total_candidates": 20,
            "search_time_ms": 187.4,
            "total_results": 10,
            "super_deal_count": 2,
            "results": []
        }
    }}
```

---

## FILE 3: `app/api/v1/endpoints/health.py`

```python
"""
app/api/v1/endpoints/health.py — Extended Health Check Endpoint
Smart-Kos Hybrid Engine — Tahap 3

GET /api/v1/health → Laporan status semua komponen kritis.
Digunakan oleh Flutter untuk cek koneksi sebelum query pertama.
"""

import time
import logging
from fastapi import APIRouter, Request

from app.api.v1.schemas.common import HealthStatus
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Catat waktu startup untuk kalkulasi uptime
_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Status semua komponen Smart-Kos",
    tags=["System"],
)
async def health_check(request: Request) -> HealthStatus:
    """
    Cek status komponen:
        - FAISS IndoBERT index (loaded / not_loaded)
        - Random Forest pricing model (loaded / not_loaded)
        - Database connection (dikembangkan di Tahap 4)

    Flutter memanggil endpoint ini saat aplikasi pertama kali dibuka
    untuk memastikan backend aktif sebelum user melakukan pencarian.
    """
    components: dict[str, str] = {
        "database": "postgresql+postgis",
    }

    # Cek FAISS indexer dari app.state
    search_service = getattr(request.app.state, "search_service", None)
    if search_service and getattr(search_service.indexer, "index", None) is not None:
        n_vectors = search_service.indexer.index.ntotal
        components["faiss_index"] = f"loaded ({n_vectors} vectors, IndoBERT 768-dim)"
    else:
        components["faiss_index"] = "not_loaded"

    # Cek RF pricing model
    if search_service and getattr(search_service.pricing_service, "_model", None):
        components["rf_pricing_model"] = "loaded (RandomForest, R²=0.87)"
    else:
        components["rf_pricing_model"] = "not_loaded"

    # Cek IndoBERT embedder device
    if search_service and search_service.embedder:
        components["indobert"] = (
            f"loaded (device={search_service.embedder.device})"
        )
    else:
        components["indobert"] = "not_loaded"

    overall_status = (
        "healthy"
        if all(v != "not_loaded" for v in components.values())
        else "degraded"
    )

    return HealthStatus(
        status=overall_status,
        version=settings.app_version,
        components=components,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
```

---

## FILE 4: `app/api/v1/endpoints/search.py`

```python
"""
app/api/v1/endpoints/search.py — Core Search Endpoint
Smart-Kos Hybrid Engine — Tahap 3

POST /api/v1/search-kos
Titik masuk tunggal untuk seluruh 11-langkah hybrid search pipeline.
Endpoint ini tipis (thin controller): validasi → delegate ke service → serialize.
Seluruh logika bisnis tetap di SearchService (Tahap 2) — tidak ada kebocoran.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.search import (
    KosResultItem,
    SearchKosRequest,
    SearchKosResponse,
)
from app.core.database import get_db_session

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/search-kos",
    response_model=SearchKosResponse,
    status_code=status.HTTP_200_OK,
    summary="Pencarian kos hybrid: Semantic + Geospatial + Pricing",
    description=(
        "Endpoint utama Smart-Kos Hybrid Engine.\n\n"
        "**Pipeline 11 langkah** dieksekusi setiap request:\n"
        "1. Sastrawi preprocessing kueri\n"
        "2. IndoBERT embedding → vektor 768-dim\n"
        "3. FAISS Search Rd.1 → top-5 feedback docs untuk PRF\n"
        "4. Fetch `deskripsi_clean` dari PostgreSQL\n"
        "5. PRF TF-IDF: ekstrak term dominan → expand kueri\n"
        "6. IndoBERT re-embed kueri diperluas\n"
        "7. FAISS Search Rd.2 (re-rank) → top-N kandidat\n"
        "8. PostGIS: fetch data kos + hitung jarak Haversine\n"
        "9. RF batch price prediction → label Super Deal\n"
        "10. Fusion scoring: 0.50×Semantic + 0.30×Geo + 0.20×Price\n"
        "11. Sort descending → return top_k\n\n"
        "Kos dengan `is_super_deal=true` ditampilkan dengan badge 🔥 di Flutter."
    ),
    tags=["Search"],
    responses={
        200: {"description": "Pencarian berhasil"},
        422: {"description": "Parameter request tidak valid"},
        503: {"description": "Komponen AI belum siap (FAISS / IndoBERT)"},
        500: {"description": "Internal server error"},
    },
)
async def search_kos(
    body: SearchKosRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> SearchKosResponse:
    """
    Handler endpoint pencarian kos.

    Pattern: Thin Controller
        - Validasi dilakukan Pydantic (otomatis, sebelum fungsi ini dipanggil)
        - Semua logika bisnis ada di SearchService.search()
        - Handler hanya: ambil service → panggil search → serialize response

    Args:
        body   : SearchKosRequest (divalidasi otomatis Pydantic)
        request: FastAPI Request (untuk akses app.state)
        db     : AsyncSession PostgreSQL (injeksi via Depends)

    Raises:
        HTTP 503: Jika SearchService belum dimuat di app.state
                  (jalankan `python scripts/rebuild_faiss_indobert.py` dulu)
        HTTP 500: Jika terjadi error tidak terduga di pipeline
    """
    # Ambil SearchService dari app.state (dimuat saat lifespan startup)
    search_service = getattr(request.app.state, "search_service", None)
    if search_service is None:
        logger.error(
            "SearchService tidak ditemukan di app.state. "
            "Pastikan lifespan startup handler sudah memuat semua komponen."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Layanan pencarian belum siap.",
                "hint": (
                    "1. Pastikan scripts/rebuild_faiss_indobert.py sudah dijalankan. "
                    "2. Restart server dengan: uvicorn app.main:app --reload"
                ),
            },
        )

    try:
        logger.info(
            f"Search request: kueri='{body.kueri}' | "
            f"loc=({body.latitude:.4f},{body.longitude:.4f}) | "
            f"top_k={body.top_k}"
        )

        # Delegasikan ke pipeline SearchService (11 langkah, Tahap 2)
        search_response = await search_service.search(
            query=body.kueri,
            user_lat=body.latitude,
            user_lon=body.longitude,
            top_k=body.top_k,
            n_candidates=body.n_candidates,
            db=db,
        )

        # Serialisasi KosScoringResult → KosResultItem (Pydantic)
        result_items = [
            KosResultItem.from_scoring_result(r)
            for r in search_response.results
        ]

        super_deal_count = sum(1 for r in result_items if r.is_super_deal)

        logger.info(
            f"Search selesai: {len(result_items)} hasil | "
            f"Super Deal: {super_deal_count} | "
            f"PRF terms: {search_response.expansion_terms} | "
            f"Latency: {search_response.search_time_ms}ms"
        )

        return SearchKosResponse(
            status="success",
            query_original=search_response.query_original,
            query_preprocessed=search_response.query_preprocessed,
            query_expanded=search_response.query_expanded,
            expansion_terms=search_response.expansion_terms,
            total_candidates=search_response.total_candidates_retrieved,
            search_time_ms=search_response.search_time_ms,
            total_results=len(result_items),
            super_deal_count=super_deal_count,
            results=result_items,
        )

    except HTTPException:
        raise  # Re-raise HTTPException yang sudah terstruktur
    except Exception as exc:
        logger.exception(f"Error pada pipeline pencarian: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Terjadi kesalahan internal saat memproses pencarian.",
                "hint": "Cek log server untuk detail error.",
            },
        )
```

---

## FILE 5: `app/api/v1/router.py`

```python
"""
app/api/v1/router.py — API v1 Route Aggregator
Smart-Kos Hybrid Engine — Tahap 3

Mengumpulkan semua sub-router endpoint ke satu APIRouter v1.
app/main.py hanya perlu include router ini — tidak perlu tahu
struktur internal endpoint.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health, search

api_v1_router = APIRouter()

# Daftarkan semua endpoint group
api_v1_router.include_router(health.router)
api_v1_router.include_router(search.router)
```

---

## FILE 6: `app/main.py` (UPDATED — Tahap 3)

```python
"""
app/main.py — FastAPI Application Entry Point
Smart-Kos Hybrid Engine — Tahap 3 (Final Backend Version)

Menggabungkan:
  - Lifespan: load semua komponen berat (IndoBERT, FAISS, RF) SEKALI saat startup
  - CORS: izinkan Flutter (emulator Android: 10.0.2.2, device fisik: IP LAN)
  - Router: include api_v1_router (health + search endpoints)
  - Error handlers: format error konsisten ke Flutter
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manajemen siklus hidup FastAPI.
    Memuat semua komponen berat ke app.state SEKALI saat startup.

    Keuntungan pola ini vs import langsung di endpoint:
    - Model IndoBERT (~438MB) dan FAISS index hanya dimuat sekali
    - Seluruh request berbagi instance yang sama (hemat memori ~10x)
    - Startup time transparan di log — terlihat oleh DevOps/dosen
    """
    startup_start = time.perf_counter()

    # ── Komponen imports (di sini agar tidak eager-load saat module import) ──
    from app.services.pricing_service import PricingService
    from app.services.search_service import SearchService
    from stki.embedder import IndoBERTEmbedder
    from stki.fusion_scorer import FusionScorer
    from stki.indexer import KosVectorIndexer
    from stki.preprocessor import IndonesianTextPreprocessor
    from stki.prf_engine import PseudoRelevanceFeedback

    logger.info("=" * 60)
    logger.info(f"🚀 STARTUP: {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)

    # ── 1. IndoBERT Embedder ──────────────────────────────────────────────
    logger.info("[1/5] Memuat IndoBERT Embedder...")
    t = time.perf_counter()
    embedder = IndoBERTEmbedder(
        model_name=settings.indobert_model_name,
        device="auto",
        max_length=settings.indobert_max_length,
    )
    logger.info(f"      IndoBERT siap ({time.perf_counter() - t:.1f}s) | Device: {embedder.device}")

    # ── 2. FAISS IndoBERT Index ───────────────────────────────────────────
    logger.info("[2/5] Memuat FAISS IndoBERT Index...")
    t = time.perf_counter()
    indexer = KosVectorIndexer(
        index_path=settings.faiss_indobert_index_path,
        id_map_path=settings.faiss_indobert_id_map_path,
        embedder=embedder,
    )
    try:
        indexer.load_index()
        logger.info(
            f"      FAISS siap ({time.perf_counter() - t:.1f}s) | "
            f"{indexer.index.ntotal} vectors × 768-dim"
        )
    except FileNotFoundError as e:
        logger.error(f"      ❌ FAISS index tidak ditemukan: {e}")
        logger.error("      Solusi: python scripts/rebuild_faiss_indobert.py")
        # Lanjutkan startup (degraded mode) agar /health tetap bisa diakses
        # Endpoint /search-kos akan return 503 sampai index tersedia

    # ── 3. RF Pricing Service ─────────────────────────────────────────────
    logger.info("[3/5] Memuat Random Forest Pricing Service...")
    t = time.perf_counter()
    try:
        pricing_service = PricingService(
            model_path=settings.model_rf_path,
            scaler_path=settings.scaler_path,
        )
        logger.info(f"      RF Pricing siap ({time.perf_counter() - t:.1f}s)")
    except FileNotFoundError as e:
        logger.error(f"      ❌ Model RF tidak ditemukan: {e}")
        pricing_service = None

    # ── 4. Inisialisasi komponen ringan ───────────────────────────────────
    logger.info("[4/5] Inisialisasi PRF Engine, FusionScorer, Preprocessor...")
    prf_engine = PseudoRelevanceFeedback(
        n_feedback_docs=settings.prf_n_feedback_docs,
        n_expansion_terms=settings.prf_n_expansion_terms,
    )
    fusion_scorer = FusionScorer(
        w_semantic=settings.fusion_weight_semantic,
        w_geospatial=settings.fusion_weight_geospatial,
        w_price=settings.fusion_weight_price,
        geo_decay_lambda=settings.fusion_geo_decay_lambda,
    )
    preprocessor = IndonesianTextPreprocessor()
    logger.info("      Komponen ringan siap.")

    # ── 5. Assembling SearchService ───────────────────────────────────────
    logger.info("[5/5] Merakit SearchService (pipeline 11 langkah)...")
    if pricing_service:
        app.state.search_service = SearchService(
            embedder=embedder,
            indexer=indexer,
            prf_engine=prf_engine,
            pricing_service=pricing_service,
            fusion_scorer=fusion_scorer,
            preprocessor=preprocessor,
        )
        total_startup = time.perf_counter() - startup_start
        logger.info("=" * 60)
        logger.info(f"✅ STARTUP SELESAI dalam {total_startup:.2f}s")
        logger.info(f"   Docs : http://{settings.api_host}:{settings.api_port}/docs")
        logger.info(f"   Health: http://{settings.api_host}:{settings.api_port}/api/v1/health")
        logger.info("=" * 60)
    else:
        app.state.search_service = None
        logger.warning("⚠  SearchService TIDAK dimuat (model RF hilang). Mode degraded.")

    yield  # ← Aplikasi aktif di antara startup dan shutdown

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info("🛑 Menghentikan Smart-Kos server...")
    app.state.search_service = None
    logger.info("Cleanup selesai. Server offline.")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP INSTANCE
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "## Smart-Kos Hybrid Engine API\n\n"
        "Platform PropTech 4 Pilar Kecerdasan untuk pencarian kos-kosan di Jakarta.\n\n"
        "| Pilar | Teknologi | Fungsi |\n"
        "|-------|-----------|--------|\n"
        "| Semantic IR | IndoBERT + FAISS | Memahami makna teks |\n"
        "| Auto-Correction | TF-IDF PRF | Ekspansi kueri otomatis |\n"
        "| Geospatial | PostGIS ST_Distance | Jarak Haversine geodesik |\n"
        "| Predictive Pricing | Random Forest | Label Super Deal |\n\n"
        "**Novelty**: Hybrid Fusion Score = 0.50×Semantic + 0.30×Geo + 0.20×Price"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi CORS wajib untuk Flutter:
#   - Emulator Android mengakses backend via 10.0.2.2 (bukan localhost/127.0.0.1)
#   - Device fisik menggunakan IP LAN: mis. 192.168.1.x
#   - Development: allow_origins=["*"] praktis tapi kurang aman
#   - Production (Tahap 4): ganti dengan domain spesifik
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,      # ["*"] untuk development
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # OPTIONS wajib untuk CORS preflight
    allow_headers=["*"],
    max_age=3600,  # Cache preflight response 1 jam (kurangi latency OPTIONS)
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTION HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Override default Pydantic validation error format.
    Default FastAPI mengembalikan struktur nested yang sulit di-parse Flutter.
    Format ini lebih mudah ditangani di Dart:
        {"status":"error","code":422,"message":"...","detail":"..."}
    """
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    field = " → ".join(str(loc) for loc in first_error.get("loc", ["unknown"]))
    msg = first_error.get("msg", "Validation error")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "code": 422,
            "message": f"Parameter tidak valid: {field}",
            "detail": msg,
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "code": 404,
            "message": f"Endpoint tidak ditemukan: {request.url.path}",
            "detail": "Cek dokumentasi API di /docs",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER ROUTERS
# ─────────────────────────────────────────────────────────────────────────────
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

# Root endpoint — konfirmasi server aktif (diakses Flutter saat app dibuka)
@app.get("/", tags=["Root"], include_in_schema=False)
async def root() -> dict:
    return {
        "status": "online",
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
        "search": f"{settings.api_v1_prefix}/search-kos",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        # workers=2  # Uncomment di production — tapi JANGAN di development
        # (reload=True tidak kompatibel dengan workers > 1)
    )
```

---

## FRONTEND — Flutter + Dart

---

## FILE 7: `smart-kos-flutter/pubspec.yaml`

```yaml
name: smart_kos
description: Smart-Kos Hybrid Engine — Flutter Mobile App
version: 1.0.0+1
publish_to: 'none'

environment:
  sdk: '>=3.0.0 <4.0.0'
  flutter: ">=3.10.0"

dependencies:
  flutter:
    sdk: flutter

  # HTTP Client — Dio: lebih powerful dari package http (interceptors, retry, logging)
  dio: ^5.4.3

  # GPS / Location
  geolocator: ^11.0.0

  # State Management — simple, no boilerplate
  provider: ^6.1.2

  # UI Utilities
  cached_network_image: ^3.3.1   # Cache foto kos dari URL
  shimmer: ^3.0.0                # Loading skeleton animation
  flutter_animate: ^4.5.0        # Badge animation (Super Deal pulse effect)

  # Format angka Rupiah
  intl: ^0.19.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0

flutter:
  uses-material-design: true
  assets:
    - assets/images/
    - assets/icons/
```

---

## FILE 8: `smart-kos-flutter/lib/core/constants.dart`

```dart
// lib/core/constants.dart
// Konfigurasi terpusat Smart-Kos Flutter App.
// Semua URL dan konstanta ada di sini — tidak ada hardcode di widget.

class AppConstants {
  AppConstants._(); // Prevent instantiation

  // ── API Base URL ──────────────────────────────────────────────────────────
  // PENTING: Emulator Android mengakses localhost host BUKAN 127.0.0.1 melainkan
  // 10.0.2.2. Ini adalah IP gateway dari emulator ke host machine.
  //
  // Pilih satu sesuai environment:
  static const String _baseUrlEmulator = 'http://10.0.2.2:8000';
  static const String _baseUrlDevice = 'http://192.168.1.100:8000'; // Ganti IP LAN-mu
  static const String _baseUrlProduction = 'https://api.smart-kos.id'; // Tahap 4

  // AKTIFKAN salah satu:
  static const String baseUrl = _baseUrlEmulator;

  // ── Endpoint Paths ────────────────────────────────────────────────────────
  static const String apiV1 = '/api/v1';
  static const String endpointSearch = '$apiV1/search-kos';
  static const String endpointHealth = '$apiV1/health';

  // ── Timeout Configuration ─────────────────────────────────────────────────
  // IndoBERT encoding CPU ~100-200ms + FAISS + PostGIS + RF = ~300-600ms
  // Set connectTimeout lebih singkat dari receiveTimeout
  static const Duration connectTimeout = Duration(seconds: 10);
  static const Duration receiveTimeout = Duration(seconds: 30);

  // ── Search Defaults ───────────────────────────────────────────────────────
  static const int defaultTopK = 10;
  static const int defaultNCandidates = 20;

  // ── Super Deal Threshold (tampilan) ──────────────────────────────────────
  // Harga aktual < 85% dari prediksi RF → badge Super Deal
  // Nilai ini HANYA untuk display kalkulasi di UI — keputusan asli ada di backend
  static const double superDealThreshold = 0.85;

  // ── Warna Super Deal Badge ────────────────────────────────────────────────
  static const int superDealColorValue = 0xFFD97706; // Amber-600
  static const int superDealBgColorValue = 0xFFFEF3C7; // Amber-100
}
```

---

## FILE 9: `smart-kos-flutter/lib/core/api_client.dart`

```dart
// lib/core/api_client.dart
// Singleton Dio HTTP Client dengan logging interceptor.
// Dipakai oleh seluruh service layer — tidak ada Dio instance lain.

import 'package:dio/dio.dart';
import 'constants.dart';

class ApiClient {
  ApiClient._internal();
  static final ApiClient _instance = ApiClient._internal();
  factory ApiClient() => _instance;

  late final Dio _dio = _buildDio();

  Dio get dio => _dio;

  Dio _buildDio() {
    final dio = Dio(
      BaseOptions(
        baseUrl: AppConstants.baseUrl,
        connectTimeout: AppConstants.connectTimeout,
        receiveTimeout: AppConstants.receiveTimeout,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'X-App-Version': '1.0.0',  // Dikirim ke backend untuk analytics
        },
      ),
    );

    // Interceptor: log request + response (aktif hanya di debug mode)
    dio.interceptors.add(
      LogInterceptor(
        requestBody: true,
        responseBody: false, // Response body besar — jangan log di production
        logPrint: (obj) => print('[DIO] $obj'),
      ),
    );

    // Interceptor: tambah timing header ke setiap request
    dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) {
          options.extra['requestTime'] = DateTime.now().millisecondsSinceEpoch;
          handler.next(options);
        },
        onResponse: (response, handler) {
          final requestTime = response.requestOptions.extra['requestTime'] as int?;
          if (requestTime != null) {
            final elapsed = DateTime.now().millisecondsSinceEpoch - requestTime;
            print('[API] ${response.requestOptions.path} → ${response.statusCode} | ${elapsed}ms');
          }
          handler.next(response);
        },
        onError: (error, handler) {
          print('[API ERROR] ${error.requestOptions.path} → ${error.response?.statusCode}: ${error.message}');
          handler.next(error);
        },
      ),
    );

    return dio;
  }
}
```

---

## FILE 10: `smart-kos-flutter/lib/models/kos_result.dart`

```dart
// lib/models/kos_result.dart
// Dart model yang merepresentasikan satu item kos dari JSON response API.
// Korespondensi 1:1 dengan KosResultItem (Pydantic schema, backend Tahap 3).

class FasilitasModel {
  final bool ac;
  final bool kamarMandiDalam;
  final bool wifi;
  final bool listrikInclude;
  final bool parkir;
  final bool dapur;
  final bool laundry;
  final bool security24Jam;

  const FasilitasModel({
    required this.ac,
    required this.kamarMandiDalam,
    required this.wifi,
    required this.listrikInclude,
    required this.parkir,
    required this.dapur,
    required this.laundry,
    required this.security24Jam,
  });

  factory FasilitasModel.fromJson(Map<String, dynamic> json) => FasilitasModel(
    ac: json['ac'] as bool? ?? false,
    kamarMandiDalam: json['kamar_mandi_dalam'] as bool? ?? false,
    wifi: json['wifi'] as bool? ?? false,
    listrikInclude: json['listrik_include'] as bool? ?? false,
    parkir: json['parkir'] as bool? ?? false,
    dapur: json['dapur'] as bool? ?? false,
    laundry: json['laundry'] as bool? ?? false,
    security24Jam: json['security_24jam'] as bool? ?? false,
  );

  /// Jumlah fasilitas yang tersedia (untuk sorting tampilan ikon)
  int get totalFasilitas =>
    [ac, kamarMandiDalam, wifi, listrikInclude, parkir, dapur, laundry, security24Jam]
        .where((f) => f).length;
}


class KosResultModel {
  // Identitas
  final int idKos;
  final String namaKos;
  final String kota;
  final String wilayah;
  final String fotoPath;

  // Harga
  final int hargaPerBulan;
  final double predictedPrice;

  // Lokasi
  final double distanceKm;

  // Skor (untuk tampilan detail dan laporan TA)
  final double finalScore;
  final double semanticScore;
  final double geospatialScore;
  final double priceBoost;

  // Label harga
  final String priceLabel;
  final bool isSuperDeal;

  // Metadata
  final double rating;
  final FasilitasModel fasilitas;

  const KosResultModel({
    required this.idKos,
    required this.namaKos,
    required this.kota,
    required this.wilayah,
    required this.fotoPath,
    required this.hargaPerBulan,
    required this.predictedPrice,
    required this.distanceKm,
    required this.finalScore,
    required this.semanticScore,
    required this.geospatialScore,
    required this.priceBoost,
    required this.priceLabel,
    required this.isSuperDeal,
    required this.rating,
    required this.fasilitas,
  });

  factory KosResultModel.fromJson(Map<String, dynamic> json) => KosResultModel(
    idKos: json['id_kos'] as int,
    namaKos: json['nama_kos'] as String? ?? 'Kos Tanpa Nama',
    kota: json['kota'] as String? ?? '',
    wilayah: json['wilayah'] as String? ?? '',
    fotoPath: json['foto_path'] as String? ?? '',
    hargaPerBulan: json['harga_per_bulan'] as int? ?? 0,
    predictedPrice: (json['predicted_price'] as num?)?.toDouble() ?? 0.0,
    distanceKm: (json['distance_km'] as num?)?.toDouble() ?? 0.0,
    finalScore: (json['final_score'] as num?)?.toDouble() ?? 0.0,
    semanticScore: (json['semantic_score'] as num?)?.toDouble() ?? 0.0,
    geospatialScore: (json['geospatial_score'] as num?)?.toDouble() ?? 0.0,
    priceBoost: (json['price_boost'] as num?)?.toDouble() ?? 0.0,
    priceLabel: json['price_label'] as String? ?? 'fair',
    isSuperDeal: json['is_super_deal'] as bool? ?? false,
    rating: (json['rating'] as num?)?.toDouble() ?? 0.0,
    fasilitas: FasilitasModel.fromJson(
      json['fasilitas'] as Map<String, dynamic>? ?? {},
    ),
  );

  /// Hemat harga dibanding prediksi (dalam Rupiah)
  double get priceSaving => predictedPrice - hargaPerBulan;

  /// Persentase hemat (positif = lebih murah dari prediksi)
  double get discountPercent =>
    predictedPrice > 0 ? (priceSaving / predictedPrice * 100) : 0.0;
}


class SearchResponseModel {
  final String status;
  final String queryOriginal;
  final String queryPreprocessed;
  final String queryExpanded;
  final List<String> expansionTerms;
  final int totalCandidates;
  final double searchTimeMs;
  final int totalResults;
  final int superDealCount;
  final List<KosResultModel> results;

  const SearchResponseModel({
    required this.status,
    required this.queryOriginal,
    required this.queryPreprocessed,
    required this.queryExpanded,
    required this.expansionTerms,
    required this.totalCandidates,
    required this.searchTimeMs,
    required this.totalResults,
    required this.superDealCount,
    required this.results,
  });

  factory SearchResponseModel.fromJson(Map<String, dynamic> json) =>
    SearchResponseModel(
      status: json['status'] as String? ?? 'unknown',
      queryOriginal: json['query_original'] as String? ?? '',
      queryPreprocessed: json['query_preprocessed'] as String? ?? '',
      queryExpanded: json['query_expanded'] as String? ?? '',
      expansionTerms: (json['expansion_terms'] as List<dynamic>?)
          ?.map((e) => e.toString())
          .toList() ?? [],
      totalCandidates: json['total_candidates'] as int? ?? 0,
      searchTimeMs: (json['search_time_ms'] as num?)?.toDouble() ?? 0.0,
      totalResults: json['total_results'] as int? ?? 0,
      superDealCount: json['super_deal_count'] as int? ?? 0,
      results: (json['results'] as List<dynamic>?)
          ?.map((e) => KosResultModel.fromJson(e as Map<String, dynamic>))
          .toList() ?? [],
    );
}
```

---

## FILE 11: `smart-kos-flutter/lib/services/search_service.dart`

```dart
// lib/services/search_service.dart
// Layer HTTP: kirim POST request ke FastAPI, parse JSON → SearchResponseModel.
// Widget layer tidak pernah berinteraksi langsung dengan Dio.

import 'package:dio/dio.dart';
import '../core/api_client.dart';
import '../core/constants.dart';
import '../models/kos_result.dart';

// Sealed class untuk state hasil search (Dart 3.0+)
sealed class SearchResult {}

class SearchSuccess extends SearchResult {
  final SearchResponseModel data;
  SearchSuccess(this.data);
}

class SearchError extends SearchResult {
  final String message;
  final String? hint;
  SearchError(this.message, {this.hint});
}

class SearchLoading extends SearchResult {}


class KosSearchService {
  final Dio _dio = ApiClient().dio;

  /// Kirim POST request ke FastAPI endpoint /api/v1/search-kos
  ///
  /// Parameters:
  ///   kueri      : Teks pencarian dari user
  ///   latitude   : GPS latitude perangkat (dari Geolocator)
  ///   longitude  : GPS longitude perangkat
  ///   topK       : Jumlah hasil yang diinginkan (default: 10)
  ///
  /// Returns: SearchResult (sealed class: SearchSuccess | SearchError)
  Future<SearchResult> search({
    required String kueri,
    required double latitude,
    required double longitude,
    int topK = AppConstants.defaultTopK,
    int nCandidates = AppConstants.defaultNCandidates,
  }) async {
    try {
      final response = await _dio.post<Map<String, dynamic>>(
        AppConstants.endpointSearch,
        data: {
          'kueri': kueri.trim(),
          'latitude': latitude,
          'longitude': longitude,
          'top_k': topK,
          'n_candidates': nCandidates,
        },
      );

      if (response.statusCode == 200 && response.data != null) {
        final parsed = SearchResponseModel.fromJson(response.data!);
        return SearchSuccess(parsed);
      }

      return SearchError(
        'Respons tidak valid dari server (status: ${response.statusCode})',
      );

    } on DioException catch (e) {
      return _handleDioError(e);
    } catch (e) {
      return SearchError('Terjadi kesalahan tidak terduga: $e');
    }
  }

  /// Cek apakah backend aktif (dipanggil saat app dibuka)
  Future<bool> checkHealth() async {
    try {
      final response = await _dio.get<Map<String, dynamic>>(
        AppConstants.endpointHealth,
      );
      return response.statusCode == 200 &&
          response.data?['status'] == 'healthy';
    } catch (_) {
      return false;
    }
  }

  SearchError _handleDioError(DioException e) {
    switch (e.type) {
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
        return SearchError(
          'Koneksi ke server timeout.',
          hint: 'Pastikan server FastAPI aktif di ${AppConstants.baseUrl}',
        );

      case DioExceptionType.receiveTimeout:
        return SearchError(
          'Server terlalu lama merespons (> ${AppConstants.receiveTimeout.inSeconds}s).',
          hint: 'IndoBERT inference mungkin berjalan lambat di CPU. '
                'Coba kurangi n_candidates atau gunakan GPU.',
        );

      case DioExceptionType.connectionError:
        return SearchError(
          'Tidak dapat terhubung ke server.',
          hint: 'Periksa: (1) FastAPI aktif? (2) URL benar? '
                'Emulator Android gunakan 10.0.2.2, bukan localhost.',
        );

      case DioExceptionType.badResponse:
        final statusCode = e.response?.statusCode;
        final serverMsg = e.response?.data?['message'] as String?;
        final serverHint = e.response?.data?['hint'] as String?;

        if (statusCode == 422) {
          return SearchError(
            serverMsg ?? 'Parameter pencarian tidak valid.',
            hint: serverHint,
          );
        }
        if (statusCode == 503) {
          return SearchError(
            serverMsg ?? 'Layanan pencarian sedang disiapkan.',
            hint: serverHint ??
                'Tunggu beberapa saat dan coba lagi.',
          );
        }
        return SearchError(
          serverMsg ?? 'Server error (kode: $statusCode)',
          hint: serverHint,
        );

      default:
        return SearchError('Error jaringan: ${e.message}');
    }
  }
}
```

---

## FILE 12: `smart-kos-flutter/lib/providers/search_provider.dart`

```dart
// lib/providers/search_provider.dart
// ChangeNotifier sebagai state manager — disuntikkan ke widget tree via Provider.
// Widget cukup listen ke provider ini; tidak perlu tahu tentang Dio atau HTTP.

import 'package:flutter/foundation.dart';
import 'package:geolocator/geolocator.dart';
import '../models/kos_result.dart';
import '../services/search_service.dart';

enum SearchState { idle, loading, success, error }

class KosSearchProvider extends ChangeNotifier {
  final KosSearchService _service = KosSearchService();

  SearchState _state = SearchState.idle;
  SearchResponseModel? _lastResponse;
  String _errorMessage = '';
  String _errorHint = '';
  Position? _currentPosition;
  String _currentQuery = '';

  // ── Getters ────────────────────────────────────────────────────────────────
  SearchState get state => _state;
  SearchResponseModel? get response => _lastResponse;
  List<KosResultModel> get results => _lastResponse?.results ?? [];
  bool get isLoading => _state == SearchState.loading;
  bool get hasResults => _lastResponse != null && results.isNotEmpty;
  String get errorMessage => _errorMessage;
  String get errorHint => _errorHint;
  String get currentQuery => _currentQuery;
  int get superDealCount => _lastResponse?.superDealCount ?? 0;
  double get searchTimeMs => _lastResponse?.searchTimeMs ?? 0.0;
  List<String> get expansionTerms => _lastResponse?.expansionTerms ?? [];

  // ── Aksi: Request GPS Permission + Get Location ────────────────────────────
  Future<bool> requestLocationAndFetch() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      _setError(
        'Layanan GPS tidak aktif.',
        hint: 'Aktifkan GPS di pengaturan perangkat Anda.',
      );
      return false;
    }

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.deniedForever ||
        permission == LocationPermission.denied) {
      _setError(
        'Izin lokasi ditolak.',
        hint: 'Berikan izin lokasi di Pengaturan > Aplikasi > Smart-Kos.',
      );
      return false;
    }

    try {
      _currentPosition = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: Duration(seconds: 10),
        ),
      );
      return true;
    } catch (e) {
      _setError('Gagal mendapatkan lokasi GPS: $e');
      return false;
    }
  }

  // ── Aksi: Eksekusi Pencarian ───────────────────────────────────────────────
  Future<void> search(String kueri) async {
    if (kueri.trim().isEmpty) return;

    // Jika belum punya GPS, ambil dulu
    if (_currentPosition == null) {
      final hasLocation = await requestLocationAndFetch();
      if (!hasLocation) return;
    }

    _currentQuery = kueri.trim();
    _state = SearchState.loading;
    _lastResponse = null;
    _errorMessage = '';
    notifyListeners();

    final result = await _service.search(
      kueri: kueri,
      latitude: _currentPosition!.latitude,
      longitude: _currentPosition!.longitude,
    );

    if (result is SearchSuccess) {
      _lastResponse = result.data;
      _state = SearchState.success;
    } else if (result is SearchError) {
      _setError(result.message, hint: result.hint);
    }

    notifyListeners();
  }

  void clearSearch() {
    _state = SearchState.idle;
    _lastResponse = null;
    _errorMessage = '';
    _currentQuery = '';
    notifyListeners();
  }

  void _setError(String message, {String? hint}) {
    _state = SearchState.error;
    _errorMessage = message;
    _errorHint = hint ?? '';
    notifyListeners();
  }
}
```

---

## FILE 13: `smart-kos-flutter/lib/screens/widgets/super_deal_badge.dart`

```dart
// lib/screens/widgets/super_deal_badge.dart
// Badge "🔥 HARGA SUPER DEAL!" dengan animasi pulse.
// Dipanggil oleh KosCard hanya jika is_super_deal == true.
//
// Animasi: opacity pulse 1.0 ↔ 0.7 setiap 800ms — efek "berkedip" yang menarik
// tanpa mengganggu keterbacaan. Menggunakan flutter_animate package.

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../core/constants.dart';

class SuperDealBadge extends StatelessWidget {
  final double discountPercent; // Persentase hemat (mis: 18.5 → "Hemat 18%")

  const SuperDealBadge({super.key, this.discountPercent = 0.0});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: const Color(AppConstants.superDealColorValue),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: const Color(AppConstants.superDealColorValue).withOpacity(0.4),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('🔥', style: TextStyle(fontSize: 12)),
          const SizedBox(width: 5),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'HARGA SUPER DEAL!',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.5,
                ),
              ),
              if (discountPercent > 0)
                Text(
                  'Hemat ${discountPercent.toStringAsFixed(0)}% dari harga wajar',
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 9,
                    fontWeight: FontWeight.w500,
                  ),
                ),
            ],
          ),
        ],
      ),
    )
    // Flutter Animate: pulse setiap 800ms
    .animate(
      onPlay: (controller) => controller.repeat(reverse: true),
    )
    .fadeIn(duration: 300.ms)
    .then()
    .shimmer(
      duration: 800.ms,
      color: Colors.white.withOpacity(0.3),
      delay: 500.ms,
    );
  }
}
```

---

## FILE 14: `smart-kos-flutter/lib/screens/widgets/kos_card.dart`

```dart
// lib/screens/widgets/kos_card.dart
// Card widget untuk satu item kos di ListView hasil pencarian.
// ─────────────────────────────────────────────────────────────────────────────
// LOGIKA UTAMA SUPER DEAL (conditional render):
//   if (kos.isSuperDeal) → tampilkan SuperDealBadge + highlight border amber
//   else                 → tampilkan kartu standar tanpa badge

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../../models/kos_result.dart';
import 'super_deal_badge.dart';

class KosCard extends StatelessWidget {
  final KosResultModel kos;
  final int rank; // Urutan di list (1-based, dari final_score tertinggi)

  const KosCard({super.key, required this.kos, required this.rank});

  @override
  Widget build(BuildContext context) {
    final isSuperDeal = kos.isSuperDeal;  // ← KUNCI: conditional dari is_super_deal API
    final rupiahFmt = NumberFormat.currency(
      locale: 'id_ID',
      symbol: 'Rp ',
      decimalDigits: 0,
    );

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        // ── CONDITIONAL BORDER: amber jika Super Deal, abu-abu jika tidak ──
        border: Border.all(
          color: isSuperDeal
              ? const Color(0xFFD97706)  // Amber-600 (Super Deal)
              : const Color(0xFFE5E7EB), // Gray-200 (Normal)
          width: isSuperDeal ? 2.0 : 1.0,
        ),
        boxShadow: [
          BoxShadow(
            color: isSuperDeal
                ? const Color(0xFFD97706).withOpacity(0.12)
                : Colors.black.withOpacity(0.05),
            blurRadius: isSuperDeal ? 16 : 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ─────────────────────────────────────────────────────────────────
          // HEADER CARD: Foto + Rank Badge + Super Deal Badge
          // ─────────────────────────────────────────────────────────────────
          Stack(
            children: [
              // Foto kos (placeholder jika path kosong)
              ClipRRect(
                borderRadius: const BorderRadius.vertical(top: Radius.circular(14)),
                child: Container(
                  height: 140,
                  width: double.infinity,
                  color: const Color(0xFFF3F4F6),
                  child: kos.fotoPath.isNotEmpty
                      ? Image.network(
                          'http://10.0.2.2:8000/static/${kos.fotoPath}',
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => _buildPhotoPlaceholder(),
                        )
                      : _buildPhotoPlaceholder(),
                ),
              ),

              // Rank badge (pojok kiri atas)
              Positioned(
                top: 10,
                left: 10,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.65),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '#$rank',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),

              // ── 🔥 SUPER DEAL BADGE (pojok kanan atas) ──────────────────
              // Hanya muncul jika is_super_deal == true dari API
              if (isSuperDeal)
                Positioned(
                  top: 10,
                  right: 10,
                  child: SuperDealBadge(
                    discountPercent: kos.discountPercent,
                  ),
                ),
            ],
          ),

          // ─────────────────────────────────────────────────────────────────
          // BODY CARD: Nama, Lokasi, Harga, Jarak, Fasilitas
          // ─────────────────────────────────────────────────────────────────
          Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Nama kos
                Text(
                  kos.namaKos,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF111827),
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 3),

                // Lokasi
                Row(
                  children: [
                    const Icon(Icons.location_on_outlined,
                        size: 13, color: Color(0xFF6B7280)),
                    const SizedBox(width: 3),
                    Expanded(
                      child: Text(
                        '${kos.wilayah}, ${kos.kota}',
                        style: const TextStyle(
                          fontSize: 11.5,
                          color: Color(0xFF6B7280),
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    const SizedBox(width: 8),
                    const Icon(Icons.navigation_outlined,
                        size: 13, color: Color(0xFF6B7280)),
                    const SizedBox(width: 2),
                    Text(
                      '${kos.distanceKm.toStringAsFixed(1)} km',
                      style: const TextStyle(
                        fontSize: 11.5,
                        color: Color(0xFF6B7280),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),

                // Harga aktual + harga prediksi
                Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    // ── Harga aktual (utama) ──────────────────────────────
                    Text(
                      rupiahFmt.format(kos.hargaPerBulan),
                      style: TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w800,
                        // Warna amber jika Super Deal, navy jika tidak
                        color: isSuperDeal
                            ? const Color(0xFFD97706)
                            : const Color(0xFF1B2A4A),
                      ),
                    ),
                    const Text(
                      '/bln',
                      style: TextStyle(fontSize: 11, color: Color(0xFF9CA3AF)),
                    ),
                    const Spacer(),

                    // ── Harga prediksi RF (ditampilkan hanya jika > 0) ────
                    if (kos.predictedPrice > 0)
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          const Text(
                            'Harga wajar',
                            style: TextStyle(
                              fontSize: 9,
                              color: Color(0xFF9CA3AF),
                            ),
                          ),
                          Text(
                            rupiahFmt.format(kos.predictedPrice),
                            style: TextStyle(
                              fontSize: 11,
                              // Coret harga prediksi jika Super Deal
                              decoration: isSuperDeal
                                  ? TextDecoration.lineThrough
                                  : null,
                              color: isSuperDeal
                                  ? const Color(0xFF9CA3AF)
                                  : const Color(0xFF6B7280),
                            ),
                          ),
                        ],
                      ),
                  ],
                ),
                const SizedBox(height: 10),

                // ── Fasilitas chips ───────────────────────────────────────
                Wrap(
                  spacing: 5,
                  runSpacing: 4,
                  children: _buildFasilitasChips(kos.fasilitas),
                ),
                const SizedBox(height: 10),

                // ── Footer: Rating + Final Score ──────────────────────────
                Row(
                  children: [
                    // Rating bintang
                    const Icon(Icons.star_rounded,
                        size: 14, color: Color(0xFFF59E0B)),
                    const SizedBox(width: 3),
                    Text(
                      kos.rating.toStringAsFixed(1),
                      style: const TextStyle(
                        fontSize: 11.5,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFF374151),
                      ),
                    ),
                    const Spacer(),

                    // Final Score chip (untuk transparansi algoritma)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: const Color(0xFFEFF6FF),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        'Score: ${(kos.finalScore * 100).toStringAsFixed(1)}',
                        style: const TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                          color: Color(0xFF1D4ED8),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Helper: Bangun chip fasilitas ─────────────────────────────────────────
  List<Widget> _buildFasilitasChips(FasilitasModel f) {
    final chips = <_FasilitasChipData>[];
    if (f.ac) chips.add(_FasilitasChipData('❄️', 'AC'));
    if (f.kamarMandiDalam) chips.add(_FasilitasChipData('🚿', 'KMD'));
    if (f.wifi) chips.add(_FasilitasChipData('📶', 'WiFi'));
    if (f.listrikInclude) chips.add(_FasilitasChipData('⚡', 'Listrik'));
    if (f.security24Jam) chips.add(_FasilitasChipData('🔒', 'Aman 24J'));
    if (f.parkir) chips.add(_FasilitasChipData('🚗', 'Parkir'));
    if (f.dapur) chips.add(_FasilitasChipData('🍳', 'Dapur'));
    if (f.laundry) chips.add(_FasilitasChipData('👕', 'Laundry'));

    return chips.map((c) => _buildChip(c)).toList();
  }

  Widget _buildChip(_FasilitasChipData data) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
    decoration: BoxDecoration(
      color: const Color(0xFFF3F4F6),
      borderRadius: BorderRadius.circular(8),
    ),
    child: Text(
      '${data.emoji} ${data.label}',
      style: const TextStyle(fontSize: 10, color: Color(0xFF374151)),
    ),
  );

  Widget _buildPhotoPlaceholder() => const Center(
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.home_work_outlined, size: 36, color: Color(0xFFD1D5DB)),
        SizedBox(height: 4),
        Text('Foto tidak tersedia',
            style: TextStyle(fontSize: 10, color: Color(0xFF9CA3AF))),
      ],
    ),
  );
}

class _FasilitasChipData {
  final String emoji;
  final String label;
  const _FasilitasChipData(this.emoji, this.label);
}
```

---

## FILE 15: `smart-kos-flutter/lib/screens/search_screen.dart`

```dart
// lib/screens/search_screen.dart
// Layar utama aplikasi Smart-Kos.
// Menggabungkan: search bar, PRF info chip, loading shimmer, ListView kos.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shimmer/shimmer.dart';
import '../providers/search_provider.dart';
import 'widgets/kos_card.dart';

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final _searchController = TextEditingController();
  final _scrollController = ScrollController();

  @override
  void dispose() {
    _searchController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _onSearch() {
    final query = _searchController.text.trim();
    if (query.isEmpty) return;
    FocusScope.of(context).unfocus();
    context.read<KosSearchProvider>().search(query);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF1F5F9),
      body: SafeArea(
        child: Column(
          children: [
            // ── HEADER + SEARCH BAR ────────────────────────────────────────
            Container(
              color: const Color(0xFF1B2A4A),
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '🏠 Smart-Kos',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const Text(
                    'Cari kos dengan AI — Semantic + GPS + Harga Wajar',
                    style: TextStyle(
                      color: Colors.white60,
                      fontSize: 11,
                    ),
                  ),
                  const SizedBox(height: 14),
                  // Search input + tombol cari
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _searchController,
                          style: const TextStyle(fontSize: 14, color: Color(0xFF111827)),
                          decoration: InputDecoration(
                            hintText: 'cari kos murah wifi AC dekat kampus...',
                            hintStyle: const TextStyle(
                                fontSize: 13, color: Color(0xFF9CA3AF)),
                            filled: true,
                            fillColor: Colors.white,
                            contentPadding: const EdgeInsets.symmetric(
                                horizontal: 14, vertical: 12),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                              borderSide: BorderSide.none,
                            ),
                            prefixIcon: const Icon(Icons.search,
                                color: Color(0xFF9CA3AF), size: 20),
                          ),
                          onSubmitted: (_) => _onSearch(),
                          textInputAction: TextInputAction.search,
                        ),
                      ),
                      const SizedBox(width: 8),
                      SizedBox(
                        height: 48,
                        child: ElevatedButton(
                          onPressed: _onSearch,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFFD97706),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            padding: const EdgeInsets.symmetric(horizontal: 16),
                          ),
                          child: const Text(
                            'Cari',
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // ── KONTEN UTAMA ───────────────────────────────────────────────
            Expanded(
              child: Consumer<KosSearchProvider>(
                builder: (context, provider, _) {
                  // Loading state → shimmer skeleton
                  if (provider.isLoading) {
                    return _buildShimmerList();
                  }

                  // Error state
                  if (provider.state == SearchState.error) {
                    return _buildErrorView(provider);
                  }

                  // Success state dengan hasil
                  if (provider.hasResults) {
                    return _buildResultsList(provider);
                  }

                  // Idle state (belum ada pencarian)
                  return _buildIdleView();
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── RESULTS LIST ─────────────────────────────────────────────────────────
  Widget _buildResultsList(KosSearchProvider provider) {
    return Column(
      children: [
        // Meta info bar (PRF expansion terms + stats)
        _buildMetaBar(provider),
        // ListView kos
        Expanded(
          child: ListView.builder(
            controller: _scrollController,
            padding: const EdgeInsets.only(top: 8, bottom: 20),
            itemCount: provider.results.length,
            itemBuilder: (context, index) {
              final kos = provider.results[index];
              return KosCard(
                kos: kos,
                rank: index + 1,
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildMetaBar(KosSearchProvider provider) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                '${provider.response?.totalResults ?? 0} kos ditemukan',
                style: const TextStyle(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFF374151),
                ),
              ),
              const Spacer(),
              if (provider.superDealCount > 0)
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFEF3C7),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '🔥 ${provider.superDealCount} Super Deal',
                    style: const TextStyle(
                      fontSize: 10.5,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFFD97706),
                    ),
                  ),
                ),
              const SizedBox(width: 6),
              Text(
                '${provider.searchTimeMs.toStringAsFixed(0)}ms',
                style: const TextStyle(
                  fontSize: 10,
                  color: Color(0xFF9CA3AF),
                ),
              ),
            ],
          ),
          // PRF expansion terms chip
          if (provider.expansionTerms.isNotEmpty) ...[
            const SizedBox(height: 6),
            Row(
              children: [
                const Text(
                  '🔄 Pencarian diperluas: ',
                  style: TextStyle(fontSize: 10.5, color: Color(0xFF6B7280)),
                ),
                Expanded(
                  child: Text(
                    provider.expansionTerms.map((t) => '+$t').join(', '),
                    style: const TextStyle(
                      fontSize: 10.5,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF6D28D9),
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  // ── SHIMMER LOADING ───────────────────────────────────────────────────────
  Widget _buildShimmerList() {
    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      itemCount: 5,
      itemBuilder: (context, _) => Shimmer.fromColors(
        baseColor: const Color(0xFFE5E7EB),
        highlightColor: const Color(0xFFF9FAFB),
        child: Container(
          margin: const EdgeInsets.only(bottom: 12),
          height: 220,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
    );
  }

  // ── ERROR VIEW ────────────────────────────────────────────────────────────
  Widget _buildErrorView(KosSearchProvider provider) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.wifi_off_rounded,
                size: 56, color: Color(0xFFD1D5DB)),
            const SizedBox(height: 12),
            Text(
              provider.errorMessage,
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: Color(0xFF374151),
              ),
            ),
            if (provider.errorHint.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                provider.errorHint,
                textAlign: TextAlign.center,
                style: const TextStyle(
                    fontSize: 12, color: Color(0xFF9CA3AF)),
              ),
            ],
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () {
                if (_searchController.text.isNotEmpty) _onSearch();
              },
              icon: const Icon(Icons.refresh),
              label: const Text('Coba Lagi'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF1B2A4A),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── IDLE VIEW ─────────────────────────────────────────────────────────────
  Widget _buildIdleView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text('🏠', style: TextStyle(fontSize: 52)),
          const SizedBox(height: 12),
          const Text(
            'Cari kos impianmu',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: Color(0xFF1B2A4A),
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Contoh: "kos putri AC dekat kampus"\n"kos murah wifi security 24 jam"',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 12, color: Color(0xFF9CA3AF)),
          ),
          const SizedBox(height: 20),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            alignment: WrapAlignment.center,
            children: [
              'kos wifi AC murah',
              'kos putri dekat kampus',
              'kos bisa masak dapur',
              'kos aman security 24 jam',
            ].map((q) => GestureDetector(
              onTap: () {
                _searchController.text = q;
                _onSearch();
              },
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 7),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: const Color(0xFFE5E7EB)),
                ),
                child: Text(
                  q,
                  style: const TextStyle(
                    fontSize: 11.5,
                    color: Color(0xFF374151),
                  ),
                ),
              ),
            )).toList(),
          ),
        ],
      ),
    );
  }
}
```

---

## FILE 16: `smart-kos-flutter/lib/main.dart`

```dart
// lib/main.dart
// Entry point aplikasi Smart-Kos Flutter.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/search_provider.dart';
import 'screens/search_screen.dart';

void main() {
  runApp(const SmartKosApp());
}

class SmartKosApp extends StatelessWidget {
  const SmartKosApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => KosSearchProvider()),
      ],
      child: MaterialApp(
        title: 'Smart-Kos',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF1B2A4A),
          ),
          useMaterial3: true,
          fontFamily: 'Roboto',
        ),
        home: const SearchScreen(),
      ),
    );
  }
}
```

---

## FILE 17: Cara Menjalankan Lokal (Instruksi Lengkap)

```bash
# ══════════════════════════════════════════════════════════
# BACKEND — FastAPI (Terminal 1)
# ══════════════════════════════════════════════════════════

cd smart-kos-backend
source venv/bin/activate

# Wajib sudah selesai dari Tahap 1 & 2:
# ✅ python scripts/augment_dataset.py
# ✅ python scripts/preprocess_dataset.py
# ✅ python scripts/init_postgis.py
# ✅ python scripts/rebuild_faiss_indobert.py  ← Jalankan di Colab jika CPU lambat

# Jalankan FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Verifikasi:
# → http://localhost:8000/docs          Swagger UI
# → http://localhost:8000/api/v1/health Cek semua komponen

# Test manual dengan curl:
curl -X POST http://localhost:8000/api/v1/search-kos \
  -H "Content-Type: application/json" \
  -d '{
    "kueri": "kos putri AC kamar mandi dalam dekat kampus",
    "latitude": -6.1862,
    "longitude": 106.8348,
    "top_k": 5
  }'


# ══════════════════════════════════════════════════════════
# FLUTTER (Terminal 2)
# ══════════════════════════════════════════════════════════

cd smart-kos-flutter
flutter pub get

# Jalankan di emulator Android (wajib: Android SDK + emulator aktif)
flutter run

# Jika device fisik: ganti _baseUrlDevice di constants.dart
# dengan IP LAN laptop: ipconfig (Windows) / ifconfig (Mac/Linux)

# Build APK untuk demo sidang:
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

---

## State yang Diingat untuk Tahap 4 (Deployment + Sidang)

```
KOMPONEN TAHAP 3 YANG HARUS ADA UNTUK TAHAP 4:

Backend:
  ✅ app/main.py              → Dockerize (FROM python:3.10-slim)
  ✅ app/api/v1/endpoints/    → Sudah production-ready
  ✅ SearchService.search()   → Diuji via Postman untuk response time baseline
  ✅ /api/v1/health           → Digunakan oleh Docker healthcheck

Flutter:
  ✅ lib/core/constants.dart  → Ganti baseUrl ke URL VPS sebelum build release
  ✅ lib/main.dart             → Ganti debug mode ke false
  ✅ flutter build apk --release → File APK untuk demo sidang

Untuk Bab 4 Laporan (Evaluasi):
  - Screenshot Postman: response time search endpoint (target < 500ms)
  - Screenshot Swagger /docs: validasi 422 jika GPS tidak dikirim
  - Screenshot Flutter: badge Super Deal menyala di device fisik
  - Tabel latency: 30 kueri × response_time_ms (dari field search_time_ms di JSON)
  - Hitung Precision@5, Recall@5, MAP dari 30 kueri test
```

---

*End of TAHAP 3 — FastAPI Bridge + Flutter Integration selesai.*
*State aktif: SearchKosRequest + SearchKosResponse (Pydantic) terkunci.*
*Flutter KosResultModel.fromJson() sinkron dengan JSON response backend.*
*Langkah berikutnya: TAHAP 4 (Docker + VPS Deployment + Metrik Evaluasi IR)*
