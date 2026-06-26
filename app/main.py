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