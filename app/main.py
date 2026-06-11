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