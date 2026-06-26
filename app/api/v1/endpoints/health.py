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