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