"""
scripts/demo_search.py — Demo & Validasi AI Engine Tahap 2
Smart-Kos Hybrid Engine

Menjalankan full search pipeline TANPA FastAPI dan TANPA database
untuk memvalidasi seluruh komponen AI sebelum Tahap 3 (API endpoint).

Mode yang tersedia:
  1. Mode Offline (default): Tanpa DB — PostGIS jarak = 999km, valid untuk
     memvalidasi IndoBERT + PRF + RF Pricing + Fusion Scoring.
  2. Mode Online: Dengan DB — jalankan setelah init_postgis.py selesai.

Prasyarat:
  ✅ scripts/rebuild_faiss_indobert.py sudah dijalankan (index ada di disk)
  ✅ models_ml/model_rf.pkl dan scaler.pkl tersedia
  ✅ python -m pip install torch transformers sentencepiece (sudah terinstall)

Usage:
  python scripts/demo_search.py
  python scripts/demo_search.py --query "kos wifi ac dekat kampus"
  python scripts/demo_search.py --query "kos murah bisa masak" --top_k 3
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.services.pricing_service import PricingService
from app.services.search_service import SearchService
from stki.embedder import IndoBERTEmbedder
from stki.fusion_scorer import FusionScorer
from stki.indexer import KosVectorIndexer
from stki.preprocessor import IndonesianTextPreprocessor
from stki.prf_engine import PseudoRelevanceFeedback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Kueri contoh — wakili variasi kebutuhan mahasiswa Jakarta
SAMPLE_QUERIES: list[str] = [
    "kos murah wifi dekat kampus",
    "kos putri AC kamar mandi dalam",
    "kos bisa masak dapur bersama",
    "kos aman security 24 jam",
    "hunian nyaman fasilitas lengkap",
]


async def run_demo(
    query: str,
    user_lat: float = -6.1862,    # Default: Jakarta Pusat
    user_lon: float = 106.8348,
    top_k: int = 5,
) -> None:
    """Inisialisasi semua komponen dan jalankan demo search pipeline."""

    logger.info("═" * 65)
    logger.info("DEMO: Smart-Kos Hybrid Engine — Tahap 2 Validation")
    logger.info("═" * 65)

    # ─── Setup komponen ─────────────────────────────────────────
    logger.info("[1/5] Memuat IndoBERT Embedder...")
    embedder = IndoBERTEmbedder(
        model_name=settings.indobert_model_name,
        device="auto",
        max_length=settings.indobert_max_length,
    )

    logger.info("[2/5] Memuat FAISS IndoBERT Index...")
    indexer = KosVectorIndexer(
        index_path=settings.faiss_indobert_index_path,
        id_map_path=settings.faiss_indobert_id_map_path,
        embedder=embedder,
    )
    indexer.load_index()

    logger.info("[3/5] Memuat RF Pricing Service...")
    pricing_service = PricingService(
        model_path=settings.model_rf_path,
        scaler_path=settings.scaler_path,
    )

    logger.info("[4/5] Inisialisasi PRF, FusionScorer, Preprocessor...")
    search_service = SearchService(
        embedder=embedder,
        indexer=indexer,
        prf_engine=PseudoRelevanceFeedback(
            n_feedback_docs=settings.prf_n_feedback_docs,
            n_expansion_terms=settings.prf_n_expansion_terms,
        ),
        pricing_service=pricing_service,
        fusion_scorer=FusionScorer(
            w_semantic=settings.fusion_weight_semantic,
            w_geospatial=settings.fusion_weight_geospatial,
            w_price=settings.fusion_weight_price,
            geo_decay_lambda=settings.fusion_geo_decay_lambda,
        ),
        preprocessor=IndonesianTextPreprocessor(),
    )

    # ─── Jalankan search ────────────────────────────────────────
    logger.info(f"[5/5] Menjalankan pencarian: '{query}'")
    response = await search_service.search(
        query=query,
        user_lat=user_lat,
        user_lon=user_lon,
        top_k=top_k,
        n_candidates=20,
        db=None,  # Mode offline
    )

    # ─── Tampilkan hasil ─────────────────────────────────────────
    print("\n" + "═" * 65)
    print("📊 HASIL PENCARIAN")
    print("═" * 65)
    print(f"  Kueri asli       : {response.query_original}")
    print(f"  Setelah Sastrawi : {response.query_preprocessed}")
    print(f"  Setelah PRF      : {response.query_expanded}")
    print(f"  Term ekspansi    : {response.expansion_terms}")
    print(f"  Total kandidat   : {response.total_candidates_retrieved}")
    print(f"  Latency          : {response.search_time_ms} ms")
    print("─" * 65)

    for i, r in enumerate(response.results, 1):
        badge = "🔥 SUPER DEAL!" if r.is_super_deal else f"[{r.price_label.upper()}]"
        fasilitas = []
        if r.ac:             fasilitas.append("AC")
        if r.kamar_mandi_dalam: fasilitas.append("KMD")
        if r.wifi:           fasilitas.append("WiFi")
        if r.security_24jam: fasilitas.append("Sec24j")
        if r.dapur:          fasilitas.append("Dapur")
        fas_str = "+".join(fasilitas) if fasilitas else "-"

        print(
            f"\n  #{i} id_kos={r.id_kos:04d} | {badge}\n"
            f"     Final Score   : {r.final_score:.4f} "
            f"(sem={r.semantic_score:.3f} + geo={r.geospatial_score:.3f} + "
            f"price_boost={r.price_boost:.1f})\n"
            f"     Harga Aktual  : Rp {r.harga_per_bulan:>12,.0f}/bln\n"
            f"     Harga Wajar   : Rp {r.predicted_price:>12,.0f}/bln (RF predict)\n"
            f"     Kota          : {r.kota} — {r.wilayah}\n"
            f"     Fasilitas     : {fas_str} | Rating: {r.rating}"
        )

    print("═" * 65)

    # ─── Rangkuman statistik ──────────────────────────────────────
    super_deals = [r for r in response.results if r.is_super_deal]
    print(f"\n📈 Ringkasan: {len(response.results)} kos | "
          f"🔥 Super Deal: {len(super_deals)} | "
          f"Latency: {response.search_time_ms}ms")

    if super_deals:
        best = super_deals[0]
        discount = round(
            (best.predicted_price - best.harga_per_bulan) / best.predicted_price * 100, 1
        )
        print(
            f"   Best Super Deal : id={best.id_kos} | "
            f"Hemat {discount}% (Rp {best.predicted_price - best.harga_per_bulan:,.0f}/bln)"
        )

    print("\n✅ Demo selesai. Tahap 2 AI Engine tervalidasi!")
    print("   Langkah berikutnya: TAHAP 3 (FastAPI Endpoints)\n")


async def run_batch_demo(queries: list[str]) -> None:
    """Jalankan multiple kueri untuk benchmarking latency."""
    print("\n" + "═" * 65)
    print("🔬 BATCH DEMO — Latency Benchmark")
    print("═" * 65)

    # Setup sekali untuk semua query (lebih efisien)
    embedder = IndoBERTEmbedder(settings.indobert_model_name, max_length=128)
    indexer = KosVectorIndexer(
        settings.faiss_indobert_index_path,
        settings.faiss_indobert_id_map_path,
        embedder,
    )
    indexer.load_index()
    pricing_service = PricingService(settings.model_rf_path, settings.scaler_path)
    search_service = SearchService(
        embedder=embedder,
        indexer=indexer,
        prf_engine=PseudoRelevanceFeedback(),
        pricing_service=pricing_service,
        fusion_scorer=FusionScorer(
            w_semantic=settings.fusion_weight_semantic,
            w_geospatial=settings.fusion_weight_geospatial,
            w_price=settings.fusion_weight_price,
        ),
        preprocessor=IndonesianTextPreprocessor(),
    )

    latencies: list[float] = []
    for i, q in enumerate(queries, 1):
        resp = await search_service.search(
            query=q, user_lat=-6.1862, user_lon=106.8348, top_k=5, db=None
        )
        latencies.append(resp.search_time_ms)
        super_deals = sum(1 for r in resp.results if r.is_super_deal)
        print(
            f"  Query {i:2d}: {q[:45]:<45} | "
            f"{resp.search_time_ms:6.1f}ms | 🔥×{super_deals}"
        )

    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    print(f"\n  Rata-rata: {avg_lat:.1f}ms | Maksimum: {max_lat:.1f}ms")
    print(
        "  (Catatan: query encoding CPU ~100-200ms, GPU ~10-30ms)\n"
        "  Target production SLA: < 500ms end-to-end dengan GPU."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo validasi Smart-Kos AI Engine Tahap 2"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="kos murah fasilitas lengkap wifi ac",
        help="Teks kueri pencarian kos",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Jumlah hasil yang ditampilkan",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Jalankan batch demo untuk benchmark latency",
    )
    args = parser.parse_args()

    if args.batch:
        asyncio.run(run_batch_demo(SAMPLE_QUERIES))
    else:
        asyncio.run(run_demo(query=args.query, top_k=args.top_k))