"""
app/services/search_service.py — Hybrid Search Pipeline Orchestrator
Smart-Kos Hybrid Engine — Tahap 2

Mengintegrasikan seluruh komponen AI engine menjadi satu pipeline pencarian.
Kelas ini MURNI logika bisnis — tidak mengandung dekorator FastAPI.
Endpoint HTTP akan dibungkus di app/api/v1/endpoints/search.py pada Tahap 3.

Pipeline Lengkap (11 Langkah):
    1.  Sastrawi preprocessing kueri
    2.  IndoBERT embedding kueri → vektor 768-dim
    3.  FAISS Search Putaran 1 → top-5 docs untuk PRF feedback
    4.  Fetch deskripsi_clean dari PostgreSQL (PRF input)
    5.  PRF: ekstrak term dominan → expand kueri
    6.  IndoBERT re-embed kueri diperluas → vektor 768-dim baru
    7.  FAISS Search Putaran 2 (re-ranking) → top-20 kandidat
    8.  Fetch data kos lengkap + jarak PostGIS Haversine dari PostgreSQL
    9.  RF batch price prediction untuk semua kandidat
    10. Fusion scoring (semantic + geo + price) untuk semua kandidat
    11. Sort descending by final_score, return top_k
"""

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.pricing_service import PricePredictionResult, PricingService
from stki.embedder import IndoBERTEmbedder
from stki.fusion_scorer import FusionScorer, KosScoringInput, KosScoringResult
from stki.indexer import KosVectorIndexer
from stki.preprocessor import IndonesianTextPreprocessor
from stki.prf_engine import PseudoRelevanceFeedback

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    """Respons lengkap dari satu search request (akan di-serialize ke JSON di Tahap 3)."""

    query_original: str           # Kueri mentah dari user
    query_preprocessed: str       # Setelah Sastrawi pipeline
    query_expanded: str           # Setelah PRF expansion
    expansion_terms: list[str]    # Term yang ditambahkan PRF (transparansi)
    results: list[KosScoringResult]           # Kos terurut by final_score
    total_candidates_retrieved: int           # Pool sebelum filter top_k
    search_time_ms: float         # Latency end-to-end (ms)


class SearchService:
    """
    Orkestrasi pipeline Smart-Kos Hybrid Search.

    Semua komponen diinjeksikan via __init__ (Dependency Injection Pattern).
    Keuntungan:
      - Mudah di-unit test (setiap komponen dapat di-mock secara terpisah)
      - Mudah dikonfigurasi per-environment (dev/staging/production)
      - Mudah diswap komponen (misal: IndoBERT → model lain di masa depan)

    Setup (akan dilakukan di FastAPI lifespan handler, Tahap 3):
        embedder = IndoBERTEmbedder(settings.indobert_model_name)
        indexer  = KosVectorIndexer(settings.faiss_indobert_index_path, ..., embedder)
        indexer.load_index()
        search_svc = SearchService(embedder, indexer, ...)
        app.state.search_service = search_svc
    """

    def __init__(
        self,
        embedder: IndoBERTEmbedder,
        indexer: KosVectorIndexer,
        prf_engine: PseudoRelevanceFeedback,
        pricing_service: PricingService,
        fusion_scorer: FusionScorer,
        preprocessor: IndonesianTextPreprocessor,
    ) -> None:
        self.embedder = embedder
        self.indexer = indexer
        self.prf_engine = prf_engine
        self.pricing_service = pricing_service
        self.fusion_scorer = fusion_scorer
        self.preprocessor = preprocessor

    async def search(
        self,
        query: str,
        user_lat: float,
        user_lon: float,
        top_k: int = 10,
        n_candidates: int = 20,
        db: AsyncSession | None = None,
    ) -> SearchResponse:
        """
        Eksekusi full hybrid search pipeline (11 langkah).

        Args:
            query       : Teks pencarian mentah dari user Flutter.
            user_lat    : Latitude GPS user (WGS84 / EPSG:4326).
            user_lon    : Longitude GPS user.
            top_k       : Jumlah kos yang dikembalikan ke Flutter.
            n_candidates: Pool kandidat dari FAISS Re-rank sebelum top_k filter.
                         Semakin besar → lebih banyak pilihan untuk scoring,
                         semakin kecil → lebih cepat. Default 20 optimal.
            db          : Async SQLAlchemy session.
                         None = mode offline (PostGIS jarak = 999 km, hanya testing).

        Returns:
            SearchResponse dengan ranked results dan metadata pipeline.
        """
        t_start = time.perf_counter()

        # ─────────────────────────────────────────────────────────
        # LANGKAH 1: Preprocessing kueri via Sastrawi
        # ─────────────────────────────────────────────────────────
        query_preprocessed = self.preprocessor.preprocess(query)
        if not query_preprocessed.strip():
            logger.warning(
                f"Kueri kosong setelah preprocessing: '{query}'. "
                "Menggunakan lowercase tanpa stemming."
            )
            query_preprocessed = query.lower().strip()

        # ─────────────────────────────────────────────────────────
        # LANGKAH 2: IndoBERT Embedding kueri awal
        # ─────────────────────────────────────────────────────────
        # query_vector: np.float32 (1, 768), L2-normalized
        # (Digunakan implisit oleh indexer.search() di langkah 3)

        # ─────────────────────────────────────────────────────────
        # LANGKAH 3: FAISS Search Putaran 1 — top-5 untuk PRF
        # ─────────────────────────────────────────────────────────
        n_prf_docs = self.prf_engine.n_feedback_docs
        initial_results = self.indexer.search(
            query_preprocessed, top_k=n_prf_docs
        )
        initial_ids = [r["id_kos"] for r in initial_results]

        # ─────────────────────────────────────────────────────────
        # LANGKAH 4: Fetch deskripsi_clean dari PostgreSQL untuk PRF
        # ─────────────────────────────────────────────────────────
        feedback_docs: list[str] = []
        if initial_ids and db is not None:
            feedback_docs = await self._fetch_deskripsi_clean(db, initial_ids)

        # ─────────────────────────────────────────────────────────
        # LANGKAH 5: PRF — Ekstraksi term + Query Expansion
        # ─────────────────────────────────────────────────────────
        query_expanded, expansion_terms = self.prf_engine.run(
            original_query=query_preprocessed,
            feedback_docs=feedback_docs,
        )

        # ─────────────────────────────────────────────────────────
        # LANGKAH 6 & 7: IndoBERT Re-embed + FAISS Search Putaran 2
        # Jika PRF menemukan term baru → gunakan expanded query
        # Jika tidak → tetap gunakan query preprocessed asli
        # ─────────────────────────────────────────────────────────
        query_for_rerank = (
            query_expanded if expansion_terms else query_preprocessed
        )
        reranked_results = self.indexer.search(
            query_for_rerank, top_k=n_candidates
        )
        candidate_ids = [r["id_kos"] for r in reranked_results]

        # Map id_kos → semantic_score untuk digunakan di fusion scoring
        semantic_score_map: dict[int, float] = {
            r["id_kos"]: r["score"] for r in reranked_results
        }

        # ─────────────────────────────────────────────────────────
        # LANGKAH 8: Fetch data KOS lengkap + jarak PostGIS
        # ─────────────────────────────────────────────────────────
        kos_data_list: list[dict] = []
        if candidate_ids:
            if db is not None:
                kos_data_list = await self._fetch_kos_with_distance(
                    db=db,
                    kos_ids=candidate_ids,
                    user_lat=user_lat,
                    user_lon=user_lon,
                )
            else:
                # Mode offline: gunakan data minimal tanpa jarak nyata
                logger.warning(
                    "DB session tidak tersedia (mode offline). "
                    "Jarak default 999 km digunakan untuk semua kandidat."
                )
                kos_data_list = [
                    {"id_kos": id_, "distance_km": 999.0, "harga_per_bulan": 0}
                    for id_ in candidate_ids
                ]

        # ─────────────────────────────────────────────────────────
        # LANGKAH 9: Batch RF Price Prediction
        # ─────────────────────────────────────────────────────────
        price_results = self.pricing_service.predict_batch(kos_data_list)
        price_map: dict[int, PricePredictionResult] = {
            r.id_kos: r for r in price_results
        }

        # ─────────────────────────────────────────────────────────
        # LANGKAH 10: Bangun KosScoringInput untuk Fusion Scorer
        # ─────────────────────────────────────────────────────────
        scoring_inputs: list[KosScoringInput] = []
        for kos in kos_data_list:
            kos_id = int(kos.get("id_kos", 0))
            pr = price_map.get(kos_id)

            scoring_inputs.append(
                KosScoringInput(
                    id_kos=kos_id,
                    semantic_score=semantic_score_map.get(kos_id, 0.0),
                    distance_km=float(kos.get("distance_km") or 999.0),
                    price_label=pr.price_label if pr else "fair",
                    nama_kos=str(kos.get("nama_kos", "")),
                    kota=str(kos.get("kota", "")),
                    wilayah=str(kos.get("wilayah", "")),
                    harga_per_bulan=int(kos.get("harga_per_bulan", 0)),
                    predicted_price=pr.predicted_price if pr else 0.0,
                    foto_path=str(kos.get("foto_path", "")),
                    rating=float(kos.get("rating") or 0.0),
                    ac=int(kos.get("ac", 0)),
                    kamar_mandi_dalam=int(kos.get("kamar_mandi_dalam", 0)),
                    wifi=int(kos.get("wifi", 0)),
                    listrik_include=int(kos.get("listrik_include", 0)),
                    parkir=int(kos.get("parkir", 0)),
                    dapur=int(kos.get("dapur", 0)),
                    laundry=int(kos.get("laundry", 0)),
                    security_24jam=int(kos.get("security_24jam", 0)),
                )
            )

        # ─────────────────────────────────────────────────────────
        # LANGKAH 11: Fusion Scoring + Sort + Return top_k
        # ─────────────────────────────────────────────────────────
        ranked_results = self.fusion_scorer.score_and_rank(scoring_inputs)
        final_results = ranked_results[:top_k]

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        super_deal_count = sum(1 for r in final_results if r.is_super_deal)

        logger.info(
            f"Search selesai: '{query}' → {len(final_results)} hasil | "
            f"PRF +{len(expansion_terms)} terms | "
            f"🔥 Super Deal: {super_deal_count} | "
            f"Latency: {elapsed_ms}ms"
        )

        return SearchResponse(
            query_original=query,
            query_preprocessed=query_preprocessed,
            query_expanded=query_expanded,
            expansion_terms=expansion_terms,
            results=final_results,
            total_candidates_retrieved=len(reranked_results),
            search_time_ms=elapsed_ms,
        )

    # ──────────────────────────────────────────────────────────────
    # HELPER: Database Queries (PostgreSQL + PostGIS)
    # ──────────────────────────────────────────────────────────────

    async def _fetch_deskripsi_clean(
        self,
        db: AsyncSession,
        kos_ids: list[int],
    ) -> list[str]:
        """
        Ambil kolom 'deskripsi_clean' dari PostgreSQL untuk PRF feedback set.

        Hanya mengambil dokumen yang deskripsi_clean-nya tidak kosong.
        Urutan tidak dijamin (tidak diperlukan untuk PRF averaging).

        Args:
            db     : AsyncSession SQLAlchemy.
            kos_ids: List id_kos dari FAISS Search Putaran 1.

        Returns:
            List string 'deskripsi_clean' untuk PRF.extract_expansion_terms().
        """
        if not kos_ids:
            return []

        result = await db.execute(
            text(
                "SELECT deskripsi_clean "
                "FROM kos_listings "
                "WHERE id_kos = ANY(:ids) "
                "  AND deskripsi_clean IS NOT NULL "
                "  AND LENGTH(TRIM(deskripsi_clean)) > 0"
            ),
            {"ids": kos_ids},
        )
        return [str(row[0]) for row in result.fetchall()]

    async def _fetch_kos_with_distance(
        self,
        db: AsyncSession,
        kos_ids: list[int],
        user_lat: float,
        user_lon: float,
    ) -> list[dict]:
        """
        Ambil seluruh data kos + hitung jarak Haversine dari lokasi user.

        Query PostGIS:
            ST_Distance(geom, user_point::GEOGRAPHY) / 1000.0 → jarak km

        Menggunakan tipe GEOGRAPHY (bukan GEOMETRY) untuk akurasi jarak
        geodesik (ellipsoidal earth model) yang lebih akurat dari flat-earth.

        Kos tanpa data geom (geom IS NULL) mendapatkan fallback distance=999 km
        agar tidak muncul di posisi teratas ranking.

        Args:
            db      : AsyncSession SQLAlchemy.
            kos_ids : List id_kos kandidat dari FAISS Re-rank.
            user_lat: Latitude GPS user (WGS84).
            user_lon: Longitude GPS user (WGS84).

        Returns:
            List dict — satu per kos, berisi semua kolom + 'distance_km'.
        """
        if not kos_ids:
            return []

        sql = text("""
            SELECT
                k.id_kos,
                k.nama_kos,
                k.kota,
                k.wilayah,
                k.harga_per_bulan,
                k.ac,
                k.kamar_mandi_dalam,
                k.wifi,
                k.listrik_include,
                k.parkir,
                k.dapur,
                k.laundry,
                k.security_24jam,
                k.total_fasilitas,
                k.jarak_ke_kampus_km,
                k.jarak_ke_transportasi_km,
                k.jarak_kampus_dekat,
                k.jarak_transportasi_dekat,
                k.kota_encoded,
                k.tipe_kos_encoded,
                k.ukuran_kamar,
                k.foto_path,
                k.rating,
                CASE
                    WHEN k.geom IS NOT NULL THEN
                        ST_Distance(
                            k.geom,
                            ST_SetSRID(
                                ST_MakePoint(:user_lon, :user_lat),
                                4326
                            )::GEOGRAPHY
                        ) / 1000.0
                    ELSE 999.0
                END AS distance_km
            FROM kos_listings k
            WHERE k.id_kos = ANY(:kos_ids)
        """)

        result = await db.execute(
            sql,
            {
                "user_lat": float(user_lat),
                "user_lon": float(user_lon),
                "kos_ids": kos_ids,
            },
        )
        return [dict(row._mapping) for row in result.fetchall()]