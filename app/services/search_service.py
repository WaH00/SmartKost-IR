"""
app/services/search_service.py — Hybrid Search Pipeline Orchestrator
Smart-Kos Hybrid Engine — Tahap 2

Mengintegrasikan seluruh komponen AI engine menjadi satu pipeline pencarian.
Kelas ini MURNI logika bisnis — tidak mengandung dekorator FastAPI.
Endpoint HTTP akan dibungkus di app/api/v1/endpoints/search.py pada Tahap 3.
"""

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.pricing_service import PricePredictionResult, PricingService
from app.utils.intent_extractor import extract_search_intent  # 🔥 TAMBAHAN: Import Extractor
from stki.embedder import IndoBERTEmbedder
from stki.fusion_scorer import FusionScorer, KosScoringInput, KosScoringResult
from stki.indexer import KosVectorIndexer
from stki.preprocessor import IndonesianTextPreprocessor
from stki.prf_engine import PseudoRelevanceFeedback
import re
from app.services.intent_parser import SmartIntentNER
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
        
        t_start = time.perf_counter()
        
        # ─────────────────────────────────────────────────────────
        # 🔥 STEP 1: SUNTIK MESIN NER DI SINI (PALING ATAS)
        # ─────────────────────────────────────────────────────────
        ner_engine = SmartIntentNER()
        intent = ner_engine.extract_intent(query)
        
        # Ganti variabel query teks lu pakai yang udah bersih dari angka budget & gender
        # Biar pas masuk IndoBERT / FAISS nanti hasilnya jauh lebih akurat
        query_preprocessed = intent["clean_query"] if intent["clean_query"] else query

        # ─────────────────────────────────────────────────────────
        # 🔥 STEP 2: SUSUN KONDISI UNTUK SATPAM SQL LU
        # ─────────────────────────────────────────────────────────
        # Kita kumpulin filter hasil ekstraksi AI NER tadi ke dalam array
        sql_filters = []
        
        if intent["gender"]:
            sql_filters.append(f"k.tipe_kos = '{intent['gender']}'")
            
        if intent["budget_max"]:
            sql_filters.append(f"k.harga_per_bulan <= {intent['budget_max']}")
            
        for fas in intent["fasilitas_wajib"]:
            sql_filters.append(f"k.{fas} = 1")

        # ─────────────────────────────────────────────────────────
        # 🔥 LANGKAH 0: Ekstraksi Niat User (Hybrid Hard Filters)
        # ─────────────────────────────────────────────────────────
        tipe_kos, hard_filters, intent_murah, intent_eksklusif = extract_search_intent(query)

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
        # LANGKAH 2 & 3: IndoBERT Embedding & FAISS Search Putaran 1
        # ─────────────────────────────────────────────────────────
        n_prf_docs = self.prf_engine.n_feedback_docs
        initial_results = self.indexer.search(
            query_preprocessed, top_k=n_prf_docs
        )
        initial_ids = [r["id_kos"] for r in initial_results]

        # ─────────────────────────────────────────────────────────
        # LANGKAH 4 & 5: Fetch deskripsi_clean & PRF Expansion
        # ─────────────────────────────────────────────────────────
        feedback_docs: list[str] = []
        if initial_ids and db is not None:
            feedback_docs = await self._fetch_deskripsi_clean(db, initial_ids)

        query_expanded, expansion_terms = self.prf_engine.run(
            original_query=query_preprocessed,
            feedback_docs=feedback_docs,
        )

        # ─────────────────────────────────────────────────────────
        # LANGKAH 6 & 7: IndoBERT Re-embed + FAISS Search Putaran 2
        # ─────────────────────────────────────────────────────────
        query_for_rerank = (
            query_expanded if expansion_terms else query_preprocessed
        )
        reranked_results = self.indexer.search(
            query_for_rerank, top_k=n_candidates
        )
        candidate_ids = [r["id_kos"] for r in reranked_results]

        semantic_score_map: dict[int, float] = {
            r["id_kos"]: r["score"] for r in reranked_results
        }

        # ─────────────────────────────────────────────────────────
        # 🔥 LANGKAH 8: Fetch data KOS + Jarak + SUNTIKAN HARD FILTER
        # ─────────────────────────────────────────────────────────
        kos_data_list: list[dict] = []
        if candidate_ids:
            if db is not None:
                kos_data_list = await self._fetch_kos_with_distance(
                    db=db,
                    kos_ids=candidate_ids,
                    user_lat=user_lat,
                    user_lon=user_lon,
                    tipe_kos_filter=tipe_kos,     # Inject gender
                    hard_filters=hard_filters,    # Inject fasilitas
                )
            else:
                logger.warning("DB session tidak tersedia (mode offline).")
                kos_data_list = [
                    {"id_kos": id_, "distance_km": 999.0, "harga_per_bulan": 0}
                    for id_ in candidate_ids
                ]

        # ─────────────────────────────────────────────────────────
        # LANGKAH 9 & 10: Batch RF Price Prediction & Siapkan Input Scorer
        # ─────────────────────────────────────────────────────────
        price_results = self.pricing_service.predict_batch(kos_data_list)
        price_map: dict[int, PricePredictionResult] = {
            r.id_kos: r for r in price_results
        }

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
        # 🔥 LANGKAH 11: Fusion Scoring & Terapkan DYNAMIC BOOSTING
        # ─────────────────────────────────────────────────────────
        ranked_results = self.fusion_scorer.score_and_rank(scoring_inputs)

        # Modifikasi Skor sesuai Intent Harga/Eksklusif (Booster & Penalty)
        for res in ranked_results:
            current_score = res.final_score
            
            if intent_murah:
                if res.price_label == "overpriced":
                    current_score -= 0.30  # Penalti berat buat kos mahal
                elif res.is_super_deal or res.harga_per_bulan <= 1500000:
                    current_score += 0.25  # Booster buat kos murah meriah
                    
            if intent_eksklusif:
                if res.ac and res.wifi and res.kamar_mandi_dalam:
                    current_score += 0.15  # Booster fasilitas lengkap
                if res.harga_per_bulan < 1000000:
                    current_score -= 0.20  # Penalti kos zonk terlalu murah

            res.final_score = max(0.0, min  (1.0, current_score))

       # Re-sort karena skor barusan kita acak-acak, lalu potong Top K
        ranked_results = sorted(ranked_results, key=lambda x: x.final_score, reverse=True)
        final_results = ranked_results[:top_k]

        # ===============================================================
        # 🔥 GENERATE DYNAMIC SNIPPET HANYA UNTUK TOP K (Biar Cepat!)
        # ===============================================================
        # Note: Pastikan 'kos_data_list' adalah variabel list dari hasil fetch SQL awal lu ya!
        kos_dict_map = {int(k["id_kos"]): str(k.get("deskripsi_clean", "")) for k in kos_data_list}
        
        for res in final_results:
            deskripsi_asli = kos_dict_map.get(res.id_kos, "")
            # Kita pakai query_expanded biar mesin snippetnya makin pinter nyari kata
            res.highlight_alasan = self._extract_dynamic_snippet(query_expanded, deskripsi_asli)
        # ===============================================================

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
        tipe_kos_filter: str | None = None,
        hard_filters: dict | None = None,
    ) -> list[dict]:
        if not kos_ids:
            return []
            
        if hard_filters is None:
            hard_filters = {}

        base_sql = """
            SELECT
                k.id_kos, k.nama_kos, k.kota, k.wilayah, k.harga_per_bulan,
                k.ac, k.kamar_mandi_dalam, k.wifi, k.listrik_include,
                k.parkir, k.dapur, k.laundry, k.security_24jam,
                k.total_fasilitas, k.jarak_ke_kampus_km,
                k.jarak_ke_transportasi_km, k.jarak_kampus_dekat,
                k.jarak_transportasi_dekat, k.kota_encoded,
                k.tipe_kos_encoded, k.ukuran_kamar, k.foto_path, k.rating,
                k.tipe_kos, k.deskripsi_clean, -- Ditambahkan supaya gender kosnya ditarik
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
        """

        params = {
            "user_lat": float(user_lat),
            "user_lon": float(user_lon),
            "kos_ids": kos_ids,
        }

        # SUNTIKAN FILTER KETAT (PENGHALANG ZONK)
        if tipe_kos_filter:
            base_sql += " AND k.tipe_kos = :tipe_kos_filter"
            params["tipe_kos_filter"] = tipe_kos_filter

        if hard_filters.get("ac"):
            base_sql += " AND k.ac = 1"
        if hard_filters.get("kamar_mandi_dalam"):
            base_sql += " AND k.kamar_mandi_dalam = 1"
        if hard_filters.get("wifi"):
            base_sql += " AND k.wifi = 1"
        if hard_filters.get("parkir"):
            base_sql += " AND k.parkir = 1"

        result = await db.execute(text(base_sql), params)
        return [dict(row._mapping) for row in result.fetchall()]
    
  

    def _extract_dynamic_snippet(self, query: str, deskripsi: str) -> str:
        """
        Mencari 1 kalimat paling relevan dari deskripsi berdasarkan kueri user.
        """
        if not deskripsi:
            return "✨ Kosan nyaman dengan fasilitas memadai."

        # 1. Pecah paragraf jadi daftar kalimat (berdasarkan titik atau enter)
        sentences = [s.strip() for s in re.split(r'[.!?\n]', str(deskripsi)) if len(s.strip()) > 15]
        
        if not sentences:
            return f"✨ Highlight: {str(deskripsi)[:80]}..."

        # 2. Siapkan kata kunci pencarian
        query_words = set(query.lower().split())
        best_sentence = sentences[0]
        max_score = -1.0

        # 3. Adu setiap kalimat dengan kueri user (Jaccard Index sederhana)
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            
            # Hitung irisan (kata yang sama-sama muncul)
            overlap = len(query_words.intersection(sentence_words))
            
            # Kasih bobot tambahan buat keyword fasilitas penting
            if "ac" in query_words and "ac" in sentence.lower(): overlap += 3
            if "wifi" in query_words and "wifi" in sentence.lower(): overlap += 3
            if "murah" in query_words and any(w in sentence.lower() for w in ["murah", "terjangkau", "hemat"]): overlap += 3
            if "kampus" in query_words and "kampus" in sentence.lower(): overlap += 3
            
            # Normalisasi skor
            score = overlap / (len(sentence_words) + 1)

            if score > max_score:
                max_score = score
                best_sentence = sentence

        # 4. Kembalikan dengan UI text yang cantik untuk Flutter
        if max_score > 0:
            return f"💡 Cocok karena: '{best_sentence}'"
        else:
            return f"✨ Highlight: '{sentences[0]}'"
    