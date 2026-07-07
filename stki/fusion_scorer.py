"""
stki/fusion_scorer.py — Hybrid Fusion Scoring Engine
Smart-Kos Hybrid Engine — Tahap 2

Menyatukan tiga sinyal independen menjadi satu skor evaluasi final (Final Score)
yang merepresentasikan nilai holistic sebuah kos bagi pengguna.

═══════════════════════════════════════════════════════════════════
FORMULA UTAMA (Bab 3 Laporan TA — Persamaan 3.1):

    F(k) = w_s × S_sem(k)  +  w_g × S_geo(k)  +  w_p × S_price(k)

    Di mana:
        F(k)       = Final Score kos k ∈ [0.0, 1.0]
        w_s = 0.50 = Bobot sinyal semantik (relevansi teks)
        w_g = 0.30 = Bobot sinyal geospatial (kedekatan lokasi)
        w_p = 0.20 = Bobot sinyal harga (nilai ekonomis)

  Sinyal Semantik — S_sem(k):
    Cosine similarity dari FAISS IndexFlatIP setelah L2-normalization.
    Range: [0.0, 1.0]. Semakin tinggi = teks kos lebih mirip kueri.

  Sinyal Geospatial — S_geo(k):
    Exponential decay berdasarkan jarak Haversine (PostGIS ST_Distance):
        S_geo = exp(−λ × d_km),   λ = 0.3

    Referensi nilai (λ=0.3):
        d=0.0 km → S=1.000  (di lokasi sama dengan user)
        d=1.0 km → S=0.741  (jarak kaki / sepeda)
        d=2.3 km → S=0.500  (threshold moderat)
        d=3.3 km → S=0.370  (1 standar deviasi, Jakarta urban context)
        d=5.0 km → S=0.223  (perlu ojek/motor)
       d=10.0 km → S=0.050  (jauh, tidak direkomendasikan)

    Pemilihan exponential decay (vs linear/sigmoid):
    Memberikan gradient yang smooth dan besar di dekat user,
    mencerminkan preferensi penyewa kos yang sangat mengutamakan jarak dekat.

  Sinyal Harga — S_price(k):
    Boost biner/bertingkat berdasarkan label Random Forest:
        "super_deal"  → 1.0  (diskon > 15% dari harga wajar)
        "underpriced" → 0.5  (diskon 5–15%)
        "fair"        → 0.0  (harga wajar, tidak ada bonus)
        "overpriced"  → 0.0  (terlalu mahal, tidak ada bonus)

  Pemilihan Bobot (0.50 : 0.30 : 0.20):
    Berdasarkan feature importance Random Forest dari dataset:
    kota_encoded=18.5%, jarak_kampus=9.3%, total_fasilitas=12.8%.
    Relevansi tekstual tetap paling dominan (intent user-centric).
    Bobot dapat dikonfigurasi untuk eksperimen ablation study di Bab 4.
═══════════════════════════════════════════════════════════════════
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KosScoringInput:
    """Paket data input untuk kalkulasi fusion score satu kandidat kos."""

    id_kos: int
    semantic_score: float    # Cosine similarity dari FAISS Search 2 [0.0–1.0]
    distance_km: float       # Jarak Haversine dari user ke kos (km, dari PostGIS)
    price_label: str         # "super_deal"|"underpriced"|"fair"|"overpriced"

    # Metadata kos (diteruskan ke output tanpa modifikasi)
    nama_kos: str = ""
    kota: str = ""
    wilayah: str = ""
    harga_per_bulan: int = 0
    predicted_price: float = 0.0
    foto_path: str = ""
    rating: float = 0.0
    ac: int = 0
    kamar_mandi_dalam: int = 0
    wifi: int = 0
    listrik_include: int = 0
    parkir: int = 0
    dapur: int = 0
    laundry: int = 0
    security_24jam: int = 0
    highlight_alasan: str = ""

@dataclass
class KosScoringResult:
    """Hasil kalkulasi fusion score satu kandidat kos (output SearchService)."""

    id_kos: int
    nama_kos: str
    kota: str
    wilayah: str
    harga_per_bulan: int
    distance_km: float
    # Skor komponen (untuk transparansi dan analisis di laporan TA)
    final_score: float
    semantic_score: float
    geospatial_score: float
    price_boost: float
    # Label dan badge
    price_label: str
    is_super_deal: bool      # True jika price_label == "super_deal" → badge 🔥
    predicted_price: float
    # Metadata
    foto_path: str
    rating: float
    ac: int
    kamar_mandi_dalam: int
    wifi: int
    listrik_include: int
    parkir: int
    dapur: int
    laundry: int
    security_24jam: int


class FusionScorer:
    """
    Mesin penyatuan skor hybrid Smart-Kos.

    Seluruh bobot dan parameter dapat dikonfigurasi untuk:
    1. Tuning performa di lingkungan berbeda (kampus vs kota besar)
    2. Eksperimen ablation study (matikan satu sinyal, ukur dampaknya)
    3. A/B testing di production (Tahap 4)

    Attributes:
        w_semantic         : Bobot sinyal semantik (default: 0.50)
        w_geospatial       : Bobot sinyal geospatial (default: 0.30)
        w_price            : Bobot sinyal harga (default: 0.20)
        geo_decay_lambda   : Parameter λ exponential decay (default: 0.3)
    """

    # Tabel price boost — nilai divalidasi secara eksperimental
    PRICE_BOOST_TABLE: dict[str, float] = {
        "super_deal":  1.0,   # Diskon > 15%: bonus penuh
        "underpriced": 0.5,   # Diskon 5–15%: bonus setengah
        "fair":        0.0,   # ±5% toleransi: tanpa bonus
        "overpriced":  0.0,   # Markup > 5%: tanpa bonus
    }

    def __init__(
        self,
        w_semantic: float = 0.50,
        w_geospatial: float = 0.30,
        w_price: float = 0.20,
        geo_decay_lambda: float = 0.3,
    ) -> None:
        """
        Inisialisasi Fusion Scorer dengan validasi kelengkapan bobot.

        Args:
            w_semantic       : Bobot sinyal semantik.
            w_geospatial     : Bobot sinyal geospatial.
            w_price          : Bobot sinyal harga.
            geo_decay_lambda : λ untuk S_geo = exp(−λ × d_km). Semakin besar λ,
                               semakin cepat skor turun terhadap jarak.

        Raises:
            ValueError: Jika total bobot ≠ 1.0 (toleransi 1e-6).
                        Mencegah final_score keluar dari rentang [0.0, 1.0].
        """
        total = w_semantic + w_geospatial + w_price
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Jumlah bobot harus tepat = 1.0. "
                f"Diterima: w_s={w_semantic} + w_g={w_geospatial} + "
                f"w_p={w_price} = {total:.8f}"
            )

        self.w_semantic = w_semantic
        self.w_geospatial = w_geospatial
        self.w_price = w_price
        self.geo_decay_lambda = geo_decay_lambda

        logger.info(
            f"FusionScorer: w_sem={w_semantic}, w_geo={w_geospatial}, "
            f"w_price={w_price}, λ={geo_decay_lambda}"
        )

    def compute_geospatial_score(self, distance_km: float) -> float:
        """
        Hitung geospatial score menggunakan exponential decay.

        Formula: S_geo(d) = exp(−λ × d)

        Dipilih exponential decay karena:
        - Diferensiabel di seluruh domain (gradient ada di semua titik)
        - Asimptotik ke 0 saat d → ∞ (kos sangat jauh → skor sangat kecil)
        - Parameter λ dapat di-tune untuk berbagai konteks urban/rural

        Args:
            distance_km: Jarak dari user ke kos dalam kilometer.
                        Nilai negatif di-clamp ke 0 (edge case GPS error).

        Returns:
            Float S_geo ∈ (0.0, 1.0]. Semakin kecil jarak = semakin tinggi skor.
        """
        return math.exp(-self.geo_decay_lambda * max(0.0, distance_km))

    def compute_price_boost(self, price_label: str) -> float:
        """
        Konversi label harga Random Forest → nilai boost numerik.

        Menggunakan PRICE_BOOST_TABLE untuk mapping deterministik.
        Label yang tidak dikenali (edge case) mendapat boost = 0.0.

        Args:
            price_label: String label dari PricingService.predict_batch().

        Returns:
            Float boost ∈ {0.0, 0.5, 1.0}.
        """
        return self.PRICE_BOOST_TABLE.get(price_label.lower().strip(), 0.0)

    def compute_final_score(
        self,
        semantic_score: float,
        geospatial_score: float,
        price_boost: float,
    ) -> float:
        """
        Hitung final score dari tiga komponen sinyal.

        Formula (Persamaan 3.1):
            F = w_s × S_sem + w_g × S_geo + w_p × S_price

        Seluruh input sudah dalam range [0.0, 1.0] sehingga:
            F_min = 0.0 (semua sinyal = 0, sangat tidak relevan)
            F_max = 1.0 (semua sinyal = 1, sempurna di semua dimensi)

        Args:
            semantic_score   : Cosine similarity dari FAISS (0.0–1.0).
            geospatial_score : Output compute_geospatial_score() (0.0–1.0].
            price_boost      : Output compute_price_boost() {0.0, 0.5, 1.0}.

        Returns:
            Final score ∈ [0.0, 1.0].
        """
        return (
            self.w_semantic * semantic_score
            + self.w_geospatial * geospatial_score
            + self.w_price * price_boost
        )

    def score_and_rank(
        self,
        candidates: list[KosScoringInput],
    ) -> list[KosScoringResult]:
        """
        Hitung final score semua kandidat sekaligus dan urutkan descending.

        Ini adalah method utama yang dipanggil oleh SearchService.
        Semua kalkulasi bersifat pure function (tidak ada side effects)
        sehingga mudah di-unit test secara terisolasi.

        Args:
            candidates: List KosScoringInput dari search pipeline.

        Returns:
            List KosScoringResult terurut berdasarkan final_score (tertinggi → terendah).
            Kos dengan final_score tertinggi di posisi [0] adalah rekomendasi terbaik.
        """
        results: list[KosScoringResult] = []

        for c in candidates:
            geo_score = self.compute_geospatial_score(c.distance_km)
            price_boost = self.compute_price_boost(c.price_label)
            final_score = self.compute_final_score(
                c.semantic_score, geo_score, price_boost
            )

            results.append(
                KosScoringResult(
                    id_kos=c.id_kos,
                    nama_kos=c.nama_kos,
                    kota=c.kota,
                    wilayah=c.wilayah,
                    harga_per_bulan=c.harga_per_bulan,
                    distance_km=round(c.distance_km, 3),
                    final_score=round(final_score, 6),
                    semantic_score=round(c.semantic_score, 6),
                    geospatial_score=round(geo_score, 6),
                    price_boost=price_boost,
                    price_label=c.price_label,
                    is_super_deal=(c.price_label.lower() == "super_deal"),
                    predicted_price=round(c.predicted_price, 0),
                    foto_path=c.foto_path,
                    rating=c.rating,
                    ac=c.ac,
                    kamar_mandi_dalam=c.kamar_mandi_dalam,
                    wifi=c.wifi,
                    listrik_include=c.listrik_include,
                    parkir=c.parkir,
                    dapur=c.dapur,
                    laundry=c.laundry,
                    security_24jam=c.security_24jam,
                )
            )

        # Sort descending: kos terbaik di posisi 0
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results