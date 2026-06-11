"""
app/services/pricing_service.py — Random Forest Pricing Service
Smart-Kos Hybrid Engine — Tahap 2

Menggunakan model_rf.pkl + scaler.pkl (dari notebook Kaila Hidayat, 2024)
untuk memprediksi harga wajar kos berdasarkan fasilitas dan lokasi,
kemudian mengklasifikasikan tingkat kewajaran harga aktual.

Klasifikasi Harga:
    "super_deal"  : actual < 85%  × predicted  → diskon > 15%
    "underpriced" : actual < 95%  × predicted  → diskon 5–15%
    "fair"        : actual ≤ 105% × predicted  → toleransi ± 5%
    "overpriced"  : actual > 105% × predicted  → markup > 5%
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PricePredictionResult:
    """Hasil prediksi harga untuk satu kos."""
    id_kos: int
    actual_price: int
    predicted_price: float    # Prediksi harga wajar Random Forest (Rp/bulan)
    price_label: str          # "super_deal"|"underpriced"|"fair"|"overpriced"
    discount_pct: float       # (predicted − actual) / predicted × 100
                              # Positif = aktual lebih murah dari prediksi


class PricingService:
    """
    Layanan prediksi harga wajar berbasis Random Forest Regressor.

    Model: RandomForestRegressor (Scikit-learn) dengan:
        R² Test  = 0.87 (menjelaskan 87% variansi harga)
        MAE Test = Rp 195.000 (< 10% dari median harga)
        RMSE     = Rp 285.000
    (Performa dari notebook dataset asli Kaila Hidayat, Nov 2024)

    Feature ordering KRITIS — harus identik dengan saat training.
    Sumber urutan: analisis_kos_jakarta.ipynb, Cell "Model Training".
    """

    # Threshold klasifikasi sebagai rasio actual/predicted
    THRESHOLD_SUPER_DEAL: float = 0.85    # actual < 85% predicted
    THRESHOLD_UNDERPRICED: float = 0.95   # actual < 95% predicted
    THRESHOLD_FAIR: float = 1.05          # actual ≤ 105% predicted

    # 16 fitur dalam urutan IDENTIK dengan saat model dilatih — JANGAN diubah
    FEATURE_COLUMNS: tuple[str, ...] = (
        "kota_encoded",
        "ukuran_kamar",
        "tipe_kos_encoded",
        "ac",
        "kamar_mandi_dalam",
        "wifi",
        "listrik_include",
        "parkir",
        "dapur",
        "laundry",
        "security_24jam",
        "jarak_ke_kampus_km",
        "jarak_ke_transportasi_km",
        "total_fasilitas",
        "jarak_kampus_dekat",
        "jarak_transportasi_dekat",
    )

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
    ) -> None:
        """
        Muat model dan scaler dari file .pkl.

        Args:
            model_path : Path ke model_rf.pkl.
            scaler_path: Path ke scaler.pkl.

        Raises:
            FileNotFoundError: Jika salah satu file tidak ditemukan.
        """
        for path in (model_path, scaler_path):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"File ML tidak ditemukan: {path}\n"
                    "Pastikan model_rf.pkl dan scaler.pkl di folder models_ml/"
                )

        logger.info(f"Memuat Random Forest model: {model_path}")
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        logger.info(f"Memuat StandardScaler: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        logger.info(
            f"PricingService siap. "
            f"Model: {type(self._model).__name__} | "
            f"Features: {len(self.FEATURE_COLUMNS)}"
        )

    def _classify(
        self,
        actual_price: int,
        predicted_price: float,
    ) -> tuple[str, float]:
        """
        Klasifikasi label harga berdasarkan rasio actual/predicted.

        Args:
            actual_price    : Harga sewa aktual (Rp/bulan).
            predicted_price : Prediksi model Random Forest (Rp/bulan).

        Returns:
            Tuple (price_label, discount_pct).
        """
        if predicted_price <= 0:
            return "fair", 0.0

        ratio = actual_price / predicted_price
        discount_pct = round((1.0 - ratio) * 100.0, 2)

        if ratio < self.THRESHOLD_SUPER_DEAL:
            label = "super_deal"
        elif ratio < self.THRESHOLD_UNDERPRICED:
            label = "underpriced"
        elif ratio <= self.THRESHOLD_FAIR:
            label = "fair"
        else:
            label = "overpriced"

        return label, discount_pct

    def _build_features_df(self, kos_list: list[dict]) -> pd.DataFrame:
        """
        Bangun DataFrame fitur dari list data kos.
        Urutan kolom dijaga identik dengan FEATURE_COLUMNS.
        Nilai yang hilang di-fill dengan 0 (default fitur binary/numerik).

        Args:
            kos_list: List dict data kos dari PostgreSQL.

        Returns:
            DataFrame shape (N, 16) dengan kolom terurut sesuai FEATURE_COLUMNS.
        """
        rows = [
            {col: float(kos.get(col, 0) or 0) for col in self.FEATURE_COLUMNS}
            for kos in kos_list
        ]
        return pd.DataFrame(rows)[list(self.FEATURE_COLUMNS)]

    def predict_single(self, kos_dict: dict) -> PricePredictionResult:
        """
        Prediksi harga wajar untuk satu kos (digunakan untuk testing/demo).

        Args:
            kos_dict: Dictionary data kos dengan key sesuai FEATURE_COLUMNS
                     ditambah 'id_kos' dan 'harga_per_bulan'.

        Returns:
            PricePredictionResult dengan label dan detail prediksi.
        """
        features_df = self._build_features_df([kos_dict])
        scaled = self._scaler.transform(features_df)
        predicted = float(self._model.predict(scaled)[0])

        actual = int(kos_dict.get("harga_per_bulan", 0))
        label, discount_pct = self._classify(actual, predicted)

        return PricePredictionResult(
            id_kos=int(kos_dict.get("id_kos", 0)),
            actual_price=actual,
            predicted_price=round(predicted, 2),
            price_label=label,
            discount_pct=discount_pct,
        )

    def predict_batch(
        self,
        kos_list: list[dict],
    ) -> list[PricePredictionResult]:
        """
        Prediksi harga wajar untuk sekelompok kos secara vectorized.

        Vectorized inference jauh lebih efisien dari loop predict_single()
        karena sklearn RandomForestRegressor memanfaatkan batch prediction
        secara internal (satu forward pass untuk seluruh batch).

        Args:
            kos_list: List dict data kos dari SearchService._fetch_kos_with_distance().

        Returns:
            List PricePredictionResult (indeks berkorespondensi dengan input).
        """
        if not kos_list:
            return []

        features_df = self._build_features_df(kos_list)
        scaled = self._scaler.transform(features_df)

        # Satu pemanggilan predict() untuk seluruh batch
        predictions: np.ndarray = self._model.predict(scaled)

        results: list[PricePredictionResult] = []
        for kos_dict, predicted in zip(kos_list, predictions):
            actual = int(kos_dict.get("harga_per_bulan", 0))
            label, discount_pct = self._classify(actual, float(predicted))
            results.append(
                PricePredictionResult(
                    id_kos=int(kos_dict.get("id_kos", 0)),
                    actual_price=actual,
                    predicted_price=round(float(predicted), 2),
                    price_label=label,
                    discount_pct=discount_pct,
                )
            )

        super_deal_count = sum(1 for r in results if r.price_label == "super_deal")
        logger.info(
            f"Batch pricing: {len(results)} kos | "
            f"Super Deal: {super_deal_count} | "
            f"Underpriced: {sum(1 for r in results if r.price_label == 'underpriced')}"
        )
        return results