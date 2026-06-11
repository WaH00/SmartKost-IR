from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Konfigurasi terpusat Smart-Kos Backend menggunakan Pydantic-Settings.
    Seluruh nilai dibaca otomatis dari file .env via pola Singleton.
    Menambahkan variabel baru cukup di sini — tidak perlu ubah modul lain.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Konfigurasi Database PostgreSQL ---
    database_url: str = (
        "postgresql+asyncpg://postgres:password@localhost:5432/smart_kos_db"
    )
    # DATABASE_DSN digunakan oleh psycopg2 di scripts admin (bukan FastAPI runtime)
    database_dsn: str = (
        "host=localhost port=5432 dbname=smart_kos_db user=postgres password=password"
    )

    # --- Konfigurasi FAISS Vector Index ---
    faiss_index_path: str = "./faiss_index/kos_tfidf.index"
    faiss_id_map_path: str = "./faiss_index/id_map.pkl"
    faiss_vectorizer_path: str = "./faiss_index/tfidf_vectorizer.pkl"

    # --- Konfigurasi Path File Data ---
    data_raw_path: str = "./data/raw/data_kos_jakarta.csv"
    data_augmented_path: str = "./data/augmented/data_kos_augmented.csv"
    data_processed_path: str = "./data/processed/data_kos_processed.csv"

    # --- Konfigurasi Model Machine Learning ---
    model_rf_path: str = "./models_ml/model_rf.pkl"
    scaler_path: str = "./models_ml/scaler.pkl"
    label_encoder_path: str = "./models_ml/label_encoders.pkl"

    # --- Konfigurasi Aplikasi ---
    app_name: str = "Smart-Kos Hybrid Engine API"
    app_version: str = "1.0.0"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    
    # Tambahkan field-field ini ke class Settings yang sudah ada di Tahap 1:

    # === IndoBERT ===
    indobert_model_name: str = "indobenchmark/indobert-base-p1"
    indobert_max_length: int = 128
    indobert_batch_size: int = 16  # Turunkan ke 4-8 jika OOM di CPU

    # === FAISS IndoBERT ===
    faiss_indobert_index_path: str = "./faiss_index/kos_indobert.index"
    faiss_indobert_id_map_path: str = "./faiss_index/id_map_indobert.pkl"

    # === Pricing Thresholds ===
    super_deal_threshold: float = 0.85
    underpriced_threshold: float = 0.95
    fair_threshold: float = 1.05

    # === Fusion Weights ===
    fusion_weight_semantic: float = 0.50
    fusion_weight_geospatial: float = 0.30
    fusion_weight_price: float = 0.20
    fusion_geo_decay_lambda: float = 0.3

    # === PRF ===
    prf_n_feedback_docs: int = 5
    prf_n_expansion_terms: int = 5


# Instans singleton — diimpor oleh seluruh modul lain dengan:
#   from app.core.config import settings
settings = Settings()