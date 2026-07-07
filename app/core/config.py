from pydantic import Field
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
        protected_namespaces=("settings_",),
    )

    # --- Konfigurasi Database PostgreSQL ---
    database_url: str = (
        "postgresql+asyncpg://postgres:1234@localhost:5432/smartkos"
    )
    # DATABASE_DSN digunakan oleh psycopg2 di scripts admin (bukan FastAPI runtime)
    database_dsn: str = (
        "host=localhost port=5432 dbname=smartkos user=postgres password=1234"
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
    # Gunakan nama khusus agar tidak bentrok dengan DEBUG milik shell/tool lain.
    debug: bool = Field(default=True, validation_alias="SMARTKOS_DEBUG")
    api_v1_prefix: str = "/api/v1"
    
    # Tambahkan field-field ini ke class Settings yang sudah ada di Tahap 1:
    # === Gemini AI ===
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-3.5-flash"

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
    
    # === API Server ===
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # CORS: ganti "*" dengan IP Flutter emulator di production
    # Emulator Android mengakses host via 10.0.2.2 (bukan localhost)
    cors_origins: list[str] = ["*"]

    # === Search defaults ===
    default_top_k: int = 10
    max_top_k: int = 50
    default_n_candidates: int = 20


# Instans singleton — diimpor oleh seluruh modul lain dengan:
#   from app.core.config import settings
settings = Settings()
