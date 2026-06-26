"""
app/api/v1/schemas/search.py — Search Request & Response Schemas
Smart-Kos Hybrid Engine — Tahap 3

Kontrak data yang mengikat Flutter (client) dan FastAPI (server).
Setiap perubahan di sini = perubahan di Flutter model, dan sebaliknya.

Prinsip desain:
  - Response schema harus serializeable langsung dari KosScoringResult (Tahap 2)
  - Flutter hanya perlu field yang relevan untuk UI — tidak semua field backend
  - Setiap field numerik memiliki batas min/max untuk mencegah abuse request
"""

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class SearchKosRequest(BaseModel):
    """
    Body POST /api/v1/search-kos — dikirim oleh Flutter.

    Validasi otomatis dilakukan Pydantic sebelum menyentuh service layer:
    - kueri tidak boleh kosong setelah strip
    - latitude: -90 s/d +90 (valid WGS84)
    - longitude: -180 s/d +180 (valid WGS84)
    - top_k: batas atas 50 agar tidak membebani inference IndoBERT
    """
    kueri: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Teks pencarian kos dari user Flutter",
        examples=["kos dekat kampus wifi AC murah"],
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude GPS perangkat user (WGS84 / EPSG:4326)",
        examples=[-6.2088],
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude GPS perangkat user (WGS84 / EPSG:4326)",
        examples=[106.8456],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Jumlah hasil yang dikembalikan (1–50, default: 10)",
    )
    n_candidates: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Pool kandidat dari FAISS Re-rank sebelum top_k filter",
    )

    @field_validator("kueri", mode="before")
    @classmethod
    def strip_kueri(cls, v: str) -> str:
        """Normalisasi spasi berlebih di awal/akhir kueri."""
        stripped = str(v).strip()
        if not stripped:
            raise ValueError("Kueri tidak boleh berisi hanya spasi.")
        return stripped

    @model_validator(mode="after")
    def validate_top_k_lte_candidates(self) -> "SearchKosRequest":
        """
        top_k tidak boleh melebihi n_candidates.
        Jika dilanggar, clamp top_k ke n_candidates secara senyap (no error)
        agar user tidak mendapat 422 untuk kasus ini.
        """
        if self.top_k > self.n_candidates:
            self.top_k = self.n_candidates
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "kueri": "kos putri dekat kampus wifi AC kamar mandi dalam",
            "latitude": -6.1862,
            "longitude": 106.8348,
            "top_k": 10,
            "n_candidates": 20,
        }
    }}


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class FasilitasSchema(BaseModel):
    """Sub-schema fasilitas kos (binary flags)."""
    ac: bool
    kamar_mandi_dalam: bool
    wifi: bool
    listrik_include: bool
    parkir: bool
    dapur: bool
    laundry: bool
    security_24jam: bool


class KosResultItem(BaseModel):
    """
    Satu item kos dalam response list — diterima oleh Flutter dan dirender
    sebagai KosCard widget. Setiap field memiliki deskripsi eksplisit
    untuk dokumentasi Swagger yang lengkap.

    Korespondensi dengan KosScoringResult (Tahap 2):
        Semua field diambil langsung dari dataclass KosScoringResult.
        Flutter hanya perlu field ini — field internal pipeline (raw scores)
        diekspose untuk kepentingan riset & transparansi di Bab 4.
    """

    # Identitas
    id_kos: int = Field(..., description="ID unik kos dari database")
    nama_kos: str = Field(..., description="Nama lengkap kos")
    kota: str = Field(..., description="Kota (Jakarta Pusat/Selatan/Timur/Barat/Utara)")
    wilayah: str = Field(..., description="Wilayah/kecamatan kos")
    foto_path: str = Field(..., description="Path relatif file foto kamar")

    # Harga
    harga_per_bulan: int = Field(..., description="Harga sewa aktual (Rp/bulan)")
    predicted_price: float = Field(
        ..., description="Harga wajar prediksi Random Forest (Rp/bulan)"
    )

    # Lokasi
    distance_km: float = Field(
        ..., description="Jarak Haversine dari GPS user ke kos (km)"
    )

    # Skor komponen (untuk transparansi riset + Bab 4 laporan)
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fusion Score akhir [0–1]: 0.5×sem + 0.3×geo + 0.2×price_boost",
    )
    semantic_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity IndoBERT [0–1]"
    )
    geospatial_score: float = Field(
        ..., ge=0.0, le=1.0, description="Skor jarak via exp decay [0–1]"
    )
    price_boost: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Boost harga: 1.0=super_deal, 0.5=underpriced, 0.0=fair/overpriced",
    )

    # Label harga
    price_label: str = Field(
        ...,
        description="Label kelayakan harga: super_deal|underpriced|fair|overpriced",
    )
    is_super_deal: bool = Field(
        ...,
        description="True jika harga aktual < 85% dari prediksi RF → badge 🔥 di Flutter",
    )

    # Fasilitas
    rating: float = Field(..., description="Rating rata-rata dari penghuni (0–5)")
    fasilitas: FasilitasSchema = Field(..., description="Detail fasilitas kos")

    @classmethod
    def from_scoring_result(cls, r: object) -> "KosResultItem":
        """
        Factory method: konversi KosScoringResult (Tahap 2 dataclass)
        → KosResultItem (Pydantic schema).

        Memisahkan kolom fasilitas individual menjadi sub-object FasilitasSchema
        untuk struktur JSON yang lebih bersih di Flutter.
        """
        return cls(
            id_kos=r.id_kos,
            nama_kos=r.nama_kos,
            kota=r.kota,
            wilayah=r.wilayah,
            foto_path=r.foto_path,
            harga_per_bulan=r.harga_per_bulan,
            predicted_price=r.predicted_price,
            distance_km=round(r.distance_km, 3),
            final_score=round(r.final_score, 6),
            semantic_score=round(r.semantic_score, 6),
            geospatial_score=round(r.geospatial_score, 6),
            price_boost=r.price_boost,
            price_label=r.price_label,
            is_super_deal=r.is_super_deal,
            rating=r.rating,
            fasilitas=FasilitasSchema(
                ac=bool(r.ac),
                kamar_mandi_dalam=bool(r.kamar_mandi_dalam),
                wifi=bool(r.wifi),
                listrik_include=bool(r.listrik_include),
                parkir=bool(r.parkir),
                dapur=bool(r.dapur),
                laundry=bool(r.laundry),
                security_24jam=bool(r.security_24jam),
            ),
        )


class SearchKosResponse(BaseModel):
    """
    Respons lengkap POST /api/v1/search-kos — diterima oleh Flutter.

    Selain daftar kos, menyertakan metadata pipeline untuk:
    1. Transparansi akademis (laporan Bab 4: bagaimana PRF bekerja)
    2. Debugging di development (cek expansion_terms, search_time_ms)
    3. UX Flutter (tampilkan "Pencarian diperluas: +kulkas +pantry" di UI)
    """

    status: str = Field("success", description="'success' jika request berhasil")
    # Metadata pipeline (untuk riset + debugging)
    query_original: str = Field(..., description="Kueri asli dari user")
    query_preprocessed: str = Field(..., description="Kueri setelah Sastrawi pipeline")
    query_expanded: str = Field(..., description="Kueri setelah PRF expansion")
    expansion_terms: list[str] = Field(
        ..., description="Term baru yang ditambahkan PRF"
    )
    total_candidates: int = Field(..., description="Jumlah kandidat sebelum filter top_k")
    search_time_ms: float = Field(..., description="Latency end-to-end (ms)")
    # Hasil
    total_results: int = Field(..., description="Jumlah kos dalam list results")
    super_deal_count: int = Field(..., description="Jumlah kos dengan label Super Deal")
    results: list[KosResultItem] = Field(..., description="Daftar kos terurut by final_score")

    model_config = {"json_schema_extra": {
        "example": {
            "status": "success",
            "query_original": "kos putri dekat kampus wifi",
            "query_preprocessed": "kos putri dekat kampus wifi",
            "query_expanded": "kos putri dekat kampus wifi kamar bersih nyaman",
            "expansion_terms": ["kamar", "bersih", "nyaman"],
            "total_candidates": 20,
            "search_time_ms": 187.4,
            "total_results": 10,
            "super_deal_count": 2,
            "results": []
        }
    }}