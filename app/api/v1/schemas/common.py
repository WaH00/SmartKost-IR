"""
app/api/v1/schemas/common.py — Shared Pydantic Schemas
Smart-Kos Hybrid Engine — Tahap 3

Skema yang digunakan bersama oleh lebih dari satu endpoint.
Menjaga DRY (Don't Repeat Yourself) di seluruh layer API.
"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Format error standar Smart-Kos API.
    Seluruh HTTPException menggunakan format ini agar Flutter
    dapat mem-parse error secara konsisten tanpa switch-case tipe.
    """
    status: str = Field("error", description="Selalu 'error' untuk respons gagal")
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Deskripsi error yang human-readable")
    detail: str | None = Field(None, description="Detail teknis (opsional)")

    model_config = {"json_schema_extra": {
        "example": {
            "status": "error",
            "code": 422,
            "message": "Parameter kueri tidak valid",
            "detail": "Field 'latitude' wajib diisi"
        }
    }}


class HealthStatus(BaseModel):
    """Respons endpoint GET /health."""
    status: str
    version: str
    components: dict[str, str]
    uptime_seconds: float | None = None