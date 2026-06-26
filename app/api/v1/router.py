"""
app/api/v1/router.py — API v1 Route Aggregator
Smart-Kos Hybrid Engine — Tahap 3

Mengumpulkan semua sub-router endpoint ke satu APIRouter v1.
app/main.py hanya perlu include router ini — tidak perlu tahu
struktur internal endpoint.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health, search

api_v1_router = APIRouter()

# Daftarkan semua endpoint group
api_v1_router.include_router(health.router)
api_v1_router.include_router(search.router)