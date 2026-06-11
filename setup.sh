#!/bin/bash
# setup.sh — Script persiapan environment Smart-Kos Backend (jalankan sekali)
set -e
echo "🔧 Menyiapkan Smart-Kos Hybrid Engine Backend..."

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Buat seluruh struktur direktori
mkdir -p data/raw data/augmented data/processed
mkdir -p models_ml faiss_index
mkdir -p app/core app/api/v1 app/models app/services
mkdir -p stki scripts migrations assets/kos_photos

# Buat file __init__.py untuk setiap package Python
touch app/__init__.py app/core/__init__.py
touch app/api/__init__.py app/api/v1/__init__.py
touch app/models/__init__.py app/services/__init__.py
touch stki/__init__.py

# Salin template .env jika belum ada
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠  File .env dibuat. Sesuaikan konfigurasi database Anda sebelum lanjut."
fi

echo "✅ Setup selesai! Ikuti Urutan Eksekusi di README_TAHAP1."