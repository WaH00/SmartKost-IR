# SmartKost Hybrid Engine Backend

Backend pencarian kos berbasis FastAPI yang menggabungkan IndoBERT, FAISS, PostGIS, pseudo-relevance feedback, dan Random Forest untuk pencarian semantik, perhitungan jarak, serta penilaian harga kos.

## Prasyarat

Siapkan perangkat berikut sebelum instalasi:

- Git.
- Python 3.11 64-bit (direkomendasikan). Python 3.10 juga didukung.
- PostgreSQL 13 atau lebih baru.
- Ekstensi PostGIS yang sesuai dengan versi PostgreSQL.
- RAM minimal 8 GB dan koneksi internet pada startup pertama.

Startup pertama akan mengunduh model `indobenchmark/indobert-base-p1` dari Hugging Face. Proses ini dapat memerlukan waktu beberapa menit dan ruang penyimpanan sekitar 500 MB. File indeks FAISS, model Random Forest, dan dataset yang diperlukan sudah tersedia di repo.

## 1. Clone repository

```bash
git clone <URL_REPOSITORY_GITHUB>
cd SmartKost-IR
```

Ganti `<URL_REPOSITORY_GITHUB>` dengan URL repo setelah repo dipublikasikan.

## 2. Buat virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instalasi `torch`, `transformers`, dan `faiss-cpu` dapat memerlukan beberapa menit.

## 3. Siapkan PostgreSQL dan PostGIS

Pastikan service PostgreSQL aktif dan PostGIS telah terpasang. Buat database baru menggunakan salah satu cara berikut.

Melalui command line PostgreSQL:

```bash
createdb -U postgres smart_kos_db
```

Atau melalui `psql`:

```sql
CREATE DATABASE smart_kos_db;
```

Jika perintah `createdb` atau `psql` tidak ditemukan di Windows, jalankan SQL melalui pgAdmin Query Tool atau tambahkan folder `bin` PostgreSQL ke `PATH`.

## 4. Atur environment

Salin template konfigurasi:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Buka `.env`, lalu ganti `GANTI_PASSWORD` dengan password user PostgreSQL. Nama database, user, host, port, dan password pada `DATABASE_URL` harus sama dengan yang ada pada `DATABASE_DSN`.

Contoh untuk database lokal:

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:password_anda@localhost:5432/smart_kos_db
DATABASE_DSN=host=localhost port=5432 dbname=smart_kos_db user=postgres password=password_anda
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

Jika password mengandung karakter khusus seperti `@`, `:`, `/`, atau `#`, karakter tersebut harus di-URL-encode pada `DATABASE_URL`. `DATABASE_DSN` tetap menggunakan password aslinya.

Jangan commit file `.env` karena berisi kredensial. File tersebut sudah masuk `.gitignore`.

## 5. Buat tabel dan import dataset

Dengan virtual environment aktif, jalankan dari root project:

```bash
python scripts/init_postgis.py
```

Script tersebut akan:

1. mengaktifkan ekstensi PostGIS;
2. menjalankan migrasi `migrations/001_create_kos_table.sql`;
3. membuat tabel, trigger, view, dan index geospasial; serta
4. mengimpor `data/processed/data_kos_processed.csv`.

Script aman dijalankan ulang karena data menggunakan mekanisme upsert. User PostgreSQL yang digunakan harus memiliki izin untuk membuat extension dan objek database.

## 6. Jalankan backend

Mode development dengan auto-reload:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Atau:

```bash
python -m app.main
```

Tunggu sampai log menampilkan `STARTUP SELESAI`. Setelah aktif, buka:

- API root: <http://localhost:8000/>
- Health check: <http://localhost:8000/api/v1/health>
- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

## 7. Uji API

Health check dari PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

Contoh pencarian dari PowerShell:

```powershell
$body = @{
    kueri = "kos putri dekat kampus wifi AC"
    latitude = -6.1862
    longitude = 106.8348
    top_k = 5
    n_candidates = 20
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri http://localhost:8000/api/v1/search-kos `
    -ContentType "application/json" `
    -Body $body
```

Contoh yang sama menggunakan `curl` di Linux/macOS:

```bash
curl -X POST http://localhost:8000/api/v1/search-kos \
  -H 'Content-Type: application/json' \
  -d '{
    "kueri": "kos putri dekat kampus wifi AC",
    "latitude": -6.1862,
    "longitude": 106.8348,
    "top_k": 5,
    "n_candidates": 20
  }'
```

## Mengakses backend dari perangkat lain

Server sudah dijalankan pada `0.0.0.0`, tetapi client harus memakai alamat yang sesuai:

- Browser di komputer server: `http://localhost:8000`.
- Android Emulator: `http://10.0.2.2:8000`.
- Perangkat fisik di Wi-Fi yang sama: `http://<IP_LAN_KOMPUTER>:8000`.

Di Windows, lihat IP LAN dengan `ipconfig`. Izinkan Python atau port TCP 8000 pada firewall jika koneksi dari perangkat lain ditolak. Jangan mengekspos mode `DEBUG=true` dan CORS terbuka langsung ke internet.

## Troubleshooting

### `password authentication failed`

Periksa kembali user, password, port, dan nama database pada kedua variabel database di `.env`.

### Error validasi pada `DEBUG`

`DEBUG` harus berisi boolean `true` atau `false`. Nilai seperti `release`, `development`, atau `production` tidak dapat dibaca oleh konfigurasi saat ini.

### `extension "postgis" is not available`

PostGIS belum terpasang pada server PostgreSQL. Di Windows, pasang melalui Stack Builder. Di Ubuntu/Debian, pasang paket PostGIS yang sesuai dengan versi PostgreSQL, lalu ulangi script inisialisasi.

### `ModuleNotFoundError`

Pastikan virtual environment aktif, jalankan perintah dari root project, lalu ulangi:

```bash
pip install -r requirements.txt
```

### FAISS index atau model RF tidak ditemukan

Pastikan folder `faiss_index/` dan `models_ml/` ikut ter-clone. Jika indeks IndoBERT perlu dibangun ulang:

```bash
python scripts/rebuild_faiss_indobert.py
```

### Startup lama atau terlihat berhenti saat memuat IndoBERT

Pada startup pertama, Transformers mengunduh dan menyimpan model IndoBERT ke cache lokal. Tunggu unduhan selesai dan pastikan koneksi internet serta ruang disk mencukupi. Startup berikutnya menggunakan cache tersebut.

### Port 8000 sudah digunakan

Jalankan server pada port lain dan sesuaikan URL client:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Struktur penting

```text
app/                 FastAPI, endpoint, konfigurasi, dan service
stki/                preprocessing dan hybrid search engine
scripts/             migrasi, import data, dan pembangunan indeks
migrations/          skema PostgreSQL/PostGIS
data/                dataset raw, augmented, dan processed
faiss_index/         indeks vektor IndoBERT
models_ml/           model Random Forest dan scaler
requirements.txt     dependensi Python
.env.example         template konfigurasi lokal
```

## Persiapan sebelum upload ke GitHub

Pastikan file rahasia dan file lokal tidak ikut masuk commit:

```bash
git status
git check-ignore .env .venv
```

Jika `__pycache__`, file `.pyc`, `.env`, atau `.venv` pernah terlanjur masuk Git index, hapus hanya dari index lalu commit ulang:

```bash
git rm -r --cached --ignore-unmatch .env .venv
git rm -r --cached --ignore-unmatch "**/__pycache__" "**/*.pyc"
git add .
```
