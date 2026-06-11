"""
Script augmentasi dataset kos-kosan Jakarta.

Menambahkan 5 kolom baru ke CSV dataset asli:
    deskripsi_promosi : Teks promosi panjang dari pemilik kos (bahasa alami)
    ulasan_penghuni   : Teks ulasan dari penghuni sebelumnya
    foto_path         : Path relatif ke file foto kamar
    latitude          : Koordinat GPS lintang (Gaussian noise sekitar pusat kota)
    longitude         : Koordinat GPS bujur

Dijalankan SEKALI sebagai langkah pertama data engineering pipeline.
Sepenuhnya reproducible karena menggunakan numpy random seed tetap (default: 42).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# KONSTANTA: Koordinat pusat setiap kota sebagai anchor GPS
# Sumber: titik tengah area administratif Jakarta (WGS84 / EPSG:4326)
# =============================================================================
CITY_COORD_ANCHORS: dict[str, dict[str, float]] = {
    "Jakarta Pusat":   {"lat": -6.1862, "lon": 106.8348},
    "Jakarta Selatan": {"lat": -6.2615, "lon": 106.8106},
    "Jakarta Timur":   {"lat": -6.2250, "lon": 106.9004},
    "Jakarta Barat":   {"lat": -6.1744, "lon": 106.7629},
    "Jakarta Utara":   {"lat": -6.1213, "lon": 106.8649},
}

# Std dev Gaussian noise koordinat: 0.025° ≈ 2.75 km (mensimulasikan sebaran kos nyata)
GPS_SCATTER_STD: float = 0.025

# =============================================================================
# TEMPLATE TEKS: Basis generasi teks sintetis yang bervariasi per baris
# =============================================================================

DESKRIPSI_TEMPLATES: list[str] = [
    (
        "Kos {tipe} strategis di {wilayah}, {kota}. Kamar {ukuran}m² dilengkapi "
        "{fasilitas_str}. Lokasi hanya {jarak_kampus} km dari kampus dan "
        "{jarak_transport} km dari moda transportasi umum. Suasana bersih, "
        "aman, dan nyaman. Cocok untuk mahasiswa maupun pekerja muda yang "
        "menginginkan hunian berkualitas dengan harga terjangkau."
    ),
    (
        "Tersedia kamar kos {tipe} di kawasan {wilayah}, {kota}. Luas kamar "
        "{ukuran}m², fasilitas: {fasilitas_str}. Dekat kampus ({jarak_kampus} km) "
        "dan transportasi umum ({jarak_transport} km). Lingkungan tenang, "
        "pemilik kos ramah dan responsif. Harga sudah mencakup biaya perawatan "
        "fasilitas bersama. Ideal untuk pelajar yang baru merantau."
    ),
    (
        "Kos eksklusif {tipe} area {wilayah}! Fasilitas lengkap: {fasilitas_str}. "
        "Kamar {ukuran}m² dengan pencahayaan dan ventilasi optimal. Akses mudah "
        "ke pusat bisnis {kota}. Penjaga aktif 24 jam, CCTV terpasang di area "
        "umum. Hanya {jarak_kampus} km ke kampus terdekat. Daftar sekarang!"
    ),
    (
        "Hunian nyaman di jantung {kota}. Kos {tipe} dengan {fasilitas_str}. "
        "Ukuran kamar {ukuran}m², cukup untuk bekerja dan beristirahat. "
        "Bebas banjir, akses 24 jam. Jarak ke transportasi umum hanya "
        "{jarak_transport} km. Cocok untuk profesional muda yang aktif."
    ),
    (
        "Kos bersih dan terawat di {wilayah}, {kota}. Kamar {tipe} berukuran "
        "{ukuran}m² tersedia dengan fasilitas {fasilitas_str}. Pengelolaan "
        "profesional, kontrak fleksibel per bulan. Jarak {jarak_kampus} km "
        "dari kampus, {jarak_transport} km dari halte atau stasiun. "
        "Survei gratis, hubungi pemilik sekarang!"
    ),
]

ULASAN_POSITIF: list[str] = [
    (
        "Sudah {durasi} bulan di sini, sangat puas! Kamar bersih dan wangi. "
        "{catatan} Pemilik kos sigap menangani keluhan. Sangat direkomendasikan!"
    ),
    (
        "Lokasi sangat strategis, ke mana-mana mudah dijangkau. {catatan} "
        "Harga sangat worth it untuk fasilitasnya. Teman sesama penghuni juga asik."
    ),
    (
        "Kos paling nyaman selama kuliah di Jakarta. {catatan} "
        "Lingkungan bersih dan aman, tidak pernah ada masalah keamanan. "
        "Sangat recommended untuk mahasiswa baru!"
    ),
    (
        "Baru {durasi} minggu tapi sudah betah banget. {catatan} "
        "Air bersih dan lancar, listrik stabil, tidak pernah mati. Harga sepadan."
    ),
    (
        "Sudah {durasi} tahun tinggal di sini dan tidak mau pindah. {catatan} "
        "Lingkungan aman dan nyaman, penghuni dipilih secara selektif."
    ),
]

ULASAN_NETRAL: list[str] = [
    (
        "Cukup nyaman untuk kebutuhan sehari-hari. {catatan} "
        "Harga standar untuk area sini. Cocok untuk yang budget terbatas."
    ),
    (
        "Oke secara keseluruhan. {catatan} "
        "Ada plus minusnya, tapi tidak mengecewakan. Lumayan untuk jangka panjang."
    ),
    (
        "Lumayan sesuai harga yang dibayar. {catatan} "
        "Respons pemilik kadang agak lambat, tapi fasilitas tetap terjaga."
    ),
]


def _build_fasilitas_str(row: pd.Series) -> str:
    """
    Menyusun deskripsi fasilitas dalam format kalimat Bahasa Indonesia yang alami.
    Digunakan untuk mengisi placeholder {fasilitas_str} pada template deskripsi.

    Args:
        row: Satu baris DataFrame kos dengan kolom fasilitas binary.

    Returns:
        String fasilitas natural: "AC, WiFi, dan kamar mandi dalam"
    """
    items: list[str] = []
    if row.get("ac", 0) == 1:
        items.append("AC")
    if row.get("kamar_mandi_dalam", 0) == 1:
        items.append("kamar mandi dalam")
    if row.get("wifi", 0) == 1:
        items.append("WiFi")
    if row.get("listrik_include", 0) == 1:
        items.append("listrik termasuk")
    if row.get("parkir", 0) == 1:
        items.append("parkir")
    if row.get("dapur", 0) == 1:
        items.append("dapur bersama")
    if row.get("laundry", 0) == 1:
        items.append("laundry")
    if row.get("security_24jam", 0) == 1:
        items.append("security 24 jam")

    if not items:
        return "fasilitas dasar"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} dan {items[1]}"
    return ", ".join(items[:-1]) + f", dan {items[-1]}"


def _build_catatan_fasilitas(row: pd.Series) -> str:
    """Menghasilkan catatan singkat fasilitas untuk template ulasan."""
    notes: list[str] = []
    if row.get("wifi", 0) == 1:
        notes.append("WiFi kencang dan stabil.")
    if row.get("ac", 0) == 1:
        notes.append("AC dingin, tidak berisik.")
    if row.get("kamar_mandi_dalam", 0) == 1:
        notes.append("Kamar mandi pribadi sangat nyaman.")
    if row.get("security_24jam", 0) == 1:
        notes.append("Keamanan 24 jam membuat tenang.")
    return " ".join(notes[:2]) if notes else "Fasilitas standar terpenuhi."


def _generate_gps(city: str, rng: np.random.Generator) -> tuple[float, float]:
    """
    Menghasilkan koordinat GPS realistis dengan Gaussian noise di sekitar pusat kota.

    Noise Gaussian (std=0.025°) mensimulasikan sebaran lokasi kos yang sesungguhnya
    di dalam satu wilayah kota. Radius ~2.75 km dari pusat kota per 1 std dev.

    Args:
        city: Nama kota (kunci CITY_COORD_ANCHORS).
        rng : NumPy Generator untuk hasil reproducible (dari seed tetap).

    Returns:
        Tuple (latitude, longitude) presisi 6 desimal.
    """
    anchor = CITY_COORD_ANCHORS.get(city, CITY_COORD_ANCHORS["Jakarta Pusat"])
    lat = round(anchor["lat"] + rng.normal(0, GPS_SCATTER_STD), 6)
    lon = round(anchor["lon"] + rng.normal(0, GPS_SCATTER_STD), 6)
    return lat, lon


def augment_dataset(
    input_path: str,
    output_path: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Fungsi utama augmentasi dataset kos-kosan Jakarta.

    Args:
        input_path  : Path ke CSV dataset asli (data_kos_jakarta.csv).
        output_path : Path output CSV yang sudah diaugmentasi.
        random_seed : Seed untuk reproducibility (default: 42).

    Returns:
        DataFrame yang sudah diaugmentasi dengan 5 kolom baru.

    Raises:
        FileNotFoundError: Jika file input tidak ditemukan.
        ValueError        : Jika kolom wajib tidak ada di dataset.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {input_path}\n"
            "Pastikan data_kos_jakarta.csv sudah ditaruh di data/raw/"
        )

    logger.info(f"Membaca dataset: {input_path}")
    df: pd.DataFrame = pd.read_csv(input_path)

    # Validasi kolom wajib
    required_cols = ["id_kos", "kota", "wilayah", "tipe_kos", "ukuran_kamar"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di dataset: {missing}")

    logger.info(f"Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom.")

    rng = np.random.default_rng(random_seed)
    durasi_opts = ["2", "3", "6", "8", "12", "18", "24"]

    deskripsi_list: list[str] = []
    ulasan_list: list[str] = []
    foto_list: list[str] = []
    lat_list: list[float] = []
    lon_list: list[float] = []

    for idx, row in df.iterrows():
        # --- 1. Koordinat GPS (Gaussian noise di sekitar pusat kota) ---
        lat, lon = _generate_gps(str(row.get("kota", "Jakarta Pusat")), rng)
        lat_list.append(lat)
        lon_list.append(lon)

        # --- 2. Path foto kamar ---
        foto_list.append(
            f"assets/kos_photos/kos_{int(row.get('id_kos', idx + 1)):04d}.jpg"
        )

        # --- 3. Deskripsi promosi (pilih template acak, isi dengan data baris) ---
        tmpl_idx = rng.integers(0, len(DESKRIPSI_TEMPLATES))
        deskripsi_list.append(
            DESKRIPSI_TEMPLATES[tmpl_idx].format(
                tipe=str(row.get("tipe_kos", "Campur")).lower(),
                wilayah=str(row.get("wilayah", "Jakarta")),
                kota=str(row.get("kota", "Jakarta")),
                ukuran=str(row.get("ukuran_kamar", 9)),
                fasilitas_str=_build_fasilitas_str(row),
                jarak_kampus=str(
                    round(float(row.get("jarak_ke_kampus_km", 2.0)), 1)
                ),
                jarak_transport=str(
                    round(float(row.get("jarak_ke_transportasi_km", 1.0)), 1)
                ),
            )
        )

        # --- 4. Ulasan penghuni (positif jika rating >= 4.0, netral jika di bawah) ---
        rating = float(row.get("rating", 3.5))
        templates = ULASAN_POSITIF if rating >= 4.0 else ULASAN_NETRAL
        ulasan_tmpl = templates[rng.integers(0, len(templates))]
        ulasan_list.append(
            ulasan_tmpl.format(
                durasi=str(rng.choice(durasi_opts)),
                catatan=_build_catatan_fasilitas(row),
            )
        )

    # Tambahkan 5 kolom baru ke DataFrame
    df["deskripsi_promosi"] = deskripsi_list
    df["ulasan_penghuni"] = ulasan_list
    df["foto_path"] = foto_list
    df["latitude"] = lat_list
    df["longitude"] = lon_list

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"✅ Augmentasi selesai. Output: {output_path}")
    logger.info(f"   Baris: {len(df)} | Kolom: {len(df.columns)}")
    logger.info(
        f"   Kolom baru: deskripsi_promosi, ulasan_penghuni, "
        f"foto_path, latitude, longitude"
    )
    return df


if __name__ == "__main__":
    augment_dataset(
        input_path=settings.data_raw_path,
        output_path=settings.data_augmented_path,
    )