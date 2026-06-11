import logging
import re
from typing import Optional

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

logger = logging.getLogger(__name__)


class IndonesianTextPreprocessor:
    """
    Pipeline preprocessing teks Bahasa Indonesia menggunakan library Sastrawi.

    Urutan 4 langkah wajib (TIDAK boleh diubah urutannya):
        1. Normalisasi  → lowercase, hapus URL dan emoji Unicode
        2. Pembersihan  → hapus angka, tanda baca, karakter non-alfabet
        3. Stopword removal → hapus kata hubung (kamus 757+ kata Sastrawi)
        4. Stemming     → reduksi morfologi ke kata dasar (algoritma ECS)

    Alasan urutan: pembersihan HARUS sebelum stemming agar stemmer tidak
    menerima token berisi karakter non-alfabet yang merusak kalkulasi
    algoritma Enhanced Confix Stripping (ECS) Nazief-Adriani.

    Contoh transformasi lengkap:
        Input : "Kos putri dengan kamar mandi dalam & AC, wifi kencang!"
        Step 1: "kos putri dengan kamar mandi dalam  ac wifi kencang"
        Step 2: "kos putri dengan kamar mandi dalam  ac wifi kencang"
        Step 3: "kos putri kamar mandi ac wifi kencang" (hapus 'dengan')
        Output: "kos putri kamar mandi ac wifi kencang" (sudah terstemming)
    """

    def __init__(self) -> None:
        """
        Inisialisasi komponen Sastrawi.
        Dilakukan SEKALI saja karena loading kamus dari disk cukup mahal (~1-2 detik).
        Instansiasi ulang di setiap request/baris = pemborosan resource.
        """
        logger.info("Menginisialisasi IndonesianTextPreprocessor...")

        # Stemmer menggunakan algoritma Enhanced Confix Stripping (ECS)
        stemmer_factory = StemmerFactory()
        self._stemmer = stemmer_factory.create_stemmer()

        # Stopword remover dengan kamus 757+ kata Bahasa Indonesia
        stopword_factory = StopWordRemoverFactory()
        self._stopword_remover = stopword_factory.create_stop_word_remover()

        # Pola regex dikompilasi sekali untuk efisiensi batch processing
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+", re.IGNORECASE
        )
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Simbol & piktogram
            "\U0001F680-\U0001F6FF"  # Transport & peta
            "\U0001F1E0-\U0001F1FF"  # Bendera negara
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        self._non_alpha_pattern = re.compile(r"[^a-zA-Z\s]")
        self._whitespace_pattern = re.compile(r"\s+")

        logger.info("IndonesianTextPreprocessor siap digunakan.")

    def _normalize(self, text: str) -> str:
        """
        Langkah 1 — Normalisasi.
        Mengubah ke huruf kecil, menghapus URL dan emoji.
        URL harus dihapus sebelum langkah lain karena '://' dan '.' akan
        menjadi token yang merusak kualitas stemming.
        """
        text = text.lower()
        text = self._url_pattern.sub(" ", text)
        text = self._emoji_pattern.sub(" ", text)
        return text

    def _clean(self, text: str) -> str:
        """
        Langkah 2 — Pembersihan karakter.
        Menghapus angka, tanda baca, dan karakter non-alfabet.
        Menormalisasi spasi ganda, tab, dan newline menjadi satu spasi.
        """
        text = self._non_alpha_pattern.sub(" ", text)
        text = self._whitespace_pattern.sub(" ", text).strip()
        return text

    def _remove_stopwords(self, text: str) -> str:
        """
        Langkah 3 — Penghapusan stopwords Bahasa Indonesia.
        Kamus Sastrawi mencakup 757+ kata umum: 'yang', 'dan', 'dengan',
        'untuk', 'adalah', 'ini', 'itu', dll.
        """
        return self._stopword_remover.remove(text)

    def _stem(self, text: str) -> str:
        """
        Langkah 4 — Stemming morfologi Bahasa Indonesia.
        Algoritma ECS (Enhanced Confix Stripping, turunan Nazief-Adriani).
        Contoh: 'memberikan' → 'beri', 'pelayanan' → 'layan',
                'kebersihan' → 'bersih', 'pemrosesan' → 'proses'.
        """
        return self._stemmer.stem(text)

    def preprocess(self, text: Optional[str]) -> str:
        """
        Eksekusi pipeline preprocessing lengkap secara berurutan.

        Args:
            text: Teks mentah Bahasa Indonesia.

        Returns:
            Teks bersih setelah 4 langkah pipeline.
            String kosong jika input None, bukan string, atau gagal diproses.
        """
        if not text or not isinstance(text, str):
            return ""

        try:
            text = self._normalize(text)
            text = self._clean(text)
            text = self._remove_stopwords(text)
            text = self._stem(text)
            return text
        except Exception as exc:
            logger.warning(
                f"Gagal memproses teks '{str(text)[:50]}...' | Error: {exc}"
            )
            return ""

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """
        Preprocessing batch untuk efisiensi pemrosesan CSV berukuran besar.
        Urutan output dijamin sama dengan urutan input.

        Args:
            texts: List teks mentah Bahasa Indonesia (satu elemen per dokumen).

        Returns:
            List teks yang sudah diproses (indeks berkorespondensi dengan input).
        """
        total = len(texts)
        logger.info(f"Memulai batch preprocessing untuk {total} dokumen...")
        results = [self.preprocess(t) for t in texts]

        # Validasi kualitas output
        empty_count = sum(1 for r in results if not r)
        logger.info(
            f"Preprocessing selesai. Berhasil: {total - empty_count}/{total}. "
            f"Output kosong: {empty_count}."
        )
        return results