"""
stki/indexer.py — FAISS Vector Index Manager
Smart-Kos Hybrid Engine — Tahap 2 (IndoBERT Version)

═══════════════════════════════════════════════════════════════
PERUBAHAN DARI TAHAP 1 (TF-IDF → IndoBERT):

  DIHAPUS  : TfidfVectorizer, vectorizer_path
  DITAMBAH : IndoBERTEmbedder sebagai dependency injection
  TETAP    : IndexFlatIP, interface publik (build/search/load)
  DIM      : 5000 (TF-IDF) → 768 (IndoBERT fixed)

Prinsip Open/Closed (SOLID) terpenuhi:
  Interface build_index(), search(), load_index() tidak berubah.
  Hanya implementasi internal _vectorize() yang diganti.
═══════════════════════════════════════════════════════════════
"""

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from stki.embedder import IndoBERTEmbedder

logger = logging.getLogger(__name__)


class KosVectorIndexer:
    """
    Manager FAISS IndoBERT Vector Index untuk Smart-Kos.

    Arsitektur FAISS:
        IndexFlatIP — Exact brute-force inner product search.
        Kompleksitas: O(N × D) per kueri.
        Untuk dataset 909 kos: sangat cepat (< 1ms per query).
        Upgrade ke IndexIVFFlat jika N > 100.000.

    Setelah L2-normalization: inner_product = cosine_similarity.
    Sehingga hasil search terurut berdasarkan kesamaan semantik (0.0–1.0).

    Attributes:
        index_path : Path FAISS binary index (.index)
        id_map_path: Path mapping posisi FAISS (int) → id_kos asli (int)
        embedder   : IndoBERTEmbedder untuk vektorisasi dokumen dan kueri
    """

    EMBEDDING_DIM: int = 768  # Fixed: IndoBERT BERT-base hidden dimension

    def __init__(
        self,
        index_path: str,
        id_map_path: str,
        embedder: "IndoBERTEmbedder",
    ) -> None:
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.embedder = embedder

        # State runtime — diisi oleh build_index() atau load_index()
        self.index: faiss.Index | None = None
        self.id_map: list[int] = []

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def build_index(
        self,
        documents: list[str],
        kos_ids: list[int],
        encode_batch_size: int = 16,
    ) -> None:
        """
        Bangun FAISS IndexFlatIP dari list dokumen via IndoBERT encoding.

        Proses:
            1. Encode semua dokumen via IndoBERT (mean pooling + L2-norm)
               → matrix float32 (N, 768)
            2. Inisialisasi FAISS IndexFlatIP(768)
            3. Add semua vektor ke index (O(1) insert untuk IndexFlat)
            4. Simpan index + id_map ke disk

        PERINGATAN PERFORMA DI CPU:
            Encoding 909 dokumen via IndoBERT (CPU Intel i5/i7):
            ~30–60 menit (batch_size=8–16). SANGAT DISARANKAN menggunakan
            Google Colab T4 GPU (~3 menit) lalu download hasilnya.

        Args:
            documents        : List 'deskripsi_clean' dari CSV processed.
            kos_ids          : List id_kos berkorespondensi (indeks harus sinkron).
            encode_batch_size: Batch size untuk IndoBERT encoding.
                               Kurangi ke 4–8 jika memory CPU habis.

        Raises:
            ValueError: Jika len(documents) ≠ len(kos_ids) atau list kosong.
        """
        if len(documents) != len(kos_ids):
            raise ValueError(
                f"Mismatch: documents={len(documents)}, "
                f"kos_ids={len(kos_ids)}. Harus sama."
            )
        if not documents:
            raise ValueError("List documents tidak boleh kosong.")

        logger.info(
            f"Membangun FAISS IndoBERT index: "
            f"{len(documents)} dokumen, batch_size={encode_batch_size}..."
        )

        # Encode semua dokumen → matrix (N, 768), L2-normalized
        vectors = self.embedder.encode(
            documents,
            batch_size=encode_batch_size,
            normalize_l2=True,
            show_progress=True,
        )

        # Inisialisasi dan isi FAISS index
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self.index.add(vectors)
        self.id_map = list(kos_ids)

        logger.info(
            f"FAISS index dibangun: {self.index.ntotal} vektor "
            f"× {self.EMBEDDING_DIM} dimensi."
        )
        self._persist()

    def search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Cari top-k kos paling relevan secara semantik terhadap kueri.

        Args:
            query_text: Teks kueri (sudah dipreproses Sastrawi + siap IndoBERT).
            top_k     : Jumlah hasil yang dikembalikan.

        Returns:
            List dict [{"id_kos": int, "score": float}] terurut descending.
            Score = cosine similarity [0.0–1.0].

        Raises:
            RuntimeError: Jika load_index() belum pernah dipanggil.
        """
        if self.index is None:
            raise RuntimeError(
                "FAISS index belum dimuat. "
                "Panggil load_index() atau build_index() terlebih dahulu."
            )

        # Encode kueri: shape (1, 768), L2-normalized
        query_vector = self.embedder.encode_single(query_text, normalize_l2=True)

        # FAISS search: kembalikan (scores, indices) shape (1, top_k)
        scores, indices = self.index.search(query_vector, top_k)

        return [
            {"id_kos": self.id_map[int(idx)], "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx != -1  # FAISS return -1 jika hasil < top_k
        ]

    def load_index(self) -> None:
        """
        Muat FAISS index dan id_map dari disk ke memori.
        Dipanggil satu kali saat FastAPI startup (lifespan handler, Tahap 3).

        Raises:
            FileNotFoundError: Jika file index tidak ditemukan.
                Solusi: python scripts/rebuild_faiss_indobert.py
        """
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS IndoBERT index tidak ditemukan: {self.index_path}\n"
                "Solusi: python scripts/rebuild_faiss_indobert.py\n"
                "        (Disarankan di Google Colab untuk performa optimal)"
            )

        logger.info(f"Memuat FAISS index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        with open(self.id_map_path, "rb") as f:
            self.id_map = pickle.load(f)

        logger.info(
            f"FAISS index dimuat: {self.index.ntotal} vektor "
            f"× {self.EMBEDDING_DIM} dim."
        )

    def _persist(self) -> None:
        """Simpan FAISS index dan id_map ke disk."""
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"FAISS index disimpan: {self.index_path}")

        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)
        logger.info(f"ID map disimpan: {self.id_map_path}")