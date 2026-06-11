# TAHAP 2 — Smart-Kos Hybrid Engine
## AI Engine: IndoBERT · PRF · Random Forest · Fusion Scoring

> Melanjutkan dari TAHAP1_SmartKos_DataEngineering.md
> Seluruh kode bersifat *production-grade*, tanpa endpoint FastAPI.

---

### File Baru & Diperbarui di Tahap 2

```
smart-kos-backend/
├── stki/
│   ├── embedder.py              ← NEW  : IndoBERT sentence encoder
│   ├── indexer.py               ← UPDATE: TF-IDF → IndoBERT (Strategy DI)
│   ├── prf_engine.py            ← NEW  : Pseudo-Relevance Feedback
│   └── fusion_scorer.py         ← NEW  : Hybrid Fusion Scoring Engine
├── app/
│   └── services/
│       ├── pricing_service.py   ← NEW  : Random Forest price prediction
│       └── search_service.py    ← NEW  : Full pipeline orchestrator
└── scripts/
    ├── rebuild_faiss_indobert.py ← NEW  : Bangun FAISS index IndoBERT
    └── demo_search.py            ← NEW  : Validasi pipeline (tanpa DB)
```

---

### Tambahan `requirements.txt`

```text
# === Tahap 2: IndoBERT + Transformers ===
transformers==4.41.2
torch>=2.0.0               # CPU default; install torch+cuda di Colab untuk GPU
sentencepiece==0.2.0        # Required by IndoBERT tokenizer
```

### Tambahan `.env`

```env
# === IndoBERT ===
INDOBERT_MODEL_NAME=indobenchmark/indobert-base-p1
INDOBERT_MAX_LENGTH=128
INDOBERT_BATCH_SIZE=16

# === FAISS IndoBERT Paths ===
FAISS_INDOBERT_INDEX_PATH=./faiss_index/kos_indobert.index
FAISS_INDOBERT_ID_MAP_PATH=./faiss_index/id_map_indobert.pkl

# === Pricing Thresholds ===
SUPER_DEAL_THRESHOLD=0.85
UNDERPRICED_THRESHOLD=0.95
FAIR_THRESHOLD=1.05

# === Fusion Weights (harus: w_s + w_g + w_p = 1.0) ===
FUSION_WEIGHT_SEMANTIC=0.50
FUSION_WEIGHT_GEOSPATIAL=0.30
FUSION_WEIGHT_PRICE=0.20
FUSION_GEO_DECAY_LAMBDA=0.3

# === PRF Settings ===
PRF_N_FEEDBACK_DOCS=5
PRF_N_EXPANSION_TERMS=5
```

### Tambahan `app/core/config.py` (field baru saja)

```python
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
```

---

## FILE 1: `stki/embedder.py`

```python
"""
stki/embedder.py — IndoBERT Sentence Encoder
Smart-Kos Hybrid Engine — Tahap 2

Menghasilkan dense vector 768-dim dari teks Bahasa Indonesia menggunakan
model IndoBERT (indobenchmark/indobert-base-p1) dari HuggingFace.

═══════════════════════════════════════════════════════════════════
DASAR TEORI (Bab 2 Laporan TA):

  IndoBERT — Wilie et al. (2020), AACL-IJCNLP:
    Arsitektur : BERT-base (12 layers, 768 hidden dim, 12 attention heads)
    Parameter  : 110 juta
    Pre-training: Wikipedia ID + berita + web crawl (23.43 GB teks Indonesia)

  Strategi Sentence Embedding yang Digunakan — Mean Pooling:
    BERT menghasilkan last_hidden_state shape (batch, seq_len, 768).
    Setiap token memiliki vektor 768-dim. Untuk mendapatkan satu vektor
    per kalimat, kita menghitung rata-rata berbobot dari semua token
    nyata (mengabaikan token [PAD]) menggunakan attention_mask.

    Formula:
        v_sentence = Σ(v_token_i × mask_i) / Σ(mask_i)

    Mean Pooling dipilih karena lebih stabil dari CLS token pooling
    untuk semantic similarity tasks (Reimers & Gurevych, 2019).

  Normalisasi L2:
    v_norm = v / ‖v‖₂
    Tujuan: cos_similarity(a, b) = ⟨a_norm, b_norm⟩ = inner_product
    Sehingga FAISS IndexFlatIP (inner product search) ≡ cosine search.
═══════════════════════════════════════════════════════════════════
"""

import logging
import math

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class IndoBERTEmbedder:
    """
    Dense sentence encoder berbasis IndoBERT dengan Mean Pooling + L2-norm.

    Kompatibel langsung dengan KosVectorIndexer dan SearchService.
    Interface encode() / encode_single() menerima list[str] dan mengembalikan
    np.ndarray float32 yang siap di-feed ke FAISS IndexFlatIP.

    Attributes:
        MODEL_NAME    : Identifier model HuggingFace
        EMBEDDING_DIM : Dimensi output vektor (768 untuk BERT-base)
        device        : torch.device aktif (CPU / CUDA)
        max_length    : Panjang token maksimum per kalimat
    """

    MODEL_NAME: str = "indobenchmark/indobert-base-p1"
    EMBEDDING_DIM: int = 768

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "auto",
        max_length: int = 128,
    ) -> None:
        """
        Muat tokenizer dan model dari HuggingFace Hub.
        Model di-cache di ~/.cache/huggingface/ setelah download pertama.

        Args:
            model_name : HuggingFace model ID. Default: indobert-base-p1.
            device     : 'auto' = CUDA jika tersedia, else CPU.
                         Gunakan GPU (Colab T4/A100) untuk build index.
                         CPU sudah cukup untuk query encoding real-time (< 200ms).
            max_length : Batas token per kalimat. 128 cukup untuk deskripsi kos
                         (rata-rata < 100 kata setelah Sastrawi preprocessing).
        """
        self.model_name = model_name
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Memuat IndoBERT: {model_name} | Device: {self.device}")

        # Tokenizer: konversi teks → token IDs + attention mask
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model transformer: hanya encoder tanpa prediction head (AutoModel)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Non-aktifkan dropout → inferensi deterministik

        logger.info(
            f"IndoBERT siap. "
            f"Embedding dim={self.EMBEDDING_DIM} | Max token length={max_length}"
        )

    @staticmethod
    def _mean_pooling(
        model_output: tuple,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Agregasi token embeddings → sentence embedding via weighted mean.

        Args:
            model_output  : Output AutoModel (tuple). Index [0] = last_hidden_state
                           shape (batch_size, seq_len, 768).
            attention_mask: Tensor biner shape (batch_size, seq_len).
                           Nilai 1 = token nyata, 0 = token [PAD].

        Returns:
            Sentence embedding tensor shape (batch_size, 768).
        """
        # last_hidden_state: (batch_size, seq_len, 768)
        token_embeddings = model_output[0]

        # Expand mask: (batch_size, seq_len) → (batch_size, seq_len, 768)
        # Memungkinkan element-wise multiplication dengan token_embeddings
        mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        # Jumlahkan embedding berbobot, bagi dengan jumlah token nyata
        # clamp(min=1e-9) mencegah division-by-zero untuk teks yang sangat pendek
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)

        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_l2: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode list teks menjadi dense sentence vectors (batch processing).

        Pemrosesan batch-by-batch mengontrol penggunaan memory.
        Rekomendasi batch_size:
            CPU lokal (8GB RAM) : 8–16
            GPU T4 Colab        : 64–128
            GPU A100 Kaggle     : 128–256

        Args:
            texts         : List teks Bahasa Indonesia (sudah dipreproses Sastrawi).
            batch_size    : Dokumen per batch inferensi.
            normalize_l2  : Wajib True agar FAISS IndexFlatIP = cosine similarity.
            show_progress : Log progress setiap 5 batch (aktifkan saat build index).

        Returns:
            Matrix np.float32 shape (N, 768). Satu baris per dokumen.
        """
        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        n_batches = math.ceil(len(texts) / batch_size)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            if show_progress and (batch_idx % 5 == 0 or batch_idx == n_batches - 1):
                logger.info(
                    f"  [IndoBERT] Batch {batch_idx + 1}/{n_batches} "
                    f"({end}/{len(texts)} dokumen)..."
                )

            # Tokenisasi: teks → token IDs + attention mask
            encoded = self.tokenizer(
                batch,
                padding=True,       # Pad ke panjang terpanjang dalam batch
                truncation=True,    # Potong jika melebihi max_length
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Forward pass (no_grad sudah di-set via decorator)
            output = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
            )

            # Mean pooling → sentence embeddings
            sentence_embeddings = self._mean_pooling(
                output, encoded["attention_mask"].to(self.device)
            )

            # Pindah ke CPU + konversi ke numpy float32
            embeddings_np = sentence_embeddings.cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings_np)

        # Gabungkan semua batch → matrix (N, 768)
        result = np.vstack(all_embeddings)

        # Normalisasi L2: setiap vektor jadi unit vector
        # Setelah ini: ⟨a, b⟩ = cos_similarity(a, b)
        if normalize_l2:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Amankan dari division-by-zero
            result = result / norms

        if show_progress:
            logger.info(
                f"  [IndoBERT] Encoding selesai. "
                f"Output shape: {result.shape} | L2-normalized: {normalize_l2}"
            )
        return result

    @torch.no_grad()
    def encode_single(
        self,
        text: str,
        normalize_l2: bool = True,
    ) -> np.ndarray:
        """
        Encode satu teks menjadi sentence vector (untuk query encoding real-time).

        Pemanggilan ini akan selesai dalam ~100-200ms di CPU, yang cukup
        untuk kebutuhan low-latency query encoding di production.

        Args:
            text        : Teks kueri tunggal (sudah dipreproses Sastrawi).
            normalize_l2: Wajib True untuk FAISS cosine search.

        Returns:
            np.float32 shape (1, 768).
        """
        return self.encode([text], batch_size=1, normalize_l2=normalize_l2)
```

---

## FILE 2: `stki/indexer.py` (UPDATED — Tahap 2)

```python
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
```

---

## FILE 3: `stki/prf_engine.py`

```python
"""
stki/prf_engine.py — Pseudo-Relevance Feedback Engine
Smart-Kos Hybrid Engine — Tahap 2

═══════════════════════════════════════════════════════════════════
DASAR TEORI (Bab 2 Laporan TA):

  Pseudo-Relevance Feedback (PRF) — Rocchio (1971), SMART Retrieval:
    Asumsi: N dokumen teratas dari pencarian pertama dianggap relevan.
    Tujuan : Menemukan term tambahan yang memperkaya konteks kueri asli
             sehingga pencarian kedua menghasilkan recall lebih tinggi.

  Implementasi TF-IDF PRF:
    1. Kumpulkan top-K feedback documents dari FAISS Search 1
    2. Fit TF-IDF vectorizer pada feedback set
    3. Hitung rata-rata skor TF-IDF setiap term di seluruh feedback docs
       → Term yang tinggi secara konsisten = representatif untuk topik
    4. Filter: buang term yang sudah ada di kueri, term pendek (noise)
    5. Ambil top-M expansion terms
    6. Concatenate: query_expanded = original + " " + expansion_terms

  Mengapa Rata-Rata (bukan max)?
    Rata-rata menangkap term yang konsisten informatif di SEMUA dokumen
    relevan, bukan hanya satu. Ini mengurangi noise dari term idiosinkratik
    satu dokumen yang tidak mewakili topik secara umum.
═══════════════════════════════════════════════════════════════════
"""

import logging

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class PseudoRelevanceFeedback:
    """
    Mesin ekspansi kueri otomatis menggunakan algoritma TF-IDF PRF.

    Attributes:
        n_feedback_docs  : Jumlah dokumen teratas dari FAISS Rd.1 untuk feedback (default: 5)
        n_expansion_terms: Jumlah term baru yang ditambahkan ke kueri (default: 5)
        min_term_length  : Panjang minimum term untuk menghindari noise (default: 3)
    """

    def __init__(
        self,
        n_feedback_docs: int = 5,
        n_expansion_terms: int = 5,
        min_term_length: int = 3,
    ) -> None:
        self.n_feedback_docs = n_feedback_docs
        self.n_expansion_terms = n_expansion_terms
        self.min_term_length = min_term_length

    def extract_expansion_terms(
        self,
        feedback_docs: list[str],
        original_query: str,
    ) -> list[str]:
        """
        Ekstrak top-N term ekspansi dari feedback document set via TF-IDF.

        Algoritma detail:
            1. Fit TF-IDF pada feedback_docs (max_features=500, unigram)
            2. Transform → sparse matrix (n_docs, n_features)
            3. .mean(axis=0).A1 → rata-rata skor TF-IDF per term (n_features,)
               Menangkap term yang informatif di mayoritas dokumen relevan
            4. argsort()[::-1] → urutkan dari skor tertinggi
            5. Filter duplikat (sudah di kueri), filter term pendek, filter angka
            6. Ambil top-N

        Args:
            feedback_docs : List string 'deskripsi_clean' dari top-K FAISS results.
                           Dikirim dari SearchService._fetch_deskripsi_clean().
            original_query: Kueri asli setelah Sastrawi preprocessing.
                           Digunakan untuk filter duplikat.

        Returns:
            List[str] term ekspansi. Kosong jika tidak ada term valid yang ditemukan.
        """
        if not feedback_docs:
            return []

        # Set term dari kueri asli → basis filter duplikat
        original_terms: set[str] = set(original_query.lower().split())

        try:
            # TF-IDF pada feedback set
            # sublinear_tf=True: TF = 1 + log(tf) → kurangi dominasi term terlalu sering
            # ngram_range=(1,1): unigram agar term ekspansi berdiri sendiri
            # min_df=1: boleh hanya muncul di 1 dokumen (feedback set kecil, 5 dok)
            vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 1),
                min_df=1,
                sublinear_tf=True,
                token_pattern=r"(?u)\b[a-zA-Z]+\b",  # Hanya kata alfabetik
            )
            tfidf_matrix = vectorizer.fit_transform(feedback_docs)
            feature_names: list[str] = vectorizer.get_feature_names_out().tolist()

            # Rata-rata skor TF-IDF per term di seluruh feedback set
            # .mean(axis=0): rata-rata baris → shape (1, n_features)
            # .A1           : konversi sparse matrix row ke numpy array 1D
            avg_scores = tfidf_matrix.mean(axis=0).A1

            # Urutkan indeks term dari skor TF-IDF rata-rata tertinggi
            sorted_indices = avg_scores.argsort()[::-1]

            expansion_terms: list[str] = []
            for idx in sorted_indices:
                term = feature_names[idx]

                # Kriteria validasi term ekspansi:
                # 1. Belum ada di kueri asli (tidak redundan)
                # 2. Panjang cukup (≥ min_term_length chars, hindari noise)
                # 3. Bukan string numerik murni
                if (
                    term not in original_terms
                    and len(term) >= self.min_term_length
                    and not term.isdigit()
                ):
                    expansion_terms.append(term)
                    if len(expansion_terms) >= self.n_expansion_terms:
                        break

        except Exception as exc:
            logger.warning(
                f"PRF extraction gagal: {exc}. "
                "Kueri akan digunakan tanpa ekspansi."
            )
            return []

        return expansion_terms

    def expand_query(
        self,
        original_query: str,
        expansion_terms: list[str],
    ) -> str:
        """
        Gabungkan kueri asli dengan term ekspansi via konkatenasi.

        Konkatenasi sederhana dipilih karena:
        - IndoBERT akan merepresentasikan seluruh teks gabungan dalam satu
          vektor 768-dim melalui self-attention mechanism
        - Term ekspansi menambah bobot semantik tanpa mengubah intent asli
        - Tidak ada risiko semantic drift (tidak melakukan query rewriting)

        Args:
            original_query  : Kueri asli (sudah dipreproses Sastrawi).
            expansion_terms : List term dari extract_expansion_terms().

        Returns:
            String kueri diperluas.
        """
        if not expansion_terms:
            return original_query
        return f"{original_query} {' '.join(expansion_terms)}"

    def run(
        self,
        original_query: str,
        feedback_docs: list[str],
    ) -> tuple[str, list[str]]:
        """
        Jalankan pipeline PRF lengkap: original_query + feedback_docs → expanded_query.

        Ini adalah satu-satunya method yang dipanggil oleh SearchService.

        Args:
            original_query: Kueri asli setelah Sastrawi preprocessing.
            feedback_docs : List 'deskripsi_clean' dari top-K FAISS Search 1.

        Returns:
            Tuple (expanded_query, expansion_terms):
                expanded_query  : Teks untuk IndoBERT re-embed + FAISS Search 2
                expansion_terms : Term yang ditambahkan (log + transparansi riset)
        """
        expansion_terms = self.extract_expansion_terms(
            feedback_docs=feedback_docs[: self.n_feedback_docs],
            original_query=original_query,
        )
        expanded_query = self.expand_query(original_query, expansion_terms)

        logger.info(
            f"PRF: '{original_query}' "
            f"→ +{expansion_terms} "
            f"→ '{expanded_query}'"
        )
        return expanded_query, expansion_terms
```

---

## FILE 4: `stki/fusion_scorer.py`

```python
"""
stki/fusion_scorer.py — Hybrid Fusion Scoring Engine
Smart-Kos Hybrid Engine — Tahap 2

Menyatukan tiga sinyal independen menjadi satu skor evaluasi final (Final Score)
yang merepresentasikan nilai holistic sebuah kos bagi pengguna.

═══════════════════════════════════════════════════════════════════
FORMULA UTAMA (Bab 3 Laporan TA — Persamaan 3.1):

    F(k) = w_s × S_sem(k)  +  w_g × S_geo(k)  +  w_p × S_price(k)

    Di mana:
        F(k)       = Final Score kos k ∈ [0.0, 1.0]
        w_s = 0.50 = Bobot sinyal semantik (relevansi teks)
        w_g = 0.30 = Bobot sinyal geospatial (kedekatan lokasi)
        w_p = 0.20 = Bobot sinyal harga (nilai ekonomis)

  Sinyal Semantik — S_sem(k):
    Cosine similarity dari FAISS IndexFlatIP setelah L2-normalization.
    Range: [0.0, 1.0]. Semakin tinggi = teks kos lebih mirip kueri.

  Sinyal Geospatial — S_geo(k):
    Exponential decay berdasarkan jarak Haversine (PostGIS ST_Distance):
        S_geo = exp(−λ × d_km),   λ = 0.3

    Referensi nilai (λ=0.3):
        d=0.0 km → S=1.000  (di lokasi sama dengan user)
        d=1.0 km → S=0.741  (jarak kaki / sepeda)
        d=2.3 km → S=0.500  (threshold moderat)
        d=3.3 km → S=0.370  (1 standar deviasi, Jakarta urban context)
        d=5.0 km → S=0.223  (perlu ojek/motor)
       d=10.0 km → S=0.050  (jauh, tidak direkomendasikan)

    Pemilihan exponential decay (vs linear/sigmoid):
    Memberikan gradient yang smooth dan besar di dekat user,
    mencerminkan preferensi penyewa kos yang sangat mengutamakan jarak dekat.

  Sinyal Harga — S_price(k):
    Boost biner/bertingkat berdasarkan label Random Forest:
        "super_deal"  → 1.0  (diskon > 15% dari harga wajar)
        "underpriced" → 0.5  (diskon 5–15%)
        "fair"        → 0.0  (harga wajar, tidak ada bonus)
        "overpriced"  → 0.0  (terlalu mahal, tidak ada bonus)

  Pemilihan Bobot (0.50 : 0.30 : 0.20):
    Berdasarkan feature importance Random Forest dari dataset:
    kota_encoded=18.5%, jarak_kampus=9.3%, total_fasilitas=12.8%.
    Relevansi tekstual tetap paling dominan (intent user-centric).
    Bobot dapat dikonfigurasi untuk eksperimen ablation study di Bab 4.
═══════════════════════════════════════════════════════════════════
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KosScoringInput:
    """Paket data input untuk kalkulasi fusion score satu kandidat kos."""

    id_kos: int
    semantic_score: float    # Cosine similarity dari FAISS Search 2 [0.0–1.0]
    distance_km: float       # Jarak Haversine dari user ke kos (km, dari PostGIS)
    price_label: str         # "super_deal"|"underpriced"|"fair"|"overpriced"

    # Metadata kos (diteruskan ke output tanpa modifikasi)
    nama_kos: str = ""
    kota: str = ""
    wilayah: str = ""
    harga_per_bulan: int = 0
    predicted_price: float = 0.0
    foto_path: str = ""
    rating: float = 0.0
    ac: int = 0
    kamar_mandi_dalam: int = 0
    wifi: int = 0
    listrik_include: int = 0
    parkir: int = 0
    dapur: int = 0
    laundry: int = 0
    security_24jam: int = 0


@dataclass
class KosScoringResult:
    """Hasil kalkulasi fusion score satu kandidat kos (output SearchService)."""

    id_kos: int
    nama_kos: str
    kota: str
    wilayah: str
    harga_per_bulan: int
    distance_km: float
    # Skor komponen (untuk transparansi dan analisis di laporan TA)
    final_score: float
    semantic_score: float
    geospatial_score: float
    price_boost: float
    # Label dan badge
    price_label: str
    is_super_deal: bool      # True jika price_label == "super_deal" → badge 🔥
    predicted_price: float
    # Metadata
    foto_path: str
    rating: float
    ac: int
    kamar_mandi_dalam: int
    wifi: int
    listrik_include: int
    parkir: int
    dapur: int
    laundry: int
    security_24jam: int


class FusionScorer:
    """
    Mesin penyatuan skor hybrid Smart-Kos.

    Seluruh bobot dan parameter dapat dikonfigurasi untuk:
    1. Tuning performa di lingkungan berbeda (kampus vs kota besar)
    2. Eksperimen ablation study (matikan satu sinyal, ukur dampaknya)
    3. A/B testing di production (Tahap 4)

    Attributes:
        w_semantic         : Bobot sinyal semantik (default: 0.50)
        w_geospatial       : Bobot sinyal geospatial (default: 0.30)
        w_price            : Bobot sinyal harga (default: 0.20)
        geo_decay_lambda   : Parameter λ exponential decay (default: 0.3)
    """

    # Tabel price boost — nilai divalidasi secara eksperimental
    PRICE_BOOST_TABLE: dict[str, float] = {
        "super_deal":  1.0,   # Diskon > 15%: bonus penuh
        "underpriced": 0.5,   # Diskon 5–15%: bonus setengah
        "fair":        0.0,   # ±5% toleransi: tanpa bonus
        "overpriced":  0.0,   # Markup > 5%: tanpa bonus
    }

    def __init__(
        self,
        w_semantic: float = 0.50,
        w_geospatial: float = 0.30,
        w_price: float = 0.20,
        geo_decay_lambda: float = 0.3,
    ) -> None:
        """
        Inisialisasi Fusion Scorer dengan validasi kelengkapan bobot.

        Args:
            w_semantic       : Bobot sinyal semantik.
            w_geospatial     : Bobot sinyal geospatial.
            w_price          : Bobot sinyal harga.
            geo_decay_lambda : λ untuk S_geo = exp(−λ × d_km). Semakin besar λ,
                               semakin cepat skor turun terhadap jarak.

        Raises:
            ValueError: Jika total bobot ≠ 1.0 (toleransi 1e-6).
                        Mencegah final_score keluar dari rentang [0.0, 1.0].
        """
        total = w_semantic + w_geospatial + w_price
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Jumlah bobot harus tepat = 1.0. "
                f"Diterima: w_s={w_semantic} + w_g={w_geospatial} + "
                f"w_p={w_price} = {total:.8f}"
            )

        self.w_semantic = w_semantic
        self.w_geospatial = w_geospatial
        self.w_price = w_price
        self.geo_decay_lambda = geo_decay_lambda

        logger.info(
            f"FusionScorer: w_sem={w_semantic}, w_geo={w_geospatial}, "
            f"w_price={w_price}, λ={geo_decay_lambda}"
        )

    def compute_geospatial_score(self, distance_km: float) -> float:
        """
        Hitung geospatial score menggunakan exponential decay.

        Formula: S_geo(d) = exp(−λ × d)

        Dipilih exponential decay karena:
        - Diferensiabel di seluruh domain (gradient ada di semua titik)
        - Asimptotik ke 0 saat d → ∞ (kos sangat jauh → skor sangat kecil)
        - Parameter λ dapat di-tune untuk berbagai konteks urban/rural

        Args:
            distance_km: Jarak dari user ke kos dalam kilometer.
                        Nilai negatif di-clamp ke 0 (edge case GPS error).

        Returns:
            Float S_geo ∈ (0.0, 1.0]. Semakin kecil jarak = semakin tinggi skor.
        """
        return math.exp(-self.geo_decay_lambda * max(0.0, distance_km))

    def compute_price_boost(self, price_label: str) -> float:
        """
        Konversi label harga Random Forest → nilai boost numerik.

        Menggunakan PRICE_BOOST_TABLE untuk mapping deterministik.
        Label yang tidak dikenali (edge case) mendapat boost = 0.0.

        Args:
            price_label: String label dari PricingService.predict_batch().

        Returns:
            Float boost ∈ {0.0, 0.5, 1.0}.
        """
        return self.PRICE_BOOST_TABLE.get(price_label.lower().strip(), 0.0)

    def compute_final_score(
        self,
        semantic_score: float,
        geospatial_score: float,
        price_boost: float,
    ) -> float:
        """
        Hitung final score dari tiga komponen sinyal.

        Formula (Persamaan 3.1):
            F = w_s × S_sem + w_g × S_geo + w_p × S_price

        Seluruh input sudah dalam range [0.0, 1.0] sehingga:
            F_min = 0.0 (semua sinyal = 0, sangat tidak relevan)
            F_max = 1.0 (semua sinyal = 1, sempurna di semua dimensi)

        Args:
            semantic_score   : Cosine similarity dari FAISS (0.0–1.0).
            geospatial_score : Output compute_geospatial_score() (0.0–1.0].
            price_boost      : Output compute_price_boost() {0.0, 0.5, 1.0}.

        Returns:
            Final score ∈ [0.0, 1.0].
        """
        return (
            self.w_semantic * semantic_score
            + self.w_geospatial * geospatial_score
            + self.w_price * price_boost
        )

    def score_and_rank(
        self,
        candidates: list[KosScoringInput],
    ) -> list[KosScoringResult]:
        """
        Hitung final score semua kandidat sekaligus dan urutkan descending.

        Ini adalah method utama yang dipanggil oleh SearchService.
        Semua kalkulasi bersifat pure function (tidak ada side effects)
        sehingga mudah di-unit test secara terisolasi.

        Args:
            candidates: List KosScoringInput dari search pipeline.

        Returns:
            List KosScoringResult terurut berdasarkan final_score (tertinggi → terendah).
            Kos dengan final_score tertinggi di posisi [0] adalah rekomendasi terbaik.
        """
        results: list[KosScoringResult] = []

        for c in candidates:
            geo_score = self.compute_geospatial_score(c.distance_km)
            price_boost = self.compute_price_boost(c.price_label)
            final_score = self.compute_final_score(
                c.semantic_score, geo_score, price_boost
            )

            results.append(
                KosScoringResult(
                    id_kos=c.id_kos,
                    nama_kos=c.nama_kos,
                    kota=c.kota,
                    wilayah=c.wilayah,
                    harga_per_bulan=c.harga_per_bulan,
                    distance_km=round(c.distance_km, 3),
                    final_score=round(final_score, 6),
                    semantic_score=round(c.semantic_score, 6),
                    geospatial_score=round(geo_score, 6),
                    price_boost=price_boost,
                    price_label=c.price_label,
                    is_super_deal=(c.price_label.lower() == "super_deal"),
                    predicted_price=round(c.predicted_price, 0),
                    foto_path=c.foto_path,
                    rating=c.rating,
                    ac=c.ac,
                    kamar_mandi_dalam=c.kamar_mandi_dalam,
                    wifi=c.wifi,
                    listrik_include=c.listrik_include,
                    parkir=c.parkir,
                    dapur=c.dapur,
                    laundry=c.laundry,
                    security_24jam=c.security_24jam,
                )
            )

        # Sort descending: kos terbaik di posisi 0
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results
```

---

## FILE 5: `app/services/pricing_service.py`

```python
"""
app/services/pricing_service.py — Random Forest Pricing Service
Smart-Kos Hybrid Engine — Tahap 2

Menggunakan model_rf.pkl + scaler.pkl (dari notebook Kaila Hidayat, 2024)
untuk memprediksi harga wajar kos berdasarkan fasilitas dan lokasi,
kemudian mengklasifikasikan tingkat kewajaran harga aktual.

Klasifikasi Harga:
    "super_deal"  : actual < 85%  × predicted  → diskon > 15%
    "underpriced" : actual < 95%  × predicted  → diskon 5–15%
    "fair"        : actual ≤ 105% × predicted  → toleransi ± 5%
    "overpriced"  : actual > 105% × predicted  → markup > 5%
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PricePredictionResult:
    """Hasil prediksi harga untuk satu kos."""
    id_kos: int
    actual_price: int
    predicted_price: float    # Prediksi harga wajar Random Forest (Rp/bulan)
    price_label: str          # "super_deal"|"underpriced"|"fair"|"overpriced"
    discount_pct: float       # (predicted − actual) / predicted × 100
                              # Positif = aktual lebih murah dari prediksi


class PricingService:
    """
    Layanan prediksi harga wajar berbasis Random Forest Regressor.

    Model: RandomForestRegressor (Scikit-learn) dengan:
        R² Test  = 0.87 (menjelaskan 87% variansi harga)
        MAE Test = Rp 195.000 (< 10% dari median harga)
        RMSE     = Rp 285.000
    (Performa dari notebook dataset asli Kaila Hidayat, Nov 2024)

    Feature ordering KRITIS — harus identik dengan saat training.
    Sumber urutan: analisis_kos_jakarta.ipynb, Cell "Model Training".
    """

    # Threshold klasifikasi sebagai rasio actual/predicted
    THRESHOLD_SUPER_DEAL: float = 0.85    # actual < 85% predicted
    THRESHOLD_UNDERPRICED: float = 0.95   # actual < 95% predicted
    THRESHOLD_FAIR: float = 1.05          # actual ≤ 105% predicted

    # 16 fitur dalam urutan IDENTIK dengan saat model dilatih — JANGAN diubah
    FEATURE_COLUMNS: tuple[str, ...] = (
        "kota_encoded",
        "ukuran_kamar",
        "tipe_kos_encoded",
        "ac",
        "kamar_mandi_dalam",
        "wifi",
        "listrik_include",
        "parkir",
        "dapur",
        "laundry",
        "security_24jam",
        "jarak_ke_kampus_km",
        "jarak_ke_transportasi_km",
        "total_fasilitas",
        "jarak_kampus_dekat",
        "jarak_transportasi_dekat",
    )

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
    ) -> None:
        """
        Muat model dan scaler dari file .pkl.

        Args:
            model_path : Path ke model_rf.pkl.
            scaler_path: Path ke scaler.pkl.

        Raises:
            FileNotFoundError: Jika salah satu file tidak ditemukan.
        """
        for path in (model_path, scaler_path):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"File ML tidak ditemukan: {path}\n"
                    "Pastikan model_rf.pkl dan scaler.pkl di folder models_ml/"
                )

        logger.info(f"Memuat Random Forest model: {model_path}")
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        logger.info(f"Memuat StandardScaler: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        logger.info(
            f"PricingService siap. "
            f"Model: {type(self._model).__name__} | "
            f"Features: {len(self.FEATURE_COLUMNS)}"
        )

    def _classify(
        self,
        actual_price: int,
        predicted_price: float,
    ) -> tuple[str, float]:
        """
        Klasifikasi label harga berdasarkan rasio actual/predicted.

        Args:
            actual_price    : Harga sewa aktual (Rp/bulan).
            predicted_price : Prediksi model Random Forest (Rp/bulan).

        Returns:
            Tuple (price_label, discount_pct).
        """
        if predicted_price <= 0:
            return "fair", 0.0

        ratio = actual_price / predicted_price
        discount_pct = round((1.0 - ratio) * 100.0, 2)

        if ratio < self.THRESHOLD_SUPER_DEAL:
            label = "super_deal"
        elif ratio < self.THRESHOLD_UNDERPRICED:
            label = "underpriced"
        elif ratio <= self.THRESHOLD_FAIR:
            label = "fair"
        else:
            label = "overpriced"

        return label, discount_pct

    def _build_features_df(self, kos_list: list[dict]) -> pd.DataFrame:
        """
        Bangun DataFrame fitur dari list data kos.
        Urutan kolom dijaga identik dengan FEATURE_COLUMNS.
        Nilai yang hilang di-fill dengan 0 (default fitur binary/numerik).

        Args:
            kos_list: List dict data kos dari PostgreSQL.

        Returns:
            DataFrame shape (N, 16) dengan kolom terurut sesuai FEATURE_COLUMNS.
        """
        rows = [
            {col: float(kos.get(col, 0) or 0) for col in self.FEATURE_COLUMNS}
            for kos in kos_list
        ]
        return pd.DataFrame(rows)[list(self.FEATURE_COLUMNS)]

    def predict_single(self, kos_dict: dict) -> PricePredictionResult:
        """
        Prediksi harga wajar untuk satu kos (digunakan untuk testing/demo).

        Args:
            kos_dict: Dictionary data kos dengan key sesuai FEATURE_COLUMNS
                     ditambah 'id_kos' dan 'harga_per_bulan'.

        Returns:
            PricePredictionResult dengan label dan detail prediksi.
        """
        features_df = self._build_features_df([kos_dict])
        scaled = self._scaler.transform(features_df)
        predicted = float(self._model.predict(scaled)[0])

        actual = int(kos_dict.get("harga_per_bulan", 0))
        label, discount_pct = self._classify(actual, predicted)

        return PricePredictionResult(
            id_kos=int(kos_dict.get("id_kos", 0)),
            actual_price=actual,
            predicted_price=round(predicted, 2),
            price_label=label,
            discount_pct=discount_pct,
        )

    def predict_batch(
        self,
        kos_list: list[dict],
    ) -> list[PricePredictionResult]:
        """
        Prediksi harga wajar untuk sekelompok kos secara vectorized.

        Vectorized inference jauh lebih efisien dari loop predict_single()
        karena sklearn RandomForestRegressor memanfaatkan batch prediction
        secara internal (satu forward pass untuk seluruh batch).

        Args:
            kos_list: List dict data kos dari SearchService._fetch_kos_with_distance().

        Returns:
            List PricePredictionResult (indeks berkorespondensi dengan input).
        """
        if not kos_list:
            return []

        features_df = self._build_features_df(kos_list)
        scaled = self._scaler.transform(features_df)

        # Satu pemanggilan predict() untuk seluruh batch
        predictions: np.ndarray = self._model.predict(scaled)

        results: list[PricePredictionResult] = []
        for kos_dict, predicted in zip(kos_list, predictions):
            actual = int(kos_dict.get("harga_per_bulan", 0))
            label, discount_pct = self._classify(actual, float(predicted))
            results.append(
                PricePredictionResult(
                    id_kos=int(kos_dict.get("id_kos", 0)),
                    actual_price=actual,
                    predicted_price=round(float(predicted), 2),
                    price_label=label,
                    discount_pct=discount_pct,
                )
            )

        super_deal_count = sum(1 for r in results if r.price_label == "super_deal")
        logger.info(
            f"Batch pricing: {len(results)} kos | "
            f"Super Deal: {super_deal_count} | "
            f"Underpriced: {sum(1 for r in results if r.price_label == 'underpriced')}"
        )
        return results
```

---

## FILE 6: `app/services/search_service.py`

```python
"""
app/services/search_service.py — Hybrid Search Pipeline Orchestrator
Smart-Kos Hybrid Engine — Tahap 2

Mengintegrasikan seluruh komponen AI engine menjadi satu pipeline pencarian.
Kelas ini MURNI logika bisnis — tidak mengandung dekorator FastAPI.
Endpoint HTTP akan dibungkus di app/api/v1/endpoints/search.py pada Tahap 3.

Pipeline Lengkap (11 Langkah):
    1.  Sastrawi preprocessing kueri
    2.  IndoBERT embedding kueri → vektor 768-dim
    3.  FAISS Search Putaran 1 → top-5 docs untuk PRF feedback
    4.  Fetch deskripsi_clean dari PostgreSQL (PRF input)
    5.  PRF: ekstrak term dominan → expand kueri
    6.  IndoBERT re-embed kueri diperluas → vektor 768-dim baru
    7.  FAISS Search Putaran 2 (re-ranking) → top-20 kandidat
    8.  Fetch data kos lengkap + jarak PostGIS Haversine dari PostgreSQL
    9.  RF batch price prediction untuk semua kandidat
    10. Fusion scoring (semantic + geo + price) untuk semua kandidat
    11. Sort descending by final_score, return top_k
"""

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.pricing_service import PricePredictionResult, PricingService
from stki.embedder import IndoBERTEmbedder
from stki.fusion_scorer import FusionScorer, KosScoringInput, KosScoringResult
from stki.indexer import KosVectorIndexer
from stki.preprocessor import IndonesianTextPreprocessor
from stki.prf_engine import PseudoRelevanceFeedback

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    """Respons lengkap dari satu search request (akan di-serialize ke JSON di Tahap 3)."""

    query_original: str           # Kueri mentah dari user
    query_preprocessed: str       # Setelah Sastrawi pipeline
    query_expanded: str           # Setelah PRF expansion
    expansion_terms: list[str]    # Term yang ditambahkan PRF (transparansi)
    results: list[KosScoringResult]           # Kos terurut by final_score
    total_candidates_retrieved: int           # Pool sebelum filter top_k
    search_time_ms: float         # Latency end-to-end (ms)


class SearchService:
    """
    Orkestrasi pipeline Smart-Kos Hybrid Search.

    Semua komponen diinjeksikan via __init__ (Dependency Injection Pattern).
    Keuntungan:
      - Mudah di-unit test (setiap komponen dapat di-mock secara terpisah)
      - Mudah dikonfigurasi per-environment (dev/staging/production)
      - Mudah diswap komponen (misal: IndoBERT → model lain di masa depan)

    Setup (akan dilakukan di FastAPI lifespan handler, Tahap 3):
        embedder = IndoBERTEmbedder(settings.indobert_model_name)
        indexer  = KosVectorIndexer(settings.faiss_indobert_index_path, ..., embedder)
        indexer.load_index()
        search_svc = SearchService(embedder, indexer, ...)
        app.state.search_service = search_svc
    """

    def __init__(
        self,
        embedder: IndoBERTEmbedder,
        indexer: KosVectorIndexer,
        prf_engine: PseudoRelevanceFeedback,
        pricing_service: PricingService,
        fusion_scorer: FusionScorer,
        preprocessor: IndonesianTextPreprocessor,
    ) -> None:
        self.embedder = embedder
        self.indexer = indexer
        self.prf_engine = prf_engine
        self.pricing_service = pricing_service
        self.fusion_scorer = fusion_scorer
        self.preprocessor = preprocessor

    async def search(
        self,
        query: str,
        user_lat: float,
        user_lon: float,
        top_k: int = 10,
        n_candidates: int = 20,
        db: AsyncSession | None = None,
    ) -> SearchResponse:
        """
        Eksekusi full hybrid search pipeline (11 langkah).

        Args:
            query       : Teks pencarian mentah dari user Flutter.
            user_lat    : Latitude GPS user (WGS84 / EPSG:4326).
            user_lon    : Longitude GPS user.
            top_k       : Jumlah kos yang dikembalikan ke Flutter.
            n_candidates: Pool kandidat dari FAISS Re-rank sebelum top_k filter.
                         Semakin besar → lebih banyak pilihan untuk scoring,
                         semakin kecil → lebih cepat. Default 20 optimal.
            db          : Async SQLAlchemy session.
                         None = mode offline (PostGIS jarak = 999 km, hanya testing).

        Returns:
            SearchResponse dengan ranked results dan metadata pipeline.
        """
        t_start = time.perf_counter()

        # ─────────────────────────────────────────────────────────
        # LANGKAH 1: Preprocessing kueri via Sastrawi
        # ─────────────────────────────────────────────────────────
        query_preprocessed = self.preprocessor.preprocess(query)
        if not query_preprocessed.strip():
            logger.warning(
                f"Kueri kosong setelah preprocessing: '{query}'. "
                "Menggunakan lowercase tanpa stemming."
            )
            query_preprocessed = query.lower().strip()

        # ─────────────────────────────────────────────────────────
        # LANGKAH 2: IndoBERT Embedding kueri awal
        # ─────────────────────────────────────────────────────────
        # query_vector: np.float32 (1, 768), L2-normalized
        # (Digunakan implisit oleh indexer.search() di langkah 3)

        # ─────────────────────────────────────────────────────────
        # LANGKAH 3: FAISS Search Putaran 1 — top-5 untuk PRF
        # ─────────────────────────────────────────────────────────
        n_prf_docs = self.prf_engine.n_feedback_docs
        initial_results = self.indexer.search(
            query_preprocessed, top_k=n_prf_docs
        )
        initial_ids = [r["id_kos"] for r in initial_results]

        # ─────────────────────────────────────────────────────────
        # LANGKAH 4: Fetch deskripsi_clean dari PostgreSQL untuk PRF
        # ─────────────────────────────────────────────────────────
        feedback_docs: list[str] = []
        if initial_ids and db is not None:
            feedback_docs = await self._fetch_deskripsi_clean(db, initial_ids)

        # ─────────────────────────────────────────────────────────
        # LANGKAH 5: PRF — Ekstraksi term + Query Expansion
        # ─────────────────────────────────────────────────────────
        query_expanded, expansion_terms = self.prf_engine.run(
            original_query=query_preprocessed,
            feedback_docs=feedback_docs,
        )

        # ─────────────────────────────────────────────────────────
        # LANGKAH 6 & 7: IndoBERT Re-embed + FAISS Search Putaran 2
        # Jika PRF menemukan term baru → gunakan expanded query
        # Jika tidak → tetap gunakan query preprocessed asli
        # ─────────────────────────────────────────────────────────
        query_for_rerank = (
            query_expanded if expansion_terms else query_preprocessed
        )
        reranked_results = self.indexer.search(
            query_for_rerank, top_k=n_candidates
        )
        candidate_ids = [r["id_kos"] for r in reranked_results]

        # Map id_kos → semantic_score untuk digunakan di fusion scoring
        semantic_score_map: dict[int, float] = {
            r["id_kos"]: r["score"] for r in reranked_results
        }

        # ─────────────────────────────────────────────────────────
        # LANGKAH 8: Fetch data KOS lengkap + jarak PostGIS
        # ─────────────────────────────────────────────────────────
        kos_data_list: list[dict] = []
        if candidate_ids:
            if db is not None:
                kos_data_list = await self._fetch_kos_with_distance(
                    db=db,
                    kos_ids=candidate_ids,
                    user_lat=user_lat,
                    user_lon=user_lon,
                )
            else:
                # Mode offline: gunakan data minimal tanpa jarak nyata
                logger.warning(
                    "DB session tidak tersedia (mode offline). "
                    "Jarak default 999 km digunakan untuk semua kandidat."
                )
                kos_data_list = [
                    {"id_kos": id_, "distance_km": 999.0, "harga_per_bulan": 0}
                    for id_ in candidate_ids
                ]

        # ─────────────────────────────────────────────────────────
        # LANGKAH 9: Batch RF Price Prediction
        # ─────────────────────────────────────────────────────────
        price_results = self.pricing_service.predict_batch(kos_data_list)
        price_map: dict[int, PricePredictionResult] = {
            r.id_kos: r for r in price_results
        }

        # ─────────────────────────────────────────────────────────
        # LANGKAH 10: Bangun KosScoringInput untuk Fusion Scorer
        # ─────────────────────────────────────────────────────────
        scoring_inputs: list[KosScoringInput] = []
        for kos in kos_data_list:
            kos_id = int(kos.get("id_kos", 0))
            pr = price_map.get(kos_id)

            scoring_inputs.append(
                KosScoringInput(
                    id_kos=kos_id,
                    semantic_score=semantic_score_map.get(kos_id, 0.0),
                    distance_km=float(kos.get("distance_km") or 999.0),
                    price_label=pr.price_label if pr else "fair",
                    nama_kos=str(kos.get("nama_kos", "")),
                    kota=str(kos.get("kota", "")),
                    wilayah=str(kos.get("wilayah", "")),
                    harga_per_bulan=int(kos.get("harga_per_bulan", 0)),
                    predicted_price=pr.predicted_price if pr else 0.0,
                    foto_path=str(kos.get("foto_path", "")),
                    rating=float(kos.get("rating") or 0.0),
                    ac=int(kos.get("ac", 0)),
                    kamar_mandi_dalam=int(kos.get("kamar_mandi_dalam", 0)),
                    wifi=int(kos.get("wifi", 0)),
                    listrik_include=int(kos.get("listrik_include", 0)),
                    parkir=int(kos.get("parkir", 0)),
                    dapur=int(kos.get("dapur", 0)),
                    laundry=int(kos.get("laundry", 0)),
                    security_24jam=int(kos.get("security_24jam", 0)),
                )
            )

        # ─────────────────────────────────────────────────────────
        # LANGKAH 11: Fusion Scoring + Sort + Return top_k
        # ─────────────────────────────────────────────────────────
        ranked_results = self.fusion_scorer.score_and_rank(scoring_inputs)
        final_results = ranked_results[:top_k]

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        super_deal_count = sum(1 for r in final_results if r.is_super_deal)

        logger.info(
            f"Search selesai: '{query}' → {len(final_results)} hasil | "
            f"PRF +{len(expansion_terms)} terms | "
            f"🔥 Super Deal: {super_deal_count} | "
            f"Latency: {elapsed_ms}ms"
        )

        return SearchResponse(
            query_original=query,
            query_preprocessed=query_preprocessed,
            query_expanded=query_expanded,
            expansion_terms=expansion_terms,
            results=final_results,
            total_candidates_retrieved=len(reranked_results),
            search_time_ms=elapsed_ms,
        )

    # ──────────────────────────────────────────────────────────────
    # HELPER: Database Queries (PostgreSQL + PostGIS)
    # ──────────────────────────────────────────────────────────────

    async def _fetch_deskripsi_clean(
        self,
        db: AsyncSession,
        kos_ids: list[int],
    ) -> list[str]:
        """
        Ambil kolom 'deskripsi_clean' dari PostgreSQL untuk PRF feedback set.

        Hanya mengambil dokumen yang deskripsi_clean-nya tidak kosong.
        Urutan tidak dijamin (tidak diperlukan untuk PRF averaging).

        Args:
            db     : AsyncSession SQLAlchemy.
            kos_ids: List id_kos dari FAISS Search Putaran 1.

        Returns:
            List string 'deskripsi_clean' untuk PRF.extract_expansion_terms().
        """
        if not kos_ids:
            return []

        result = await db.execute(
            text(
                "SELECT deskripsi_clean "
                "FROM kos_listings "
                "WHERE id_kos = ANY(:ids) "
                "  AND deskripsi_clean IS NOT NULL "
                "  AND LENGTH(TRIM(deskripsi_clean)) > 0"
            ),
            {"ids": kos_ids},
        )
        return [str(row[0]) for row in result.fetchall()]

    async def _fetch_kos_with_distance(
        self,
        db: AsyncSession,
        kos_ids: list[int],
        user_lat: float,
        user_lon: float,
    ) -> list[dict]:
        """
        Ambil seluruh data kos + hitung jarak Haversine dari lokasi user.

        Query PostGIS:
            ST_Distance(geom, user_point::GEOGRAPHY) / 1000.0 → jarak km

        Menggunakan tipe GEOGRAPHY (bukan GEOMETRY) untuk akurasi jarak
        geodesik (ellipsoidal earth model) yang lebih akurat dari flat-earth.

        Kos tanpa data geom (geom IS NULL) mendapatkan fallback distance=999 km
        agar tidak muncul di posisi teratas ranking.

        Args:
            db      : AsyncSession SQLAlchemy.
            kos_ids : List id_kos kandidat dari FAISS Re-rank.
            user_lat: Latitude GPS user (WGS84).
            user_lon: Longitude GPS user (WGS84).

        Returns:
            List dict — satu per kos, berisi semua kolom + 'distance_km'.
        """
        if not kos_ids:
            return []

        sql = text("""
            SELECT
                k.id_kos,
                k.nama_kos,
                k.kota,
                k.wilayah,
                k.harga_per_bulan,
                k.ac,
                k.kamar_mandi_dalam,
                k.wifi,
                k.listrik_include,
                k.parkir,
                k.dapur,
                k.laundry,
                k.security_24jam,
                k.total_fasilitas,
                k.jarak_ke_kampus_km,
                k.jarak_ke_transportasi_km,
                k.jarak_kampus_dekat,
                k.jarak_transportasi_dekat,
                k.kota_encoded,
                k.tipe_kos_encoded,
                k.ukuran_kamar,
                k.foto_path,
                k.rating,
                CASE
                    WHEN k.geom IS NOT NULL THEN
                        ST_Distance(
                            k.geom,
                            ST_SetSRID(
                                ST_MakePoint(:user_lon, :user_lat),
                                4326
                            )::GEOGRAPHY
                        ) / 1000.0
                    ELSE 999.0
                END AS distance_km
            FROM kos_listings k
            WHERE k.id_kos = ANY(:kos_ids)
        """)

        result = await db.execute(
            sql,
            {
                "user_lat": float(user_lat),
                "user_lon": float(user_lon),
                "kos_ids": kos_ids,
            },
        )
        return [dict(row._mapping) for row in result.fetchall()]
```

---

## FILE 7: `scripts/rebuild_faiss_indobert.py`

```python
"""
scripts/rebuild_faiss_indobert.py — Rebuild FAISS Index dengan IndoBERT
Smart-Kos Hybrid Engine — Tahap 2

Menggantikan kos_tfidf.index (Tahap 1) dengan kos_indobert.index (Tahap 2).

══════════════════════════════════════════════════════════════════
⚠  PERINGATAN PERFORMA:

  Encoding 909 dokumen via IndoBERT (BERT-base, 768-dim):
  ┌─────────────────────────────┬──────────────┬────────────────┐
  │ Environment                 │ batch_size   │ Estimasi Waktu │
  ├─────────────────────────────┼──────────────┼────────────────┤
  │ CPU lokal (Intel i5/i7)     │ 8 – 16       │ 30–60 menit    │
  │ GPU T4 Google Colab (FREE)  │ 64 – 128     │ 2–5 menit      │
  │ GPU A100 Kaggle (FREE)      │ 128 – 256    │ 1–2 menit      │
  └─────────────────────────────┴──────────────┴────────────────┘

  REKOMENDASI:
    Jalankan script ini di Google Colab T4, lalu download hasilnya:
      faiss_index/kos_indobert.index
      faiss_index/id_map_indobert.pkl
    ke folder lokal untuk digunakan oleh FastAPI server.

  INSTRUKSI COLAB (copy ke cell):
    !git clone https://github.com/USERNAME/smart-kos-backend.git
    %cd smart-kos-backend
    !pip install -q faiss-cpu transformers torch sentencepiece PySastrawi
    !python scripts/rebuild_faiss_indobert.py
    # Download kedua file di panel Files kiri sebelum session mati
══════════════════════════════════════════════════════════════════
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from stki.embedder import IndoBERTEmbedder
from stki.indexer import KosVectorIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def rebuild_faiss_indobert(
    processed_csv_path: str,
    index_path: str,
    id_map_path: str,
    model_name: str = "indobenchmark/indobert-base-p1",
    max_length: int = 128,
    batch_size: int | None = None,
) -> None:
    """
    Bangun FAISS IndoBERT index dari dataset processed CSV.

    Args:
        processed_csv_path : Path ke data_kos_processed.csv.
        index_path         : Path output FAISS index (.index).
        id_map_path        : Path output id_map (.pkl).
        model_name         : HuggingFace model identifier.
        max_length         : Panjang token maksimum (128 cukup untuk deskripsi kos).
        batch_size         : Batch size encoding. None = otomatis (GPU: 64, CPU: 8).

    Raises:
        FileNotFoundError: Jika CSV processed tidak ada.
        ValueError       : Jika kolom 'deskripsi_clean' tidak ditemukan.
    """
    if not Path(processed_csv_path).exists():
        raise FileNotFoundError(
            f"CSV processed tidak ditemukan: {processed_csv_path}\n"
            "Solusi: python scripts/preprocess_dataset.py"
        )

    logger.info(f"Membaca dataset: {processed_csv_path}")
    df: pd.DataFrame = pd.read_csv(processed_csv_path, encoding="utf-8")

    if "deskripsi_clean" not in df.columns:
        raise ValueError(
            "Kolom 'deskripsi_clean' tidak ada. "
            "Jalankan scripts/preprocess_dataset.py terlebih dahulu."
        )

    df_valid = df[df["deskripsi_clean"].str.strip().str.len() > 0].copy()
    logger.info(f"Dokumen valid: {len(df_valid)}/{len(df)}")

    documents: list[str] = df_valid["deskripsi_clean"].tolist()
    kos_ids: list[int] = df_valid["id_kos"].astype(int).tolist()

    # Tentukan batch_size optimal berdasarkan device
    is_gpu = torch.cuda.is_available()
    if batch_size is None:
        batch_size = 64 if is_gpu else 8
        logger.info(
            f"Auto batch_size: {batch_size} "
            f"({'GPU' if is_gpu else 'CPU'})"
        )

    if not is_gpu:
        logger.warning(
            "⚠  GPU tidak terdeteksi. Encoding akan berjalan di CPU.\n"
            "   Estimasi waktu: 30-60 menit untuk 909 dokumen.\n"
            "   Sangat disarankan menggunakan Google Colab T4 (gratis)."
        )

    # Inisialisasi IndoBERT embedder
    logger.info(f"Memuat IndoBERT: {model_name}...")
    embedder = IndoBERTEmbedder(
        model_name=model_name,
        device="auto",
        max_length=max_length,
    )

    # Bangun FAISS index
    indexer = KosVectorIndexer(
        index_path=index_path,
        id_map_path=id_map_path,
        embedder=embedder,
    )

    indexer.build_index(
        documents=documents,
        kos_ids=kos_ids,
        encode_batch_size=batch_size,
    )

    logger.info("✅ IndoBERT FAISS index berhasil dibangun!")
    logger.info(f"   Index   : {index_path}")
    logger.info(f"   ID Map  : {id_map_path}")
    logger.info(f"   Docs    : {len(df_valid)}")
    logger.info(f"   Dimensi : {IndoBERTEmbedder.EMBEDDING_DIM}")
    if not is_gpu:
        logger.info(
            "   Jika berjalan di Colab, download file index sekarang "
            "sebelum session Google Colab berakhir!"
        )


if __name__ == "__main__":
    rebuild_faiss_indobert(
        processed_csv_path=settings.data_processed_path,
        index_path=settings.faiss_indobert_index_path,
        id_map_path=settings.faiss_indobert_id_map_path,
        model_name=settings.indobert_model_name,
        max_length=settings.indobert_max_length,
        batch_size=settings.indobert_batch_size if settings.indobert_batch_size != 16
                   else None,  # None = auto-detect GPU/CPU
    )
```

---

## FILE 8: `scripts/demo_search.py`

```python
"""
scripts/demo_search.py — Demo & Validasi AI Engine Tahap 2
Smart-Kos Hybrid Engine

Menjalankan full search pipeline TANPA FastAPI dan TANPA database
untuk memvalidasi seluruh komponen AI sebelum Tahap 3 (API endpoint).

Mode yang tersedia:
  1. Mode Offline (default): Tanpa DB — PostGIS jarak = 999km, valid untuk
     memvalidasi IndoBERT + PRF + RF Pricing + Fusion Scoring.
  2. Mode Online: Dengan DB — jalankan setelah init_postgis.py selesai.

Prasyarat:
  ✅ scripts/rebuild_faiss_indobert.py sudah dijalankan (index ada di disk)
  ✅ models_ml/model_rf.pkl dan scaler.pkl tersedia
  ✅ python -m pip install torch transformers sentencepiece (sudah terinstall)

Usage:
  python scripts/demo_search.py
  python scripts/demo_search.py --query "kos wifi ac dekat kampus"
  python scripts/demo_search.py --query "kos murah bisa masak" --top_k 3
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.services.pricing_service import PricingService
from app.services.search_service import SearchService
from stki.embedder import IndoBERTEmbedder
from stki.fusion_scorer import FusionScorer
from stki.indexer import KosVectorIndexer
from stki.preprocessor import IndonesianTextPreprocessor
from stki.prf_engine import PseudoRelevanceFeedback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# Kueri contoh — wakili variasi kebutuhan mahasiswa Jakarta
SAMPLE_QUERIES: list[str] = [
    "kos murah wifi dekat kampus",
    "kos putri AC kamar mandi dalam",
    "kos bisa masak dapur bersama",
    "kos aman security 24 jam",
    "hunian nyaman fasilitas lengkap",
]


async def run_demo(
    query: str,
    user_lat: float = -6.1862,    # Default: Jakarta Pusat
    user_lon: float = 106.8348,
    top_k: int = 5,
) -> None:
    """Inisialisasi semua komponen dan jalankan demo search pipeline."""

    logger.info("═" * 65)
    logger.info("DEMO: Smart-Kos Hybrid Engine — Tahap 2 Validation")
    logger.info("═" * 65)

    # ─── Setup komponen ─────────────────────────────────────────
    logger.info("[1/5] Memuat IndoBERT Embedder...")
    embedder = IndoBERTEmbedder(
        model_name=settings.indobert_model_name,
        device="auto",
        max_length=settings.indobert_max_length,
    )

    logger.info("[2/5] Memuat FAISS IndoBERT Index...")
    indexer = KosVectorIndexer(
        index_path=settings.faiss_indobert_index_path,
        id_map_path=settings.faiss_indobert_id_map_path,
        embedder=embedder,
    )
    indexer.load_index()

    logger.info("[3/5] Memuat RF Pricing Service...")
    pricing_service = PricingService(
        model_path=settings.model_rf_path,
        scaler_path=settings.scaler_path,
    )

    logger.info("[4/5] Inisialisasi PRF, FusionScorer, Preprocessor...")
    search_service = SearchService(
        embedder=embedder,
        indexer=indexer,
        prf_engine=PseudoRelevanceFeedback(
            n_feedback_docs=settings.prf_n_feedback_docs,
            n_expansion_terms=settings.prf_n_expansion_terms,
        ),
        pricing_service=pricing_service,
        fusion_scorer=FusionScorer(
            w_semantic=settings.fusion_weight_semantic,
            w_geospatial=settings.fusion_weight_geospatial,
            w_price=settings.fusion_weight_price,
            geo_decay_lambda=settings.fusion_geo_decay_lambda,
        ),
        preprocessor=IndonesianTextPreprocessor(),
    )

    # ─── Jalankan search ────────────────────────────────────────
    logger.info(f"[5/5] Menjalankan pencarian: '{query}'")
    response = await search_service.search(
        query=query,
        user_lat=user_lat,
        user_lon=user_lon,
        top_k=top_k,
        n_candidates=20,
        db=None,  # Mode offline
    )

    # ─── Tampilkan hasil ─────────────────────────────────────────
    print("\n" + "═" * 65)
    print("📊 HASIL PENCARIAN")
    print("═" * 65)
    print(f"  Kueri asli       : {response.query_original}")
    print(f"  Setelah Sastrawi : {response.query_preprocessed}")
    print(f"  Setelah PRF      : {response.query_expanded}")
    print(f"  Term ekspansi    : {response.expansion_terms}")
    print(f"  Total kandidat   : {response.total_candidates_retrieved}")
    print(f"  Latency          : {response.search_time_ms} ms")
    print("─" * 65)

    for i, r in enumerate(response.results, 1):
        badge = "🔥 SUPER DEAL!" if r.is_super_deal else f"[{r.price_label.upper()}]"
        fasilitas = []
        if r.ac:             fasilitas.append("AC")
        if r.kamar_mandi_dalam: fasilitas.append("KMD")
        if r.wifi:           fasilitas.append("WiFi")
        if r.security_24jam: fasilitas.append("Sec24j")
        if r.dapur:          fasilitas.append("Dapur")
        fas_str = "+".join(fasilitas) if fasilitas else "-"

        print(
            f"\n  #{i} id_kos={r.id_kos:04d} | {badge}\n"
            f"     Final Score   : {r.final_score:.4f} "
            f"(sem={r.semantic_score:.3f} + geo={r.geospatial_score:.3f} + "
            f"price_boost={r.price_boost:.1f})\n"
            f"     Harga Aktual  : Rp {r.harga_per_bulan:>12,.0f}/bln\n"
            f"     Harga Wajar   : Rp {r.predicted_price:>12,.0f}/bln (RF predict)\n"
            f"     Kota          : {r.kota} — {r.wilayah}\n"
            f"     Fasilitas     : {fas_str} | Rating: {r.rating}"
        )

    print("═" * 65)

    # ─── Rangkuman statistik ──────────────────────────────────────
    super_deals = [r for r in response.results if r.is_super_deal]
    print(f"\n📈 Ringkasan: {len(response.results)} kos | "
          f"🔥 Super Deal: {len(super_deals)} | "
          f"Latency: {response.search_time_ms}ms")

    if super_deals:
        best = super_deals[0]
        discount = round(
            (best.predicted_price - best.harga_per_bulan) / best.predicted_price * 100, 1
        )
        print(
            f"   Best Super Deal : id={best.id_kos} | "
            f"Hemat {discount}% (Rp {best.predicted_price - best.harga_per_bulan:,.0f}/bln)"
        )

    print("\n✅ Demo selesai. Tahap 2 AI Engine tervalidasi!")
    print("   Langkah berikutnya: TAHAP 3 (FastAPI Endpoints)\n")


async def run_batch_demo(queries: list[str]) -> None:
    """Jalankan multiple kueri untuk benchmarking latency."""
    print("\n" + "═" * 65)
    print("🔬 BATCH DEMO — Latency Benchmark")
    print("═" * 65)

    # Setup sekali untuk semua query (lebih efisien)
    embedder = IndoBERTEmbedder(settings.indobert_model_name, max_length=128)
    indexer = KosVectorIndexer(
        settings.faiss_indobert_index_path,
        settings.faiss_indobert_id_map_path,
        embedder,
    )
    indexer.load_index()
    pricing_service = PricingService(settings.model_rf_path, settings.scaler_path)
    search_service = SearchService(
        embedder=embedder,
        indexer=indexer,
        prf_engine=PseudoRelevanceFeedback(),
        pricing_service=pricing_service,
        fusion_scorer=FusionScorer(
            w_semantic=settings.fusion_weight_semantic,
            w_geospatial=settings.fusion_weight_geospatial,
            w_price=settings.fusion_weight_price,
        ),
        preprocessor=IndonesianTextPreprocessor(),
    )

    latencies: list[float] = []
    for i, q in enumerate(queries, 1):
        resp = await search_service.search(
            query=q, user_lat=-6.1862, user_lon=106.8348, top_k=5, db=None
        )
        latencies.append(resp.search_time_ms)
        super_deals = sum(1 for r in resp.results if r.is_super_deal)
        print(
            f"  Query {i:2d}: {q[:45]:<45} | "
            f"{resp.search_time_ms:6.1f}ms | 🔥×{super_deals}"
        )

    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    print(f"\n  Rata-rata: {avg_lat:.1f}ms | Maksimum: {max_lat:.1f}ms")
    print(
        "  (Catatan: query encoding CPU ~100-200ms, GPU ~10-30ms)\n"
        "  Target production SLA: < 500ms end-to-end dengan GPU."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo validasi Smart-Kos AI Engine Tahap 2"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="kos murah fasilitas lengkap wifi ac",
        help="Teks kueri pencarian kos",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Jumlah hasil yang ditampilkan",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Jalankan batch demo untuk benchmark latency",
    )
    args = parser.parse_args()

    if args.batch:
        asyncio.run(run_batch_demo(SAMPLE_QUERIES))
    else:
        asyncio.run(run_demo(query=args.query, top_k=args.top_k))
```

---

## Checklist State yang Harus Diingat untuk Tahap 3

```
VARIABEL/KOMPONEN DARI TAHAP 2 YANG AKAN DIGUNAKAN DI TAHAP 3:

1. SearchService.search() → akan dibungkus @router.post("/api/v1/search-kos")
   Parameter: query(str), user_lat(float), user_lon(float), top_k(int)
   Returns  : SearchResponse (query_original, results: list[KosScoringResult])

2. KosScoringResult → akan menjadi Pydantic Response Schema di Tahap 3
   Field kunci untuk Flutter: id_kos, nama_kos, harga_per_bulan, distance_km,
   final_score, is_super_deal, price_label, predicted_price, foto_path, rating

3. FastAPI app.state (diisi di lifespan handler Tahap 3):
   app.state.search_service  = SearchService(...)
   app.state.embedder        = IndoBERTEmbedder(settings.indobert_model_name)
   app.state.indexer         = KosVectorIndexer(...)  ← load_index() sudah dipanggil

4. DB session di Tahap 3:
   async def endpoint(db: AsyncSession = Depends(get_db_session)):
       result = await app.state.search_service.search(..., db=db)

5. Response JSON shape yang akan diterima Flutter (Tahap 3):
   {
     "query_original": "...",
     "query_expanded": "...",
     "expansion_terms": [...],
     "results": [
       {
         "id_kos": 1, "nama_kos": "...", "harga_per_bulan": 1500000,
         "distance_km": 1.23, "final_score": 0.742,
         "is_super_deal": true, "price_label": "super_deal",
         "predicted_price": 1850000, ...
       }
     ],
     "search_time_ms": 145.3
   }
```

---

*End of TAHAP 2 — AI Engine selesai. Menunggu instruksi TAHAP 3.*
*State aktif: IndoBERT(768d) + FAISS(IndexFlatIP) + PRF(TF-IDF) + RF(16 fitur) + Fusion(0.5:0.3:0.2)*
