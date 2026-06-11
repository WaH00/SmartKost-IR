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