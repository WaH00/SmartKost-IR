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