import os
import re
import json
from sqlalchemy import select
from google import genai
from app.services.search_service import SearchService
from app.core.database import ChatHistory

class VisionaryRAGService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        # Gunakan SDK 2026 Terbaru
        self.client = genai.Client(api_key="MASUKKAN_GCP_API_KEY_DISINI")
        self.model_name = "gemini-3.5-flash"
        
        # Kata kunci terlarang buat ngeblokir user iseng
        self.out_of_scope_keywords = [
            "crypto", "bitcoin", "skripsi", "tugas kuliah", "coding", 
            "pacaran", "selingkuh", "politik", "presiden", "game", "mlbb"
        ]

    # ─────────────────────────────────────────────────────────
    # FUNGSI 1: PENARIK INGATAN (MEMORY FETCHER)
    # ─────────────────────────────────────────────────────────
    async def _get_clean_history(self, db, session_id: str) -> list[dict]:
        """Menarik 10 obrolan terakhir dari PostgreSQL biar token API gak jebol."""
        stmt = (
            select(ChatHistory)
            .where(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.timestamp.asc())
            .limit(10) 
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        # Format sesuai standar SDK google.genai
        return [{"role": row.role, "parts": [row.message]} for row in rows]

    # ─────────────────────────────────────────────────────────
    # FUNGSI 2: PENERJEMAH KONTEKS (QUERY REPHRASER)
    # ─────────────────────────────────────────────────────────
    async def _rephrase_query(self, user_message: str, chat_history: list) -> str:
        """Menggabungkan chat lama dan baru jadi satu kueri mesin pencari."""
        if not chat_history:
            return user_message
            
        prompt = f"""
        Tulis ulang pesan terbaru pengguna menjadi satu kalimat pencarian mandiri (standalone query) berdasarkan riwayat obrolannya.
        Riwayat: {chat_history}
        Pesan Terbaru: "{user_message}"
        ATURAN MUTLAK: Hanya kembalikan teks hasil tulisan ulang. Dilarang memberikan basa-basi atau penjelasan.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            return user_message # Fallback aman

    # ─────────────────────────────────────────────────────────
    # FUNGSI 3: JANTUNG UTAMA (GENERATE SMART REPLY)
    # ─────────────────────────────────────────────────────────
    async def generate_smart_reply(self, session_id: str, user_message: str, lat: float, lon: float, db) -> dict:
        message_clean = user_message.lower().strip()

        # 1. HARD GUARDRAILS: Cegat duluan sebelum buang-buang token API
        if any(kw in message_clean for kw in self.out_of_scope_keywords):
            return {
                "reply": "Waduh Bosku, otak makelar gua belum update buat bahas itu. Daripada pusing mikirin itu, mending kita cari kosan yang ada AC sama WiFi kencang biar lu bisa nugas atau nge-game dengan tenang! Yuk, mau cari di daerah mana?",
                "is_kos_related": False,
                "suggested_action": "REDIRECT_TO_SEARCH"
            }

        # 2. PANGGIL INGATAN & REPHRASE
        past_history = await self._get_clean_history(db, session_id)
        standalone_query = await self._rephrase_query(user_message, past_history)

        # 3. HYBRID SEARCH KE DATABASE (Pake Kueri Hasil Rephrase)
        search_results = await self.search_service.search(
            query=standalone_query, 
            user_lat=lat, 
            user_lon=lon,
            db=db
        )
        
        # 4. RANGKUM DATA KOS BUAT MAKANAN LLM
        context_kos = ""
        for i, kos in enumerate(search_results.results, 1):
            nama = getattr(kos, "nama_kos", f"Kosan ID {kos.id_kos}")
            harga = getattr(kos, "harga_per_bulan", "Menyesuaikan")
            tipe = getattr(kos, "tipe_kos", getattr(kos, "gender", "Campur")) 
            ac = getattr(kos, "ac", 0)
            wifi = getattr(kos, "wifi", 0)
            km_dalam = getattr(kos, "kamar_mandi_dalam", 0)
            alasan = getattr(kos, "highlight_alasan", "Fasilitas lengkap!")

            context_kos += f"""
            {i}. Kos: {nama} | Harga: Rp {harga}/bln | Tipe: {tipe}
               Fasilitas: AC={ac}, WiFi={wifi}, KM Dalam={km_dalam}
               Highlight: {alasan}
            \n"""

        # 5. SYSTEM PROMPT MAKELAR GAUL (Anti-Halusinasi)
        system_prompt = f"""
        Kamu adalah 'Bang Kos', asisten virtual/makelar AI super cerdas, asyik, dan jago jualan dari aplikasi Smart Kos.
        Gaya bicaramu harus gaul layaknya mahasiswa (gunakan kata: 'Bosku', 'Aman', 'Gass', 'Yoi', 'Mending').
        
        TUGAS UTAMAMU:
        Jawab pertanyaan berdasarkan data kosan real-time berikut:
        {context_kos if context_kos else "Tidak ada kosan spesifik yang cocok di database saat ini."}
        
        ATURAN KRITIS:
        1. JIKA DATA KOSAN KOSONG: Jangan mengarang data! Tawarkan user untuk mencari dengan kata kunci fasilitas atau budget lain.
        2. DILARANG KERAS MENGUBAH HARGA ATAU NAMA KOS. Jujurlah pada data yang diberikan.
        3. Jika pengguna bertanya di luar topik properti, gunakan teknik 'BROKER PIVOT': Jawab dengan candaan ringan, lalu alihkan kembali ke pentingnya mencari kos yang nyaman.
        
        Format jawabanmu senatural mungkin sebagai teman ngobrol.
        """

        # 6. EKSEKUSI CHAT BERKELANJUTAN (SDK 2026)
        try:
            chat_session = self.client.chats.create(
                model=self.model_name,
                history=past_history,
                config={"system_instruction": system_prompt}
            )
            
            response = chat_session.send_message(user_message)
            reply_text = response.text
            
            # Semantic Double-Check
            is_related = True
            if "kosan" not in reply_text.lower() and any(x in reply_text.lower() for x in ["otak", "patah hati", "resep", "maaf"]):
                is_related = False

            return {
                "reply": reply_text,
                "is_kos_related": is_related,
                "suggested_action": "NONE" if is_related else "SHOW_FUNNY_EMOTE"
            }
            
        except Exception as e:
            print(f"❌❌❌ ERROR DARI GEMINI: {str(e)} ❌❌❌")
            return {
                "reply": "Aman Bosku! Server makelar lagi tarik napas bentar karena kepenuhan. Coba ketik ulang pesannya ya!",
                "is_kos_related": True,
                "suggested_action": "RETRY"
            }