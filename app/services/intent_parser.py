import re
from typing import Dict, Any

class SmartIntentNER:
    def __init__(self):
        # Kamus sinonim untuk menangkap entitas fasilitas
        self.fasilitas_map = {
            "ac": "ac", "pendingin": "ac",
            "wifi": "wifi", "internet": "wifi",
            "kamar mandi dalam": "kamar_mandi_dalam", "km dalam": "kamar_mandi_dalam",
            "dapur": "dapur", "masak": "dapur",
            "parkir": "parkir", "motor": "parkir", "mobil": "parkir"
        }
        
    def extract_intent(self, query: str) -> Dict[str, Any]:
        query = query.lower()
        intent_data = {
            "budget_max": None,
            "gender": None,
            "fasilitas_wajib": [],
            "clean_query": query
        }

        # 1. Ekstraksi Entitas Harga (NER Budget)
        # Nangkep pola: "1.5 juta", "1,5jt", "1.500.000", "500rb"
        price_pattern = re.search(r'(\d+[\.,]?\d*)\s*(juta|jt|ribu|rb)', query)
        if price_pattern:
            angka = float(price_pattern.group(1).replace(',', '.'))
            satuan = price_pattern.group(2)
            
            if satuan in ['juta', 'jt']:
                intent_data["budget_max"] = int(angka * 1000000)
            elif satuan in ['ribu', 'rb']:
                intent_data["budget_max"] = int(angka * 1000)
                
            # Hapus teks harga dari kueri asli biar AI pencocok kata (FAISS) nggak bingung
            intent_data["clean_query"] = query.replace(price_pattern.group(0), "").strip()

        # 2. Ekstraksi Entitas Gender (Tipe Kos)
        if re.search(r'\b(putri|cewek|perempuan|cw)\b', query):
            intent_data["gender"] = "putri"
            intent_data["clean_query"] = re.sub(r'\b(putri|cewek|perempuan|cw)\b', '', intent_data["clean_query"]).strip()
        elif re.search(r'\b(putra|cowok|laki|cwk)\b', query):
            intent_data["gender"] = "putra"
            intent_data["clean_query"] = re.sub(r'\b(putra|cowok|laki|cwk)\b', '', intent_data["clean_query"]).strip()

        # 3. Ekstraksi Entitas Fasilitas
        for keyword, db_column in self.fasilitas_map.items():
            if keyword in query:
                if db_column not in intent_data["fasilitas_wajib"]:
                    intent_data["fasilitas_wajib"].append(db_column)

        return intent_data