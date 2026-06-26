import re

def extract_search_intent(kueri: str):
    kueri_lower = kueri.lower()
    
    # 1. Ekstraksi Gender (Hard Constraint)
    tipe_kos = None
    if any(word in kueri_lower for word in ["putri", "cewek", "wanita", "puan"]):
        tipe_kos = "Putri"
    elif any(word in kueri_lower for word in ["putra", "cowok", "pria", "laki"]):
        tipe_kos = "Putra"
    elif "campur" in kueri_lower:
        tipe_kos = "Campur"
        
    # 2. Ekstraksi Fasilitas Wajib (Boolean Hard Filter)
    filters = {
        "ac": True if "ac" in kueri_lower else None,
        "kamar_mandi_dalam": True if any(word in kueri_lower for word in ["km dalam", "kamar mandi dalam", "kmd"]) else None,
        "wifi": True if any(word in kueri_lower for word in ["wifi", "wi-fi", "internet"]) else None,
        "parkir": True if "parkir" in kueri_lower else None
    }
    
    # 3. Ekstraksi Niat Harga (Untuk Bahan Boosting Skor)
    intent_murah = True if any(word in kueri_lower for word in ["murah", "murmer", "budget", "miring", "hemat"]) else False
    intent_eksklusif = True if any(word in kueri_lower for word in ["eksklusif", "exclusive", "mewah", "sultan", "vip"]) else False

    return tipe_kos, filters, intent_murah, intent_eksklusif