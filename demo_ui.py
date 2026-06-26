import streamlit as st
import requests

# Konfigurasi Halaman Web
st.set_page_config(page_title="Smart Kos AI", page_icon="🏠", layout="wide")

st.title("🚀 Smart Kos-Kosan AI")
st.markdown("Mesin Pencari Kos Berbasis **IndoBERT Semantik** & **PostGIS Spatial**")
st.divider()

# Bikin Sidebar buat Input
with st.sidebar:
    st.header("🎯 Parameter Pencarian")
    kueri = st.text_area("Deskripsikan Kos Idamanmu:", "kos murah fasilitas lengkap dekat kampus")
    
    st.subheader("📍 Titik Lokasi Kampus (Lat/Lon)")
    lat = st.number_input("Latitude", value=-6.1862, format="%.4f")
    lon = st.number_input("Longitude", value=106.8348, format="%.4f")
    
    top_k = st.slider("Jumlah Rekomendasi", min_value=1, max_value=10, value=5)
    
    tombol_cari = st.button("Cari Kos Sekarang 🔍", use_container_width=True)

# Main Area buat Nampilin Hasil
if tombol_cari:
    payload = {
        "kueri": kueri,
        "latitude": lat,
        "longitude": lon,
        "top_k": top_k
    }
    
    # Nembak API FastAPI lu yang lagi nyala
    API_URL = "http://127.0.0.1:8000/api/v1/search-kos"
    
    with st.spinner("AI sedang menganalisis jutaan kemungkinan..."):
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                hasil_kos = data.get("data", []) # Sesuaiin dengan struktur JSON lu
                
                # Tampilkan metrik kecepatan dan AI PRF
                col1, col2 = st.columns(2)
                col1.success(f"✅ Ditemukan {len(hasil_kos)} rekomendasi kos")
                # Kalau ada data latency atau kata kunci tambahan dari backend, bisa taruh sini
                
                st.subheader("🏆 Hasil Rekomendasi:")
                
                # Looping hasil pencarian dari JSON
                for i, kos in enumerate(hasil_kos):
                    with st.container():
                        st.markdown(f"### #{i+1} - {kos.get('nama_kos', 'Nama Tidak Diketahui')}")
                        
                        # Bikin kolom buat nampilin Harga dan Jarak biar rapi
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Harga Per Bulan", f"Rp {kos.get('harga_per_bulan', 0):,}")
                        c2.metric("Jarak ke Kampus", f"{kos.get('distance_km', 0):.2f} KM")
                        
                        # Indikator AI Pricing (Super Deal / Wajar)
                        status_harga = kos.get('kategori_harga', 'Wajar')
                        if status_harga == 'Super Deal':
                            c3.metric("Rekomendasi AI", "🔥 SUPER DEAL")
                        else:
                            c3.metric("Rekomendasi AI", "✅ Harga Wajar")
                        
                        # Expander buat detail fasilitas
                        with st.expander("Lihat Detail Fasilitas & Alamat"):
                            st.write(f"**Alamat:** {kos.get('kota', '')}, {kos.get('wilayah', '')}")
                            fasilitas = []
                            if kos.get('ac'): fasilitas.append("AC")
                            if kos.get('wifi'): fasilitas.append("WiFi")
                            if kos.get('kamar_mandi_dalam'): fasilitas.append("K. Mandi Dalam")
                            if kos.get('parkir'): fasilitas.append("Parkir")
                            
                            st.write(f"**Fasilitas Utama:** {', '.join(fasilitas)}")
                            st.write(f"**Skor Relevansi AI:** {kos.get('score', 0):.4f}")
                        
                        st.divider()
                        
            else:
                st.error(f"Pencarian Gagal. Status Code: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.ConnectionError:
            st.error("Gagal terhubung ke Backend! Pastikan Uvicorn FastAPI lu udah nyala di terminal sebelah.")