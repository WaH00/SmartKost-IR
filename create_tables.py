import asyncio

# Pastikan path import ini sesuai dengan letak file database dan models lu
from app.core.database import async_engine, Base, ChatHistory
# Wajib import model tabelnya di sini biar SQLAlchemy ngebaca skemanya 

async def init_models():
    print("⏳ Menghubungkan ke PostgreSQL...")
    # Karena kita pakai async_engine, kita harus pakai 'run_sync'
    async with async_engine.begin() as conn:
        print("🔨 Sedang membangun tabel di database...")
        await conn.run_sync(Base.metadata.create_all)
        print("✅ BOOM! Tabel 'chat_histories' sukses diciptakan!")

if __name__ == "__main__":
    # Jalankan event loop async
    asyncio.run(init_models())