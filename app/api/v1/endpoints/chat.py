from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from app.services.rag_service import VisionaryRAGService
from app.core.database import get_db_session # Sesuaikan path file-nya kalau beda!
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import ChatHistory

router = APIRouter()

# Schema Input dari Flutter
class ChatInput(BaseModel):
    session_id: str
    message: str
    user_lat: float
    user_lon: float

# Schema Output Pintar ke Flutter
class ChatOutput(BaseModel):
    reply: str
    is_kos_related: bool
    suggested_action: str

@router.post("/chat", response_model=ChatOutput)
async def handle_makelar_chat(
    payload: ChatInput, 
    request: Request,
    db: AsyncSession = Depends(get_db_session)  # 🔥 SUNTIK KONEKSI DB DI SINI
):
    search_service = request.app.state.search_service
    if not search_service:
        raise HTTPException(status_code=503, detail="Mesin pencarian belum siap!")

    rag_service = VisionaryRAGService(search_service=search_service)
    
    # 1. Panggil fungsi RAG yang udah di-upgrade (CUKUP 1 KALI SAJA)
    response_data = await rag_service.generate_smart_reply(
        session_id=payload.session_id,  # Kirim ID Sesi
        user_message=payload.message,
        lat=payload.user_lat,
        lon=payload.user_lon,
        db=db
    )
    
    # 2. Simpan percakapan ini ke PostgreSQL biar AI-nya ingat!
    db.add(ChatHistory(session_id=payload.session_id, role="user", message=payload.message))
    db.add(ChatHistory(session_id=payload.session_id, role="model", message=response_data["reply"]))
    await db.commit() # Kunci permanen di DB
    
    return response_data