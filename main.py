# main.py

from dotenv import load_dotenv
load_dotenv()  # load OPENAI_API_KEY / ANTHROPIC_API_KEY / LLM_PROVIDER / LLM_MODEL

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# This must match your package structure: regnavigator/api.py with `router = APIRouter()`
from regnavigator.api import router as rag_router

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("regnavigator.main")

app = FastAPI(
    title="RegNavigator RAG API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middlewares ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- Include RAG router ---
app.include_router(rag_router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    return {
        "name": "RegNavigator RAG API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "chat_endpoint": "/api/v1/chat",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 RegNavigator RAG API starting up...")
    logger.info("✓ CORS enabled")
    logger.info("✓ GZip enabled")
    logger.info("✓ RAG router mounted at /api/v1")
    logger.info("🎯 Ready to serve")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
