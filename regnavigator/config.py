import os
from dotenv import load_dotenv

load_dotenv()

# -------- Embeddings --------
EMBEDDING_MODEL = "intfloat/e5-large-v2"

# -------- Reranker --------
BGE_RERANKER = "BAAI/bge-reranker-base"

# -------- Hybrid dense + sparse weights (inside VectorStore) --------
# How much to trust dense vs BM25 when we combine them.
# 0.5 / 0.5 is a good default; you can tune later.
MERGE_WEIGHT_DENSE = float(os.getenv("MERGE_WEIGHT_DENSE", "0.5"))
MERGE_WEIGHT_SPARSE = float(os.getenv("MERGE_WEIGHT_SPARSE", "0.5"))

# -------- Hybrid vs Rerank (in retriever) --------
# RERANK_WEIGHT is applied in HybridRetriever: final = (1-w)*hybrid + w*rerank
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "0.55"))

# -------- LLM Provider --------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")  # "claude" or "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")

# -------- Default Jurisdiction --------
DEFAULT_JURISDICTION = os.getenv("DEFAULT_JURISDICTION", "CA").upper()

# -------- Data root directory for PDFs --------
# e.g. data/CA/pdfs, data/CO/pdfs, etc.
DATA_DIR = os.getenv("DATA_DIR", "data")

# -------- Storage paths --------
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")
BM25_DIR = os.getenv("BM25_DIR", "bm25_store")

# Ensure dirs
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(BM25_DIR, exist_ok=True)
