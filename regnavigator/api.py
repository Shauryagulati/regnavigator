import time
import uuid
import logging
from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_JURISDICTION
from .retriever import HybridRetriever
from .llm import LLM

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session store: session_id -> [{"role": "user"/"assistant", "content": "..."}]
_sessions: Dict[str, List[Dict[str, str]]] = {}


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    jurisdiction: Optional[str] = Field(None, description="Two-letter jurisdiction like CA, CO, EU")
    top_k: int = Field(6, ge=1, le=20)
    session_id: Optional[str] = None


def _build_contextual_query(query: str, history: List[Dict[str, str]]) -> str:
    if not history:
        return query
    recent = history[-6:]
    parts = []
    for msg in recent:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    ctx = " | ".join(parts)
    return f"Context: {ctx}\n\nCurrent question: {query}"

def rewrite_query(latest_query: str, history: List[Dict[str, str]], llm: LLM) -> str:
    """
    Rewrites the user's latest query into a fully-contextualized standalone question.
    """

    if not history:
        return latest_query

    # Use last 10 messages max
    trimmed = history[-10:]

    conv = ""
    for msg in trimmed:
        role = msg.get("role")
        content = msg.get("content")
        conv += f"{role.capitalize()}: {content}\n"

    prompt = f"""
Rewrite the user's latest query into a complete, standalone question using context from the conversation.

Conversation History:
{conv}

Latest Query: {latest_query}

Rules:
- The rewritten question must contain full context.
- Replace pronouns like "it", "they", "that bill" with explicit references.
- Do NOT answer the question.
- Return ONLY the rewritten query.
"""

    rewritten = llm.provider.chat(
        system="You are a query-rewriting assistant.",
        user=prompt,
        temperature=0.1
    )

    return rewritten.strip()


@router.post("/chat")
async def chat(req: ChatRequest):
    t0 = time.time()
    session_id = req.session_id or str(uuid.uuid4())
    jurisdiction = (req.jurisdiction or DEFAULT_JURISDICTION).upper()

    # Load past messages
    history = _sessions.get(session_id, [])

    # Create LLM instance (for rewriting + answering)
    llm = LLM()
    if not llm.available():
        raise HTTPException(status_code=503, detail="LLM unavailable")

    # --- STEP 1: Rewrite query based on conversation ---
    rewritten_query = rewrite_query(req.query, history, llm)

    # --- STEP 2: Retrieve using REWRITTEN QUERY ---
    try:
        retriever = HybridRetriever(jurisdiction)
        hits = retriever.retrieve(rewritten_query, top_k=req.top_k)
    except Exception as e:
        logger.error(f"[{session_id}] Retrieval error: {e}")
        raise HTTPException(status_code=503, detail="Retrieval unavailable")

    # --- STEP 3: Generate answer (using original user query) ---
    try:
        answer = llm.answer(req.query, hits)
    except Exception as e:
        logger.error(f"[{session_id}] LLM error: {e}")
        raise HTTPException(status_code=500, detail="Answer generation failed")

    # --- STEP 4: Update conversation memory ---
    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": answer})
    _sessions[session_id] = history[-12:]  # cap history

    # --- STEP 5: Citations ---
    citations = []
    for i, h in enumerate(hits, start=1):
        m = h.get("meta", {})
        citations.append({
            "n": i,
            "chunk_id": m.get("chunk_id"),
            "source_file": m.get("source_file"),
            "jurisdiction": m.get("jurisdiction"),
            "page": m.get("page"),
            "header": m.get("header"),
            "char_start": m.get("char_start"),
            "char_end": m.get("char_end"),
        })

    return {
        "answer": answer,
        "rewritten_query": rewritten_query,   # 🚀 useful for debugging
        "citations": citations,
        "session_id": session_id,
        "metadata": {
            "jurisdiction": jurisdiction,
            "duration_ms": int((time.time() - t0) * 1000),
            "hits_count": len(hits),
        },
    }



@router.get("/health")
async def health():
    # Very simple health check
    try:
        _ = HybridRetriever(DEFAULT_JURISDICTION)
        llm = LLM()
        return {
            "status": "ok",
            "llm_available": llm.available()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }
