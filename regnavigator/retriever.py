from typing import List, Dict, Any
from .embeddings import EmbeddingModel
from .store import VectorStore
from .reranker import BGEReranker, min_max_norm
from .config import RERANK_WEIGHT

class HybridRetriever:
    def __init__(self, jurisdiction: str, rerank_weight: float = None):
        self.jurisdiction = jurisdiction.upper()
        self.store = VectorStore(self.jurisdiction)
        self.embedder = EmbeddingModel()
        self.reranker = BGEReranker()
        self.rerank_weight = RERANK_WEIGHT if rerank_weight is None else float(rerank_weight)

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        q = self.embedder.encode_query(query)
        res = self.store.query_hybrid(
            query_text=query,
            query_dense=q["dense"],
            query_sparse=q.get("sparse"),
            top_k=max(top_k * 3, 30),
            where={"jurisdiction": self.jurisdiction}
        )
        
        docs = res.get("documents", [[]])[0]
        if not docs:
            return []
        
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        base_scores = res.get("scores", [[]])[0]
        
        rr_raw = self.reranker.score(query, docs)
        rr_norm = min_max_norm(rr_raw)
        
        W_RR, W_HY = self.rerank_weight, 1.0 - self.rerank_weight
        out = []
        for _id, doc, meta, hy, rr, rrn in zip(ids, docs, metas, base_scores, rr_raw, rr_norm):
            out.append({
                "id": _id,
                "text": doc,
                "meta": meta,
                "score_hybrid": float(hy),
                "score_rerank": float(rr),
                "score_rerank_norm": float(rrn),
                "score_final": W_HY * float(hy) + W_RR * float(rrn)
            })
        
        out.sort(key=lambda x: x["score_final"], reverse=True)
        return out[:top_k]
