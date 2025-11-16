from typing import List, Tuple
from rank_bm25 import BM25Okapi
from pathlib import Path
import pickle, re
from .config import BM25_DIR

_token = re.compile(r"[A-Za-z0-9_]+")
def _tok(s: str) -> List[str]:
    return _token.findall((s or "").lower())

class BM25Store:
    def __init__(self, jurisdiction: str):
        self.jurisdiction = jurisdiction.upper()
        self.path = Path(BM25_DIR) / f"bm25_{self.jurisdiction.lower()}.pkl"
        self.ids: List[str] = []
        self.docs_tok: List[List[str]] = []
        self.bm25 = None
        if self.path.exists():
            with open(self.path, "rb") as f:
                obj = pickle.load(f)
            self.ids = obj["ids"]
            self.docs_tok = obj["docs_tok"]
            self.bm25 = BM25Okapi(self.docs_tok)
            print(f"[bm25] loaded existing index for {self.jurisdiction}")

    def add_docs(self, docs: List[str], ids: List[str]):
        toks = [_tok(d) for d in docs]
        self.ids.extend(ids)
        self.docs_tok.extend(toks)
        self.bm25 = BM25Okapi(self.docs_tok)
        with open(self.path, "wb") as f:
            pickle.dump({"ids": self.ids, "docs_tok": self.docs_tok}, f)
        print(f"[bm25] added {len(docs)} docs")

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        if not self.bm25 or not self.ids:
            return []
        scores = self.bm25.get_scores(_tok(query))
        pairs = list(zip(self.ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]
