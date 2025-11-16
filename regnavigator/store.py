import os
import chromadb
from typing import List, Dict, Any, Optional
import logging
from .config import CHROMA_DIR, MERGE_WEIGHT_DENSE, MERGE_WEIGHT_SPARSE
from .bm25_store import BM25Store

logger = logging.getLogger(__name__)


def _minmax(xs: List[float]) -> List[float]:
    """Min-max normalize scores to [0, 1] range"""
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [1.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


class VectorStore:
    """
    Vector store using ChromaDB (dense only) + BM25 (sparse)
    
    NOTE: Local ChromaDB does NOT support sparse embeddings.
    Only dense vectors are stored in ChromaDB.
    BM25 provides the sparse/keyword matching component.
    """
    
    def __init__(self, jurisdiction: str):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        
        self.jurisdiction = jurisdiction.upper()
        self.collection_name = f"regulations_{jurisdiction.lower()}"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        
        # Create/get collection (cosine similarity for dense vectors)
        self.collection = self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize BM25 for sparse/keyword matching
        self.bm25 = BM25Store(jurisdiction)
        
        logger.info(f"[store] Initialized for {jurisdiction}")
        logger.info(f"[store] ChromaDB: {self.collection_name} (dense vectors only)")
        logger.info(f"[store] BM25: keyword matching")
    
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
        sparse_embeddings: Optional[List[Dict[str, List[float]]]] = None
    ):
        """
        Add documents to vector store
        
        Args:
            ids: Document IDs
            documents: Text content
            metadatas: Metadata dicts
            embeddings: Dense vectors (1024-dim from BGE-M3)
            sparse_embeddings: IGNORED (local ChromaDB doesn't support)
        
        Note: sparse_embeddings parameter is kept for API compatibility
        but is NOT used. BM25 handles sparse matching instead.
        """
        if sparse_embeddings:
            logger.debug("[store] sparse_embeddings provided but ignored (using BM25 instead)")
        
        # Add to ChromaDB (dense vectors only)
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            logger.info(f"[store] Added {len(ids)} documents to ChromaDB (dense)")
        except Exception as e:
            logger.error(f"[store] ChromaDB error: {e}")
            raise
        
        # Add to BM25 (keyword/sparse matching)
        try:
            self.bm25.add_docs(documents, ids)
            logger.info(f"[store] Added {len(ids)} documents to BM25 (sparse)")
        except Exception as e:
            logger.error(f"[store] BM25 error: {e}")
            raise
    
    def query_hybrid(
        self,
        query_text: str,
        query_dense: List[float],
        query_sparse: Optional[Dict[str, List[float]]] = None,
        top_k: int = 30,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Hybrid search: Dense (ChromaDB) + Sparse (BM25)
        
        Args:
            query_text: Query string (for BM25)
            query_dense: Dense embedding vector
            query_sparse: IGNORED (kept for API compatibility)
            top_k: Number of results
            where: Metadata filters
        
        Returns:
            Dict with documents, metadatas, ids, and hybrid scores
        """
        if query_sparse:
            logger.debug("[store] query_sparse provided but ignored (using BM25 instead)")
        
        # 1. Dense retrieval from ChromaDB
        try:
            # Build query parameters
            query_params = {
                "query_embeddings": [query_dense],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Only add 'where' if it has actual filters
            if where and len(where) > 0:
                query_params["where"] = where
            
            dense_results = self.collection.query(**query_params)
            
        except Exception as e:
            logger.error(f"[store] ChromaDB query error: {e}")
            raise
        
        # Extract results (IDs are returned automatically)
        docs = dense_results.get("documents", [[]])[0]
        metas = dense_results.get("metadatas", [[]])[0]
        ids = dense_results.get("ids", [[]])[0]
        dists = dense_results.get("distances", [[]])[0]
        
        if not ids:
            logger.warning("[store] No results from ChromaDB")
            return {
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]],
                "scores": [[]],
                "scores_dense": [[]],
                "scores_bm25": [[]]
            }
        
        # Convert distances to similarities (cosine: 1 - distance)
        s_dense = [1.0 - float(d) for d in dists]
        
        # 2. BM25 retrieval (sparse/keyword matching)
        try:
            bm25_hits = dict(self.bm25.search(query_text, top_k=top_k))
        except Exception as e:
            logger.error(f"[store] BM25 query error: {e}")
            bm25_hits = {}
        
        # Get BM25 scores for the ChromaDB results
        s_sparse = [bm25_hits.get(_id, 0.0) for _id in ids]
        
        # 3. Normalize and merge scores
        s_dense_norm = _minmax(s_dense)
        s_sparse_norm = _minmax(s_sparse)
        
        # Weighted hybrid score
        s_merged = [
            MERGE_WEIGHT_DENSE * sd + MERGE_WEIGHT_SPARSE * sb
            for sd, sb in zip(s_dense_norm, s_sparse_norm)
        ]
        
        logger.info(f"[store] Retrieved {len(ids)} results (dense + BM25 hybrid)")
        
        return {
            "documents": [docs],
            "metadatas": [metas],
            "ids": [ids],
            "scores": [s_merged],           # Hybrid scores
            "scores_dense": [s_dense],      # Raw dense scores
            "scores_bm25": [s_sparse]       # Raw BM25 scores
        }
    
    def count(self) -> int:
        """Get document count"""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"[store] Cleared ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"[store] Error clearing collection: {e}")