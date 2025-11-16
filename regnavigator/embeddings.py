from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try sentence-transformers first
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("[embeddings] sentence-transformers not available")

class EmbeddingModel:
    """
    E5 Embedding Model
    
    E5 is excellent for retrieval tasks and requires "query: " / "passage: " prefixes.
    Falls back to all-MiniLM-L6-v2 if E5 not available.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize E5 model
        
        Args:
            model_name: Model name (default: intfloat/e5-large-v2)
                       Alternative: intfloat/e5-base-v2 (faster, smaller)
                       Alternative: all-MiniLM-L6-v2 (proven fallback)
            device: 'cuda' or 'cpu' (auto-detected if None)
            batch_size: Encoding batch size
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"[embeddings] Loading {model_name} on {device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            logger.info(f"[embeddings] ✅ {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"[embeddings] Failed to load {model_name}: {e}")
            # Fallback to proven model
            logger.info("[embeddings] Falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            self.model_name = 'all-MiniLM-L6-v2'
        
        self.device = device
        self.batch_size = batch_size
        self.use_e5_prefix = 'e5' in self.model_name.lower()
    
    def encode_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        Encode documents for indexing
        
        Args:
            texts: List of document texts
            
        Returns:
            Dict with 'dense' and 'sparse' keys
        """
        if not texts:
            return {"dense": [], "sparse": []}
        
        # E5 requires "passage: " prefix for documents
        if self.use_e5_prefix:
            prefixed_texts = [f"passage: {text}" for text in texts]
        else:
            prefixed_texts = texts
        
        # Encode with normalization
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # L2 normalize
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        logger.info(
            f"[embeddings] Encoded {len(texts)} texts -> "
            f"{embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}"
        )
        
        return {
            "dense": embeddings.tolist(),
            "sparse": [{"indices": [], "values": []} for _ in texts]
        }
    
    def encode_query(self, text: str) -> Dict[str, Any]:
        """
        Encode query for search
        
        Args:
            text: Query string
            
        Returns:
            Dict with 'dense' and 'sparse' keys
        """
        # E5 requires "query: " prefix for queries
        if self.use_e5_prefix:
            prefixed_text = f"query: {text}"
        else:
            prefixed_text = text
        
        # Encode with normalization
        embedding = self.model.encode(
            [prefixed_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        return {
            "dense": embedding.tolist(),
            "sparse": {"indices": [], "values": []}
        }
    
    def get_embedding_dim(self) -> int:
        """Get dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()


# For backwards compatibility
def get_embedding_model(**kwargs) -> EmbeddingModel:
    """Get or create embedding model instance"""
    return EmbeddingModel(**kwargs)