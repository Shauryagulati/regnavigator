from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import BGE_RERANKER

class BGEReranker:
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or BGE_RERANKER
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passages: List[str]) -> List[float]:
        batch = self.tok([query] * len(passages), passages, truncation=True,
                        padding=True, return_tensors="pt", max_length=512).to(self.device)
        with torch.inference_mode():
            out = self.model(**batch).logits.squeeze(-1).detach().cpu().tolist()
        return [float(x) for x in out] if isinstance(out, list) else [float(out)]

def min_max_norm(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-9:
        return [1.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]
