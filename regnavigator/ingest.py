from typing import List, Dict, Any
from .loaders import find_pdf_files, load_pdf_pages
from .chunker import split_with_offsets, detect_header
from .embeddings import EmbeddingModel
from .store import VectorStore

def _sanitize_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

def ingest_all():
    embedder = EmbeddingModel()
    pdf_files = list(find_pdf_files())
    if not pdf_files:
        print("[ingest] no PDFs found under data/*/pdfs")
        return
    
    for pdf in pdf_files:
        path, jur, filename = pdf["path"], pdf["jurisdiction"], pdf["filename"]
        print(f"[ingest] Processing: {path} (jurisdiction={jur})")
        store = VectorStore(jur)
        texts, metas, ids = [], [], []
        idx = 0
        
        for page in load_pdf_pages(path, jur, filename):
            base = _sanitize_meta(page["meta"])
            current_header = None
            page_text = page.get("text", "").strip()
            if not page_text:
                continue
            
            for piece in split_with_offsets(page_text):
                passage = piece["text"].strip()
                if not passage:
                    continue

                # Detect header in this passage and remember it
                hdr = detect_header(passage)
                if hdr:
                    current_header = hdr
                
                chunk_id = f"{filename}|p{base.get('page', -1)}|{idx}"
                meta = _sanitize_meta({
                    **base,
                    "header": current_header or "",
                    "char_start": piece["start"],
                    "char_end": piece["end"],
                    "chunk_id": chunk_id,
                    "jurisdiction": jur,
                    "source_file": filename
                })
                texts.append(passage)
                metas.append(meta)
                ids.append(chunk_id)
                idx += 1
        
        if not texts:
            print(f"[ingest] no text from {path}")
            continue
        
        print(f"[ingest] encoding {len(texts)} passages...")
        try:
            enc = embedder.encode_texts(texts)
            store.add_documents(ids, texts, metas, enc["dense"], enc.get("sparse"))
            print(f"[ingest] ✓ DONE: {filename} ({len(texts)} chunks)")
        except Exception as e:
            print(f"[ingest] ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    ingest_all()
