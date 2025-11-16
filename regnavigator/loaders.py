import os
from pathlib import Path
from typing import Dict, Generator
from langchain_community.document_loaders import PyPDFLoader
from .config import DATA_DIR

def find_pdf_files() -> Generator[Dict, None, None]:
    for root, _, files in os.walk(DATA_DIR):
        parts = Path(root).parts
        if len(parts) >= 2 and parts[-1] == "pdfs":
            jurisdiction = parts[-2].upper()
            for f in files:
                if f.lower().endswith(".pdf"):
                    yield {
                        "path": str(Path(root) / f),
                        "jurisdiction": jurisdiction,
                        "filename": f,
                    }

def load_pdf_pages(path: str, jurisdiction: str, filename: str):
    loader = PyPDFLoader(path)
    pages = loader.load()
    for idx, page in enumerate(pages):
        yield {
            "text": page.page_content or "",
            "meta": {
                "source_file": filename,
                "source_path": path,
                "jurisdiction": jurisdiction,
                "page": idx + 1,
            },
        }
