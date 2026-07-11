# RegNavigator

**Ask a regulatory question, get an answer grounded in the actual statute — cited down to the source file, page, and section.**

RegNavigator is a Retrieval-Augmented Generation (RAG) system for regulatory corpora (bills, statutes, guidance). It answers *strictly* from your own PDFs — no invented law — and every answer comes back with traceable citations (source file, jurisdiction, page, header, character offsets). It ships with a California AI/tech-law corpus (~50 bills and code sections) as a working example.

## Why it's different

The headline is **retrieval quality**. Most RAG demos do dense-vector search and stop there. RegNavigator layers:

- **Hybrid retrieval** — dense embeddings (E5 via ChromaDB) **plus BM25 sparse** keyword search, min-max normalized and blended with tunable weights. Catches both semantic matches *and* the exact statutory language (bill IDs like `AB 489`, code sections) that pure vectors miss.
- **Cross-encoder reranking** — a BGE reranker re-scores the top candidates before the final cut.
- **Grounded generation** — the LLM answers only from retrieved snippets and cites them; it refuses to fill gaps from memory.
- **Multi-turn context** — follow-ups ("and the penalties?") inherit prior context through lightweight session history.

## Architecture

```
Ingestion:
  PDFs → pages → chunks (600c / 100 overlap) → embeddings ─┬─→ ChromaDB (dense)
                                                           └─→ BM25 (sparse)

Query:
  question → contextual rewrite → hybrid retrieval → cross-encoder rerank → grounded LLM → answer + citations
```

| Layer | Files |
|---|---|
| API / app | `main.py`, `regnavigator/api.py` |
| Retrieval | `regnavigator/store.py` (hybrid), `regnavigator/retriever.py`, `regnavigator/reranker.py` |
| Ingestion | `regnavigator/ingest.py`, `regnavigator/chunker.py`, `regnavigator/loaders.py` |
| Embeddings | `regnavigator/embeddings.py` (E5, MiniLM fallback) |
| Providers | `regnavigator/llm.py`, `regnavigator/llm_providers.py` (OpenAI / Anthropic) |
| UI | `index.html` (Tailwind single-page client) |

## Quickstart

```bash
pip install -r requirements.txt

# Configure providers — create a .env with:
#   LLM_PROVIDER=openai            # or: anthropic
#   OPENAI_API_KEY=...             # and/or ANTHROPIC_API_KEY=...
#   (optional) MERGE_WEIGHT_DENSE, MERGE_WEIGHT_SPARSE, RERANK_WEIGHT

# Ingest the bundled CA corpus (builds chroma_store/ + bm25_store/)
python -m regnavigator.ingest

# Run the API
python main.py                     # serves http://127.0.0.1:8000

# Open the UI
open index.html                    # health indicator should read "API online"

# Sanity check env, deps, indexes, provider readiness
python check_system.py
```

Drop PDFs under `data/<JURISDICTION>/pdfs/` (e.g. `data/CA/pdfs/`) to index your own corpus.

## API

- `POST /api/v1/chat` — `{ query, jurisdiction?="CA", top_k?=6, session_id? }` → `{ answer, citations[], session_id, metadata }`
- `GET /api/v1/health` — readiness check (retriever + LLM availability)

## Tuning

| Knob | Where | Effect |
|---|---|---|
| `MERGE_WEIGHT_DENSE` / `MERGE_WEIGHT_SPARSE` | `.env` | Dense vs. keyword balance |
| `RERANK_WEIGHT` | `.env` | Hybrid vs. reranker blend in final ranking |
| chunk size / overlap | `regnavigator/chunker.py` | Recall vs. index size |
| `top_k` | request | Snippets fed to the LLM per answer |

## Deep dive

Full code map, data-flow, and sequence diagrams live in **[docs/README.md](docs/README.md)**.
