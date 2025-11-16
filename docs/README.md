RegNavigator – Project Documentation

**Executive Summary**
- Purpose: Answer regulatory questions with grounded citations from your own PDF corpora (e.g., bills, statutes, guidance).
- Approach: Retrieval‑Augmented Generation (RAG) with hybrid retrieval (dense vectors + BM25), cross‑encoder reranking, and an LLM that answers strictly from retrieved snippets.
- Deliverables: FastAPI backend, Tailwind single‑page UI, ingestion pipeline, configurable providers (OpenAI/Anthropic), reproducible vector stores.

**System Architecture**
- Data ingestion: parse PDFs into overlapping text chunks, infer headers, compute embeddings, persist to ChromaDB (dense) and BM25 (sparse).
- Query serving: encode query, hybrid search (cosine + keyword), rerank with BGE cross‑encoder, prompt LLM with top snippets, return answer + citations.
- API and UI: FastAPI endpoints for chat and health; a minimal web client that displays answers and sources with page/jurisdiction.

**Data Flow**
- PDFs → pages → chunks → embeddings → stores → retrieval → reranking → answer + citations
- Chunking window: 600 chars with 100‑char overlap to improve recall around boundaries (`regnavigator/chunker.py:15`).
- Hybrid scoring: min‑max normalized dense similarity vs BM25 score with tunable weights (`regnavigator/store.py:171` and `regnavigator/config.py:15`).
- Final ranking: convex blend of hybrid score and normalized reranker score (`regnavigator/retriever.py:36`).

**Components (Code Map)**
- Chunking
  - `regnavigator/chunker.py:15` split_with_offsets: sliding window with overlap; preserves character offsets for citations.
  - `regnavigator/chunker.py:51` detect_header: heuristics for ARTICLE/SECTION, bill IDs (AB/SB), code sections.
- Loading
  - `regnavigator/loaders.py:7` find_pdf_files: discovers `data/<JURISDICTION>/pdfs/*.pdf`.
  - `regnavigator/loaders.py:20` load_pdf_pages: loads per‑page text via `PyPDFLoader`.
- Ingestion
  - `regnavigator/ingest.py:18` ingest_all: orchestrates pages → chunks → embeddings → stores; attaches page, header, offsets; IDs like `file.pdf|p12|34`.
- Embeddings
  - `regnavigator/embeddings.py:1` EmbeddingModel: defaults to E5 (`intfloat/e5-large-v2`) with “query:”/“passage:” prefixes; falls back to MiniLM if needed.
- Storage (Hybrid)
  - `regnavigator/store.py:21` VectorStore: ChromaDB for dense vectors, BM25 pickle index for keyword recall.
  - `regnavigator/store.py:97` query_hybrid: queries Chroma, looks up BM25 scores for same IDs, normalizes, blends by `MERGE_WEIGHT_*`.
  - `regnavigator/bm25_store.py:1` BM25Store: persists tokenized docs to `bm25_store/bm25_<jur>.pkl`.
- Reranking
  - `regnavigator/reranker.py:1` BGEReranker: cross‑encoder (`BAAI/bge-reranker-base`) scoring query‑passage relevance.
- Retrieval Orchestrator
  - `regnavigator/retriever.py:15` HybridRetriever.retrieve: encode query, expand top‑K from hybrid, rerank, select final K.
- LLM Orchestration
  - `regnavigator/llm.py:32` LLM.answer: formats a grounded prompt with numbered snippets; enforces “use snippets only” behavior.
  - `regnavigator/llm_providers.py:62` get_provider: returns OpenAI or Anthropic provider based on `.env`. Note: `LLM_PROVIDER=claude` resolves to Anthropic provider.
- API
  - `regnavigator/api.py:44` POST `/api/v1/chat`: multi‑turn chat with short in‑memory session history; returns `answer`, `citations`, and metadata.
  - `regnavigator/api.py:103` GET `/api/v1/health`: lightweight readiness check (retriever + LLM availability).
  - `main.py:39` mounts router, enables CORS and GZip, exposes `/` and `/health`.
- Frontend
  - `index.html:1` and `regnav-frontend/index.html:1`: Tailwind UI; calls `/api/v1/chat`, shows answer and sources, tracks a `session_id` for multi‑turn context.
- Diagnostics
  - `check_system.py:1` CLI checks for `.env`, deps, data presence, store population, and import‑readiness.

**Setup & Running**
- Prereqs: Python 3.10+, CUDA optional for reranker; network access to download models on first run.
- Install: `pip install -r requirements.txt`.
- Configure `.env` (examples):
  - `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` (or `claude` → Anthropic)
  - `LLM_MODEL=gpt-4o-mini` or `claude-3-5-sonnet-20241022`
  - `OPENAI_API_KEY=...` and/or `ANTHROPIC_API_KEY=...`
  - Optional tuning: `MERGE_WEIGHT_DENSE`, `MERGE_WEIGHT_SPARSE`, `RERANK_WEIGHT`
- Prepare data: place PDFs under `data/CA/pdfs` (and other jurisdictions similarly).
- Ingest: `python -m regnavigator.ingest` (creates/updates `chroma_store/` and `bm25_store/`).
- Run API: `python main.py` (serves on `http://127.0.0.1:8000`).
- Use UI: open `index.html` (or `regnav-frontend/index.html`) in a browser; health indicator should show “API online”.

**API Contract**
- POST `/api/v1/chat` (`regnavigator/api.py:44`)
  - Request: `{ query: str, jurisdiction?: str = "CA", top_k?: int = 6, session_id?: str }`
  - Response: `{ answer: str, citations: [{ n, chunk_id, source_file, jurisdiction, page, header, char_start, char_end }], session_id, metadata }`
- GET `/api/v1/health` (`regnavigator/api.py:103`)
  - Response: `{ status: "ok" | "degraded", llm_available?: bool, error?: str }`

**Citations & Traceability**
- Each snippet retains page number, source filename, jurisdiction, header, and character offsets.
- The UI displays chunk numbers and file/page badges; backend returns machine‑readable citation arrays for auditing.

**Tuning & Performance**
- Chunk size/overlap: adjust in `regnavigator/chunker.py:4-6` to trade recall vs. index size.
- Hybrid weights: `MERGE_WEIGHT_DENSE` vs `MERGE_WEIGHT_SPARSE` in `.env` (`regnavigator/config.py:15-16`).
- Rerank weight: `RERANK_WEIGHT` controls final blend of hybrid vs reranker (`regnavigator/config.py:19-20`).
- Top‑K fanout: retriever expands to ~3× target before rerank (`regnavigator/retriever.py:21`).
- Hardware: reranker benefits from GPU; embeddings are precomputed.

**Security & Privacy**
- Data stays local in ChromaDB/BM25 stores; only prompts/snippets go to the LLM provider.
- Redact or pre‑filter sensitive PDFs before ingestion if required; consider adding an allowlist for snippet export.
- Lock down CORS in `main.py:29-35` for deployment; add auth if exposing beyond trusted networks.

**Troubleshooting**
- Run `python check_system.py` to validate env, deps, indexes, provider readiness.
- Empty answers: verify ingestion ran and `chroma_store/` + `bm25_store/` are populated; confirm `.env` keys and provider.
- Slow/expensive queries: lower `top_k`, reduce `RERANK_WEIGHT`, or use a lighter reranker model.
- Model download issues: ensure network access on first run; pre‑populate Hugging Face caches when air‑gapped.

**Is It Multi‑Agent?**
- Current design: single‑pipeline RAG with one “answering” agent (LLM) operating over a deterministic retrieval stack.
- Why not multi‑agent yet:
  - The retrieval + rerank pipeline already reduces hallucination with grounded snippets.
  - Simplicity, latency, and cost: one LLM call per question (post‑retrieval) is predictable and fast.

**Path to Multi‑Agent**
- Planner agent: decomposes complex questions into sub‑queries, selects jurisdictions/sections to target.
- Retrieval agents: specialized dense/sparse retrievers per domain; could vote or union candidates.
- Critic/Judge agent: checks answer faithfulness to snippets; requests additional retrieval if gaps found.
- Citation agent: extracts and formats pinpoint citations (sections, paragraphs) and confidence scoring.
- Orchestrator: tool‑calling runtime that coordinates agents via function calls (e.g., “retrieve”, “rerank”, “expand_scope”).
- Implementation sketch:
  - Expose retriever/reranker as callable tools behind FastAPI or local Python functions.
  - Use a controller LLM to plan and call tools iteratively up to a budget (latency/cost guardrails).
  - Add automated self‑consistency (n‑best answers) and agreement checks before finalizing.

**Roadmap**
- Short‑term
  - Add per‑jurisdiction filters in the UI; allow dataset switching.
  - Streaming answers and partial result rendering.
  - Export answers + citations as a report (PDF/Docx).
- Mid‑term
  - Add structured evaluators: answer faithfulness and coverage tests with golden Q/A sets.
  - Scale reranker and caching; optional approximate BM25.
  - Add user auth and per‑tenant stores.
- Long‑term
  - Multi‑agent orchestration as above with planner/critic loops.
  - Domain‑adaptive reranker fine‑tuning on internal Q/A pairs.
  - Continuous ingestion watch to auto‑index new regulations.

**FAQ (Stakeholder Prep)**
- What guarantees correctness? Grounded prompting + citations, hybrid retrieval, and cross‑encoder reranking reduce hallucinations.
- Can it handle follow‑ups? Yes; short session history is folded into the retriever’s contextual query (`regnavigator/api.py:28, 52`).
- How do we add a new jurisdiction? Place PDFs under `data/<JURISDICTION>/pdfs`, run ingestion, then query with that code.
- What if providers are offline? The API returns clear errors; choose the other provider or run a local LLM as a future option.
- How big can the corpus be? ChromaDB handles large collections; BM25 and reranking scale linearly. Shard by jurisdiction as needed.

**Presenting the Demo**
- Start API: `python main.py` → check `/health` reports `{"status": "ok"}`.
- Open UI: `index.html`, ask a question, point out the “Sources Used” with file/page and headers.
- Show traceability: hover or read chunk numbers map to backend `citations` objects.

**Key Files (Quick Links)**
- `regnavigator/api.py:44` chat endpoint
- `regnavigator/retriever.py:15` retrieval and reranking
- `regnavigator/store.py:97` hybrid query
- `regnavigator/chunker.py:15` chunking
- `regnavigator/llm.py:32` grounded prompting
- `main.py:39` router mounting and app setup
- `index.html:1` frontend client

