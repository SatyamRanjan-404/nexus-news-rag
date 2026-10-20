# Architecture Decision Records — Nexus News RAG

> Append-only. One record per significant technical decision.  
> Format: ADR-NNN | Date | Status | Context → Options → Decision → Consequences

---

## ADR-001 | 2026-10-20 | Accepted
### Package layout: `src/nexus/` flat-module sub-packages

**Context:** Original code is a collection of flat scripts with no namespace, making imports fragile and testing impossible without mocking files.

**Options:**
1. Keep flat scripts — simple but cannot namespace or pip-install.
2. `src/nexus/` layout — standard Python packaging, enables `pip install -e .`, importable in tests.
3. Monorepo with separate api/ui packages — overkill.

**Decision:** Option 2. Sub-packages: `config`, `ingestion`, `chunking`, `embeddings`, `indexing`, `retrieval`, `rerank`, `generation`, `claims`, `evaluation`, `api`, `ui`.

**Consequences:** Imports become `from nexus.ingestion.fetcher import ...`. Requires `pyproject.toml`. Old flat scripts preserved in `scripts/legacy/`.

---

## ADR-002 | 2026-10-20 | Accepted
### Config: pydantic-settings + `.env`

**Context:** `load_dotenv()` scattered across files; no validation, no type safety.

**Decision:** One `nexus/config/settings.py` `Settings(BaseSettings)` class. All modules import `get_settings()`. `.env.example` committed; `.env` gitignored.

---

## ADR-003 | 2026-10-20 | Accepted
### Full-text extraction: trafilatura with RSS summary fallback

**Context:** Only 2-3 sentence RSS summaries are embedded. Chunking 500-char text is pointless. Retrieval is fundamentally limited by content depth.

**Options:** RSS only | BeautifulSoup scraping | trafilatura | newspaper3k

**Decision:** `trafilatura`. Try full text via `trafilatura.fetch_url()` → if None or <200 chars, fall back to RSS `<summary>`. Record method in metadata (`full_text` vs `summary_fallback`). 3s timeout per article.

---

## ADR-004 | 2026-10-20 | Accepted
### HTML cleaning: BeautifulSoup, not regex

**Context:** `re.compile('<.*?>')` fails on multi-line tags, doesn't decode HTML entities.

**Decision:** `BeautifulSoup(html, 'html.parser').get_text()` in `nexus/ingestion/utils.py`.

---

## ADR-005 | 2026-10-20 | Accepted
### Chunking: both RecursiveCharacter AND SemanticChunker behind one interface

**Context:** Resume claims both. Need to A/B evaluate.

**Decision:** `BaseChunker` interface; `RecursiveChunker` and `SemanticChunker` implementations. `ChunkerFactory.create(strategy)`. Baseline: recursive 500/50.

**Consequences:** SemanticChunker calls embedding model during chunking (slower ingestion). Only used in eval experiments and final config if data supports.

---

## ADR-006 | 2026-10-20 | Accepted
### Retrieval: BM25 hybrid with Reciprocal Rank Fusion (RRF)

**Context:** Dense-only misses exact entity/number matches. News is entity-heavy.

**Decision:** `rank_bm25` (pure Python) + ChromaDB dense, fused via RRF (score = Σ 1/(k+rank_i), k=60). BM25 index rebuilt at startup from stored documents (~1s for 10k docs).

---

## ADR-007 | 2026-10-20 | Accepted
### Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Context:** Resume claims reranking. Cross-encoders jointly encode query+doc, far more accurate than bi-encoder cosine similarity.

**Decision:** Retrieve top-20 candidates, rerank with cross-encoder, return top-5. Candidate pool size is an experiment variable (10/20/30/50).

**Consequences:** ~1-3s latency for rerank. Configurable via `RERANK_CANDIDATE_K` and `FINAL_K`.

---

## ADR-008 | 2026-10-20 | Accepted
### LLM: Groq `llama-3.3-70b-versatile` for generation, `llama-3.1-8b-instant` for judge

**Context:** Resume specifies Groq. Different generator/judge models reduce self-preference bias.

**Decision:** Generator: llama-3.3-70b-versatile (temp 0.2 for QA, 0 for claims). Judge: llama-3.1-8b-instant (temp 0, structured JSON output).

---

## ADR-009 | 2026-10-20 | Accepted
### FastAPI: separate API process from Streamlit UI

**Context:** Resume requires FastAPI endpoints. Running both in one Streamlit process is an antipattern.

**Decision:** `nexus-api` (FastAPI, port 8000) + `nexus-ui` (Streamlit, port 8501). Streamlit calls FastAPI via HTTP. Docker Compose orchestrates both.

---

## ADR-010 | 2026-10-20 | Accepted
### Deduplication: exact URL hash + SimHash on article body

**Context:** Syndicated wire stories appear near-identically across feeds, wasting embedding tokens and polluting retrieval.

**Decision:** Two layers: (1) URL-based chunk ID for idempotent upsert, (2) SimHash on body text, Hamming distance ≤ 3 = near-duplicate, keep highest-tier source.

---

## ADR-011 | 2026-10-20 | Accepted
### Source tiering for claim checker

**Decision:**
- Tier 1: Government/official (PIB, RBI, SEBI) — 180-day retention
- Tier 2: Fact-checkers (Alt News, BOOM, Factly, Vishvas, PolitiFact, Snopes) — 365-day retention
- Tier 3: Credible news outlets — 14-day retention

Claim checker retrieval order: T2 → T1 → T3. Independent T3 sources counted by simhash grouping.

---

## ADR-012 | 2026-10-20 | Accepted
### Claim verdict: Pydantic structured output, temperature 0

**Decision:** `llm.with_structured_output(ClaimVerdict)`. Fields: `verdict`, `confidence`, `explanation`, `evidence[]`, `independent_sources`, `freshness_note`. Post-LLM citation verification: each evidence quote checked as substring of retrieved text (fuzzy threshold 0.85); failing quotes removed.

---

## ADR-013 | 2026-10-20 | Accepted
### Rate limiting: slowapi on FastAPI

**Decision:** Per-IP: 20 req/min on `/v1/ask`, 10 req/min on `/v1/check-claim`. `/v1/ingest` requires `X-API-Key` header.

---

## ADR-014 | 2026-10-20 | Accepted
### Query rewriting for follow-up questions

**Decision:** Lightweight LCEL chain (llama-3.1-8b-instant, temp 0) rewrites follow-up queries into standalone questions. Only activated when conversation history length > 0.

---

## ADR-015 | 2026-10-20 | Accepted
### Evaluation: frozen corpus snapshot, separate eval ChromaDB collection

**Decision:** `scripts/create_snapshot.py` dumps ChromaDB to `data/snapshots/YYYY-MM-DD.jsonl`. Eval loads into separate `news_eval_YYYY-MM-DD` collection. Baseline config always runnable from same snapshot.

---

## ADR-016 | 2026-10-20 | Accepted
### Naming fixes

**Decision:**
- `schedular.py` → `scripts/legacy/schedular.py` + new `nexus/ingestion/scheduler.py`
- `requirement.txt` → `requirements.txt`
- `debug_db.py` → `scripts/debug_db.py` (gitignored from root)
- `db_manager.py` (README fiction) → implemented as `nexus/indexing/chroma_store.py`

---

## ADR-017 | 2026-10-20 | Accepted
### Secrets audit result

**Finding:** `.env` was NEVER committed in any of the 3 commits. Git log grep for "GROQ" returned empty. **No key rotation required.** `.gitignore` correctly excludes `.env` from the initial commit.
