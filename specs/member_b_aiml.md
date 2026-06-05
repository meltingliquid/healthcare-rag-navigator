# Member B — AI/ML Engineer Task Spec

**Owner:** Member B  
**Directory:** `retrieval/`  
**Branch pattern:** `feature/embedding-*`, `feature/hybrid-*`, `feature/prompt-*`

---

## Day 1 — Embedding PoC + FAISS Test

### Task B1.1: OpenAI Embedding Proof of Concept
- **Reference:** `docs/01_architecture_overview.md` Section 1 (Decision #2: `text-embedding-3-small`, 1536-dim)
- **Input:** OpenAI API key in `.env`, `mock_chunks.json` from Member A
- **Output:** Working script that embeds sample text using `text-embedding-3-small`
- **Acceptance:**
  - [ ] Returns a 1536-dimensional vector
  - [ ] Handles API errors (rate limit, auth failure) gracefully
  - [ ] Verified with all 10 mock chunks
  - [ ] Embedding latency logged

### Task B1.2: FAISS Smoke Test
- **Reference:** `docs/03_data_engineering.md` — Notebook 06 (`faiss.IndexFlatIP` for cosine similarity)
- **Input:** 10 embeddings from B1.1
- **Output:** FAISS index created, queried, and results verified
- **Acceptance:**
  - [ ] `faiss.IndexFlatIP` with 1536 dimensions
  - [ ] Vectors normalized before adding (for cosine similarity)
  - [ ] Top-3 search returns correct chunk IDs
  - [ ] `chunk_id_map` correctly maps FAISS integer IDs → chunk_ids

### Task B1.3: Review Integration Contracts
- **Reference:** `docs/04_collaboration_plan.md` Section 1 — Integration Contracts table
- **Acceptance:**
  - [ ] Confirm chunk schema works for embedding pipeline
  - [ ] Confirm `retrieve(query: str, top_k: int = 5) -> list[dict]` signature with Member C
  - [ ] Confirm FAISS index + `chunk_id_map.json` handoff format with Member C

---

## Day 2 — Embedding Pipeline on Real Chunks

### Task B2.1: Embedding Pipeline Script
- **Reference:** `docs/03_data_engineering.md` — Notebook 05 + embedding code sample
- **Input:** Chunks JSON (real chunks from Member A)
- **Output:** `retrieval/embed_chunks.py`
- **Acceptance:**
  - [X] Batches at 100 chunks per API call with 1-second delay (per `docs/03_data_engineering.md` rate limiting code)
  - [X] Saves embeddings as NumPy array (`embeddings.npy`)
  - [X] Saves `chunk_id_map.json` mapping FAISS integer index → chunk_id
  - [X] Handles interruption gracefully (can resume from last batch)
  - [X] Cost estimate logged (~$0.50 for 20K chunks per `docs/06_assumptions_tradeoffs.md` Assumption #10)

### Task B2.2: Verify Real Chunks from Member A
- **Acceptance:**
  - [ ] 10+ real chunks loaded successfully from S3
  - [ ] Schema matches `mock_chunks.json` exactly (all fields present)
  - [ ] Embeddings generated without errors
  - [ ] Any schema issues flagged back to Member A immediately

---

## Day 3 — Index Building + Hybrid Retriever

### Task B3.1: FAISS Index Builder
- **Reference:** `docs/03_data_engineering.md` — Notebook 06 + `docs/01_architecture_overview.md` Section 4
- **Input:** `embeddings.npy` + `chunk_id_map.json`
- **Output:** `retrieval/build_faiss_index.py` → `faiss_index.bin`
- **Acceptance:**
  - [X] `faiss.IndexFlatIP` for cosine similarity (vectors normalized)
  - [X] Index saved to S3 `indexes/faiss/faiss_index.bin`
  - [X] `chunk_id_map.json` saved to S3 `config/chunk_id_map.json`
  - [X] Top-20 retrieval works for a test query

### Task B3.2: BM25 Index Builder
- **Reference:** `docs/01_architecture_overview.md` Section 4 — BM25 for exact term matching
- **Input:** `chunks.json` (`chunk_text` field)
- **Output:** `retrieval/build_bm25_index.py` → `bm25_index.pkl`
- **Acceptance:**
  - [X] Uses `rank_bm25.BM25Okapi`
  - [X] Tokenization matches search-time tokenization
  - [X] Index saved to S3 `indexes/bm25/bm25_index.pkl`
  - [X] Top-20 retrieval works — captures exact terms like "§40.1" or "HCPCS G0179" that FAISS misses

### Task B3.3: Hybrid Retriever (FAISS + BM25 + RRF)
- **Reference:** `docs/01_architecture_overview.md` Section 4 — Retrieve 20 → RRF merge → top 15 → rerank to 5
- **Input:** Both indexes + chunks
- **Output:** `retrieval/hybrid_retriever.py`
- **Acceptance:**
  - [X] FAISS returns top-20 candidates
  - [X] BM25 returns top-20 candidates
  - [X] Reciprocal Rank Fusion (RRF) merges and deduplicates → top 15
  - [X] `retrieve(query: str, top_k: int = 5) -> list[dict]` function exposed
  - [X] Each returned dict includes: `chunk_id`, `chunk_text`, `score`, `manual_id`, `chapter_title`, `section_title`, `page_num`
  - [X] Tested with 3 queries: one semantic ("home health coverage"), one keyword-heavy ("§40.1 HCPCS G0179"), one mixed

---

## Day 4 — Reranking + Prompt Engineering + Integration

### Task B4.1: Cross-Encoder Reranker
- **Reference:** `docs/01_architecture_overview.md` Section 4 step 4 + `docs/06_assumptions_tradeoffs.md` Section 2 (reranker: +100-200ms worth ity)
- **Input:** Top-15 candidates from hybrid retriever
- **Output:** `retrieval/reranker.py`
- **Acceptance:**
  - [X] Uses lightweight cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - [X] Re-scores and returns top-5 chunks
  - [X] Falls back to RRF scores if cross-encoder fails
  - [X] Latency is under 200ms for 15 candidates

### Task B4.2: Prompt Engineering
- **Reference:** `docs/01_architecture_overview.md` Section 8 (output format) + `docs/02_security_compliance.md` Section 2 (masked prompts only)
- **Input:** Top-5 chunks + optional FHIR patient context (stripped of PHI)
- **Output:** `retrieval/prompts/system_prompt.txt` + `retrieval/prompts/few_shot_examples.json`
- **LLM Config (per `docs/06_assumptions_tradeoffs.md` Section 2):**
  - Use `gpt-4o` for answer generation
  - Use `gpt-4o-mini` for query classification and eval (cheaper)
- **Acceptance:**
  - [X] System prompt instructs: cite sources, include confidence, never fabricate information not in context
  - [X] Few-shot examples show expected output format: `answer`, `citations[]`, `confidence{}` (per doc 01 Section 8)
  - [X] Handles both general and patient-specific query prompts
  - [X] Confidence has two signals: retrieval_score (cosine) + LLM self-assessment (High/Medium/Low) per doc 01 Section 7

### Task B4.3: Integration with Member C's Notebook Pipeline
- **Acceptance:**
  - [X] `retrieve()` callable as a Python function from a notebook cell
  - [X] Returns the format expected by the prompt builder
  - [X] End-to-end tested in notebook: query → retrieve → rerank → prompt → LLM → response

---

## Day 5 — Evaluation + Databricks Notebooks

### Task B5.1: RAGAS Evaluation Run
- **Reference:** `docs/05_testing_evaluation.md` Section 4 — RAGAS metrics and target scores
- **Input:** `evaluation/eval_dataset.json` (50–100 Q&A pairs)
- **Output:** `evaluation/run_ragas.py` + results in `evaluation/results/`
- **Target metrics (per doc 05):**
  - Context Precision ≥ 0.80
  - Context Recall ≥ 0.75
  - Faithfulness ≥ 0.90
  - Answer Relevance ≥ 0.85
  - Hallucination Rate ≤ 5%
- **Acceptance:**
  - [ ] All 5 metrics computed and logged
  - [ ] LLM-as-Judge faithfulness check implemented (per doc 05 Section 4)
  - [ ] Results saved as JSON
  - [ ] At least one full evaluation run completed
  - [ ] Weak areas identified and documented

### Task B5.2: Write Q&A Pairs — Coverage + Claims
- **Reference:** `docs/05_testing_evaluation.md` Section 5 — Category Distribution
- **Category split (per doc 05):**
  - 15 Coverage scope questions ("Is X covered?", "What does Part A cover?")
  - 10 Claims processing questions ("How to file...", "Billing codes for...")
- **Acceptance:**
  - [ ] 25 pairs in the correct eval format (per doc 05)
  - [ ] Each references correct source manual/chapter/section
  - [ ] Member A validates 5 random pairs against chunks

### Task B5.3: Create Databricks Notebooks
- **Reference:** `docs/08_azure_databricks_setup.md` Section 3
- **Input:** Local scripts from `retrieval/`
- **Output:** `notebooks/05_embed.py`, `notebooks/06_build_index.py`
- **Acceptance:**
  - [ ] S3 connection + OpenAI key configured at top of each notebook
  - [ ] Runs on `Shared-Capstone-Cluster`
  - [ ] Produces same indexes as local scripts
