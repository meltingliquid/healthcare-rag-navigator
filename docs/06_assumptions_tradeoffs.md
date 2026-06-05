# Assumptions, Trade-offs & Improvements — MVP

## 1. Assumptions

| # | Assumption | Impact if Wrong |
|---|-----------|-----------------|
| 1 | OpenAI API key available for `gpt-4o` + `text-embedding-3-small` | If quota issues, switch to `gpt-4o-mini` (cheaper, slightly less accurate) |
| 2 | Public HAPI FHIR (`hapi.fhir.org/baseR4`) is available and responsive | Cache sample responses locally as fallback |
| 3 | CMS manuals are text-extractable PDFs (not scanned images) | If scanned, need OCR (adds complexity) — but they are text PDFs ✅ |
| 4 | 3 CMS manuals (100-02, 100-03, 100-04) are sufficient scope | If more needed, pipeline scales — just add PDFs |
| 5 | In-memory FAISS is fine for ~5K-20K chunks | ✅ FAISS handles millions; 20K is trivial |
| 6 | English-only queries | Multi-language = entire translation layer |
| 7 | Single-user demo (no concurrency concerns) | If multi-user, need async FastAPI + index locking |
| 8 | Azure Databricks workspace accessible to all 3 members | If not, notebooks run as local Python scripts |
| 9 | No CMS manual updates during project | One-time ingestion is sufficient |
| 10 | OpenAI API costs manageable (~$5-15 for full project) | Embedding 20K chunks ≈ $0.50; 100 eval queries ≈ $5 |

---

## 2. Trade-offs (MVP-Specific)

| Decision | Options | Our Choice | Why |
|----------|---------|-----------|-----|
| **Embedding model** | OpenAI `3-small` vs `3-large` | `3-small` (1536-dim) | 5x cheaper, only ~2-4% less accurate |
| **Chunk size** | 256 / 500 / 1024 tokens | ~500 tokens | Balance: 256 splits mid-concept; 1024 dilutes relevance |
| **Top-K** | Retrieve 5 / 10 / 20 | Retrieve 20 → rerank to 5 | Broad retrieval + tight reranking = best precision |
| **Reranker** | Include or skip | ✅ Include | +100-200ms latency is worth the accuracy gain in healthcare |
| **LLM** | gpt-4o vs gpt-4o-mini | `gpt-4o` for answers; `gpt-4o-mini` for eval/classification | Cost optimization without quality sacrifice |
| **Vector DB** | FAISS vs managed service | In-memory FAISS | Free, fast, sufficient for MVP scale |
| **Session** | Stateless vs timed | 15-min in-memory dict | Enables follow-up questions; purges PHI on expiry |
| **Retrieval** | Vector-only vs Hybrid | Hybrid (FAISS + BM25) | CMS has exact codes/terms that pure vectors miss |
| **GraphRAG** | Build vs skip | ❌ Skip | Overkill for 3 hierarchical manuals |
| **Unity Catalog** | Use vs skip | ❌ Skip for MVP | JSON metadata is sufficient |
| **S3 structure** | Complex medallion vs simple | 3 folders: raw/processed/logs | No need for bronze/silver/gold at MVP scale |
| **Auth/RBAC** | Implement vs defer | ❌ Defer | Single-user demo |

---

## 3. Scope: In vs Out

### In Scope ✅ (MVP)
- CMS manual download, extraction, cleaning, chunking (3 manuals)
- OpenAI embeddings + FAISS index + BM25 index
- Hybrid retrieval (FAISS + BM25) + cross-encoder reranking
- FHIR patient data fetch at query time (public HAPI server)
- PHI masking with Microsoft Presidio
- LangGraph orchestration (routing + self-correction)
- FastAPI serving endpoint
- Confidence scoring + source citation
- In-memory session management (15-min TTL)
- Simple audit logging (masked JSONL)
- Prompt injection defense (regex-based)
- RAGAS evaluation with 50-100 Q&A pairs
- Databricks notebooks for data pipeline
- S3 for storage

### Out of Scope ❌ (Post-MVP)
- Table extraction from PDFs
- Multi-language support
- User authentication (OAuth/JWT)
- RBAC / IAM role configuration
- Encryption setup (S3 SSE-KMS)
- Production deployment (Kubernetes, CI/CD)
- Concurrent user handling
- Streaming / real-time data updates
- Unity Catalog / Delta Tables
- GPU clusters
- Semantic caching
- Custom fine-tuned models
- Knowledge graph / GraphRAG

---

## 4. Improvements Over Current Draft Architecture

| Aspect | Original Draft | MVP Architecture | Why Better |
|--------|---------------|-----------------|-----------|
| **Pipeline** | `Docs → Chunk → Embed → VectorDB → RAG → LLM` | Separate offline pipeline (Databricks) + query pipeline (LangGraph) | Clean separation; each can be developed independently |
| **Retrieval** | Vector-only (FAISS) | Hybrid (FAISS + BM25) + Reranker | Catches exact terms like "§40.1" that vectors miss |
| **Orchestration** | Linear chain | LangGraph with routing + self-correction | Routes general vs patient-specific queries; retries on low relevance |
| **PHI Handling** | "Security and PHI masking" (idea only) | Presidio integration with code | Production-verifiable, demo-ready |
| **FHIR Integration** | "Fetch patient data" (vague) | Scoped REST client → strip PHI → extract clinical context only | Clear data flow; PHI never reaches LLM |
| **Chunking** | Basic metadata | `parent_chunk_id` + deterministic `chunk_id` | Enables Parent Document Retrieval + deduplication |
| **Evaluation** | "Confidence score" (undefined) | RAGAS framework with 5 metrics + LLM-as-judge | Quantitative, systematic, presentation-ready |
| **Input Security** | Not mentioned | Regex-based prompt injection blocker | Prevents adversarial misuse |
| **Session** | "Timed session" (idea) | In-memory dict with 15-min TTL + cleanup | Concrete implementation |
| **Output** | "Answer + Source + Confidence" | Answer + Cited Manual/Chapter/Section + Confidence Level + Score | Fully traceable |
| **Error Handling** | Not addressed | Graceful fallbacks (FHIR down, out-of-scope, ambiguous) | Production mindset |

---

## 5. Presentation Tips

1. **Lead with the problem**: "Patients ask complex coverage questions. Existing tools hallucinate or mix up Part A vs Part B."

2. **Show the threat model**: Walk through PHI masking flow → demonstrates security maturity.

3. **Demo failure cases**: Show prompt injection blocked. Show out-of-scope handled. Senior engineers value resilience.

4. **Show quantitative results**: "Faithfulness score of 0.92 — 92% of claims are grounded in CMS policy text."

5. **Explain what you chose NOT to do**: "We evaluated GraphRAG and decided against it because..." — shows engineering judgment.

6. **Show the upgrade path**: "For MVP we use in-memory FAISS. For production, we'd switch to Mosaic AI Vector Search. The interface doesn't change."
