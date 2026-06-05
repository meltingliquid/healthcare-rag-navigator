# Daily Checkpoints — Team Verification Gates

Every checkpoint must be verified by all 3 members before anyone moves on to the next day's work. Mark items `[x]` when verified and commit the update.

---

## Day 1 Checkpoint — Contracts + Mock Data

| # | Verification Item | Owner | Status |
|---|---|---|---|
| 1.1 | `mock_chunks.json` (10 sample chunks) exists in S3 `chunks/` | Member A | [✔] |
| 1.2 | All 3 members can read `mock_chunks.json` from S3 (Console or boto3) | All | [ ✔] |
| 1.3 | Integration contracts agreed — chunk schema, `retrieve()` signature, `mask_phi()` signature (per `docs/04_collaboration_plan.md` Section 1) | All | [ ] |
| 1.4 | OpenAI embedding test returns 1536-dim vector for mock chunk text | Member B | [x] |
| 1.5 | Presidio detects and masks a sample name + SSN in a test string | Member C | [ ] |
| 1.6 | Input sanitizer blocks "Ignore all instructions" test case | Member C | [ ] |

**Gate:** Everyone confirms on group chat. Mark all `[x]` and commit.

---

## Day 2 Checkpoint — Real Chunks Delivered

| # | Verification Item | Owner | Status |
|---|---|---|---|
| 2.1 | PDF extraction works for at least 1 full chapter — raw text is clean | Member A | [ ] |
| 2.2 | Text cleaning removes headers/footers/page numbers correctly | Member A | [ ] |
| 2.3 | Member A delivers 10+ real extracted chunks to S3 `chunks/` | A → B | [ ] |
| 2.4 | Embedding pipeline processes the real chunks without errors | Member B | [x] |
| 2.5 | FHIR client fetches a test Patient resource from HAPI FHIR | Member C | [ ] |
| 2.6 | LangGraph skeleton runs in notebook (with mock retrieval) — graph compiles and returns a response | Member C | [ ] |

**Gate:** Member B confirms real chunks embed successfully. Member C confirms graph runs.

---

## Day 3 Checkpoint — Retriever Works with Full Data

| # | Verification Item | Owner | Status |
|---|---|---|---|
| 3.1 | Full chunking pipeline complete — all 3 manuals chunked | Member A | [ ] |
| 3.2 | `chunks.json` uploaded to S3 `chunks/` with correct schema | Member A | [ ] |
| 3.3 | FAISS index built from full chunks — `faiss_index.bin` on S3 | Member B | [ ] |
| 3.4 | BM25 index built — `bm25_index.pkl` on S3 | Member B | [ ] |
| 3.5 | Hybrid retriever returns relevant results for 3 different query types (semantic, keyword, mixed) | Member B | [ ] |
| 3.6 | LangGraph router correctly routes general vs patient-specific queries (5+ tests each) | Member C | [ ] |
| 3.7 | Session manager creates, stores, and expires sessions correctly | Member C | [ ] |

**Gate:** Member B demos retriever returning meaningful chunks. Member C demos routing logic.

---

## Day 4 Checkpoint — End-to-End Pipeline Works

| # | Verification Item | Owner | Status |
|---|---|---|---|
| 4.1 | Full notebook pipeline works: query → sanitize → mask → retrieve → rerank → LLM → response | All | [ ] |
| 4.2 | Patient-specific query path works (FHIR fetch → strip PHI → retrieve → answer with `patient_context_used: true`) | B + C | [ ] |
| 4.3 | Response includes citations matching `docs/01_architecture_overview.md` Section 8 format | Member B | [ ] |
| 4.4 | Audit logger writes masked entries to JSONL (no PHI in logs) | Member C | [ ] |
| 4.5 | Functional test cases F-01 through F-07 pass (per `docs/05_testing_evaluation.md`) | Member C | [ ] |
| 4.6 | At least 25 Q&A pairs each from A and B written for eval dataset | A + B | [ ] |
| 4.7 | All modules merged to `develop` branch — no broken imports | All | [ ] |

**Gate:** Full pipeline demo in the notebook — one general query AND one patient-specific query, both return answers with citations and confidence scores.

---

## Day 5 Checkpoint — Demo Ready

| # | Verification Item | Owner | Status |
|---|---|---|---|
| 5.1 | Eval dataset finalized — 50–100 Q&A pairs with correct category distribution (per doc 05 Section 5) | A + B | [ ] |
| 5.2 | RAGAS evaluation run completes — all 5 metrics logged, targets checked | Member B | [ ] |
| 5.3 | Edge cases E-01 through E-10 handled (per `docs/05_testing_evaluation.md` Section 2) | Member C | [ ] |
| 5.4 | Security tests S-01 through S-03 pass (per doc 05 Section 3) | Member C | [ ] |
| 5.5 | Streamlit demo app works — query input, answer display, citations, confidence | Member C | [ ] |
| 5.6 | All Databricks notebooks (01–07) run on `Shared-Capstone-Cluster` | All | [ ] |
| 5.7 | Code merged to `main` — tagged `releases/v1.0` | All | [ ] |
| 5.8 | Demo dry run completed — all 3 members walk through their section | All | [ ] |

**Gate:** Full dry run of the demo presentation. Ready for evaluation.
