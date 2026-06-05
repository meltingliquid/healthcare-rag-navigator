# Member A — Data Engineer Task Spec

**Owner:** Member A  
**Directory:** `data/`  
**Branch pattern:** `feature/cms-*`, `feature/text-*`, `feature/chunking`

---

## Day 1 — CMS Download + Mock Data

### Task A1.1: Create mock_chunks.json
- **Priority:** CRITICAL (unblocks Members B and C)
- **Reference:** Chunk schema from `docs/01_architecture_overview.md` Section 5
- **Output:** `data/mock_chunks.json` — 10 sample chunks with all metadata fields
- **Acceptance:**
  - [X] Contains 10 chunks with valid `chunk_id`, `parent_chunk_id`, `manual_id`, `chapter_num`, `chapter_title`, `section_title`, `page_num`, `source_url`, `chunk_text`, `token_count`
  - [X] Uses deterministic ID format: `{manual_id}_ch{chapter_num}_s{section}_p{page}_{seq}`
  - [X] Uploaded to S3 `chunks/mock_chunks.json`
  - [ ] Members B and C confirm they can load it

### Task A1.2: CMS Download Script
- **Reference:** `docs/03_data_engineering.md` Section 1 — Notebook 01
- **Input:** CMS URLs for manuals 100-02, 100-03, 100-04
- **Output:** `data/download_cms.py` — downloads chapter PDFs to local `data/raw/` and to S3
- **Acceptance:**
  - [X] Downloads chapters from all 3 manuals (100-02, 100-03, 100-04)
  - [X] Handles network errors gracefully (retry/skip)
  - [X] PDFs land in S3 `raw/cms_manuals/{manual_id}/`
  - [X] CMS manuals are text-extractable PDFs (not scanned — confirmed in `docs/06_assumptions_tradeoffs.md` Assumption #3)

### Task A1.3: S3 Connection Verification
- **Reference:** `docs/07_aws_setup_guide.md` Part B
- **Output:** Confirm `.env` + boto3 work on your machine
- **Acceptance:**
  - [X] Verification script runs and all checks pass

---

## Day 2 — PDF Extraction + Text Cleaning

### Task A2.1: PDF Text Extraction
- **Reference:** `docs/03_data_engineering.md` — Notebook 02 (PyMuPDF, text-only, skip tables)
- **Input:** Downloaded CMS PDFs from S3 `raw/cms_manuals/`
- **Output:** `data/extract_text.py`
- **Acceptance:**
  - [X] Uses PyMuPDF for text extraction
  - [X] Extracts text from at least 1 full chapter
  - [X] Output is structured as JSON with per-page text + metadata (manual_id, chapter, page)
  - [X] Tables are skipped (text-only extraction for MVP — per `docs/01_architecture_overview.md` Section 5)
  - [X] Results saved to S3 `processed/extracted_text/`

### Task A2.2: Text Cleaning
- **Reference:** `docs/03_data_engineering.md` — Notebook 03
- **Input:** Extracted raw text from A2.1
- **Output:** `data/clean_text.py`
- **Acceptance:**
  - [X] Headers/footers removed (manual title, page numbers)
  - [X] Hyphenated line breaks fixed ("cover-\nage" → "coverage")
  - [X] OCR junk removed (if any)
  - [X] Multiple blank lines collapsed
  - [X] Results saved to S3 `processed/cleaned_text/`

### Task A2.3: Deliver 10+ Real Chunks to Member B
- **Acceptance:**
  - [X] 10+ actual extracted chunks (not mock) shared via S3 `chunks/`
  - [X] Schema matches `mock_chunks.json` format exactly
  - [ ] Member B confirms they load and embed without errors

---

## Day 3 — Hierarchical Chunking (Full Corpus)

### Task A3.1: Hierarchical Chunking Script
- **Reference:** `docs/01_architecture_overview.md` Section 5 + `docs/03_data_engineering.md` — Notebook 04
- **Input:** Cleaned text from A2.2
- **Output:** `data/chunk_hierarchical.py`
- **Acceptance:**
  - [X] Detects chapter and section boundaries via regex or font-based heuristics
  - [X] ~500 tokens per chunk, 50-token overlap (per `docs/06_assumptions_tradeoffs.md`)
  - [X] Generates deterministic `chunk_id` per schema: `{manual_id}_ch{N}_s{X}_p{P}_{seq}`
  - [X] Generates `parent_chunk_id` for parent document retrieval
  - [X] All metadata fields populated: `manual_id`, `manual_title`, `chapter_num`, `chapter_title`, `section_title`, `page_num`, `source_url`, `token_count`
  - [X] Complete `chunks.json` uploaded to S3 `chunks/`

### Task A3.2: Validate Chunk Quality
- **Acceptance:**
  - [X] Spot-check 10 random chunks — text is coherent, not cut mid-sentence
  - [X] No duplicate `chunk_id` values
  - [X] Token counts are in 400–600 range
  - [X] Total chunk count is reasonable (~500–2000 from 3 manuals per `docs/06_assumptions_tradeoffs.md` Assumption #5)
  - [X] Coverage across all 3 manuals verified

---

## Day 4 — Evaluation Dataset (Support Role)

### Task A4.1: Write Q&A Pairs — Eligibility + Exclusions
- **Reference:** `docs/05_testing_evaluation.md` Section 5 — Category Distribution
- **Input:** Final `chunks.json` — read chunks to formulate questions
- **Output:** Contribute 25 Q&A pairs to `evaluation/eval_dataset.json`
- **Category split (per doc 05):**
  - 15 Eligibility questions ("Who qualifies for...", "Requirements for...")
  - 10 Exclusion questions ("What is NOT covered?", "Exceptions to...")
- **Format per pair (per doc 05):**
  ```json
  {
    "id": "eval_001",
    "question": "What are the eligibility requirements for Medicare home health services?",
    "ground_truth_answer": "To be eligible, a patient must...",
    "source_manual": "100-02",
    "source_chapter": 7,
    "source_section": "30",
    "difficulty": "medium",
    "category": "eligibility",
    "requires_fhir": false
  }
  ```
- **Acceptance:**
  - [X] 25 pairs written in the correct format
  - [X] Each references correct source manual, chapter, section
  - [X] Member B validates 5 random pairs against the chunks

---

## Day 5 — Finalize + Databricks Notebooks

### Task A5.1: Finalize Eval Dataset
- **Acceptance:**
  - [X] Combined eval dataset has 70 Q&A pairs (with Member B's contributions)
  - [X] Category distribution matches `docs/05_testing_evaluation.md` Section 5
  - [X] Uploaded to S3 `evaluation/eval_dataset.json`

### Task A5.2: Create Databricks Notebooks
- **Reference:** `docs/03_data_engineering.md` Section 1 + `docs/08_azure_databricks_setup.md` Section 3
- **Input:** Local scripts from `data/`
- **Output:** `notebooks/01_download_cms.py`, `02_extract_text.py`, `03_clean_text.py`, `04_chunk.py`
- **Acceptance:**
  - [ ] Each notebook has S3 connection block at top (Spark `fs.s3a.*` config)
  - [ ] Sequential execution produces the same `chunks.json` as local scripts
  - [ ] Tested on `Shared-Capstone-Cluster`
  - [ ] Can be chained as a Databricks Workflow (DAG) per doc 03