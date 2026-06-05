# AWS Setup Guide — Healthcare RAG Capstone Project

## Current Status (as of Day 1)

| Item | Status | Details |
|------|--------|---------|
| IAM User | ✅ Done | `project2` — member of `freeloaders` group |
| IAM Permissions | ✅ Done | `AmazonS3FullAccess` (sufficient for demo project) |
| S3 Bucket | ✅ Done | Created with versioning + SSE-S3 encryption |
| Folder Structure | ✅ Done | `raw/`, `processed/`, `chunks/`, `indexes/`, `fhir_cache/`, `evaluation/`, `config/` |
| CORS Config | ✅ Done | Configured for notebook/web access |
| Credentials Shared | ✅ Done | Access Key ID + Secret shared with team |

**Nothing left for the team lead to do on AWS.** The rest of this guide is for teammates.

---

## What All Teammates Need to Do

There are **two separate things** to set up:

| Purpose | Method | Required? |
|---------|--------|-----------|
| Browse S3 files, upload/download manually | AWS Console (web GUI) | All members |
| Run pipeline code that reads/writes S3 | `.env` + `boto3` in Python | All members (the code needs it) |

---

## Part A — AWS Console Access (GUI)

This is how you browse the bucket, check uploaded files, and do manual uploads/downloads.

### A.1 Sign In

1. Go to [https://console.aws.amazon.com/](https://console.aws.amazon.com/)
2. Sign in with the shared IAM credentials:
   - **Account ID / alias:** *(Team lead will share)*
   - **Username:** `project2`
   - **Password:** *(Team lead will share)*

> [!NOTE]
> The IAM user `project2` has Console access enabled. If you get a login error, confirm with the team lead that Console access is activated on the user.

### A.2 Navigate to the Project Bucket

1. In the Console, search for **S3** in the top search bar.
2. Click on the project bucket name.
3. You will see all the folders created for the project.

### A.3 Folder Ownership (Who Uploads Where)

Use the Console to manually upload or check files in your assigned prefix:

| Member | Assigned Prefix(es) | What goes there |
|--------|-------------------|-----------------|
| **Member A** (Data Engineer) | `raw/cms_manuals/` | Raw CMS PDF files |
| | `processed/extracted_text/` | Extracted plain text |
| | `processed/cleaned_text/` | Cleaned text |
| | `chunks/` | `chunks.json`, `mock_chunks.json` |
| **Member B** (AI/ML) | `indexes/faiss/` | `faiss_index.bin` |
| | `indexes/bm25/` | `bm25_index.pkl` |
| | `config/` | `chunk_id_map.json` |
| | `embeddings/` | Embedding cache (optional) |
| **Member C** (Backend) | `fhir_cache/` | Cached FHIR JSON responses |
| **All Members** | `evaluation/` | Eval dataset, RAGAS results |

### A.4 How to Upload a File via Console

1. Open your assigned folder (click into it).
2. Click **Upload → Add files** → select your file → **Upload**.
3. After upload, the file appears in the folder listing.

### A.5 How to Download a File via Console

1. Click on the file name.
2. Click **Download** (top right of the file detail page).

> [!TIP]
> To share a file link with a teammate without downloading, use **Copy S3 URI** — this gives you the path like `s3://healthcare-rag-capstone/chunks/mock_chunks.json` which can be used directly in Python code.

---

## Part B — Python / Code Access (boto3)

Even though browsing is done via the Console, **the actual pipeline code** (Member A's upload scripts, Member B's index builder, Member C's FHIR cache) reads and writes S3 programmatically. This part is still required for coding.

### B.1 Create a `.env` File in the Project Root

You received the credentials from the team lead. Create a `.env` file in the **root of the cloned repo**:

```env
# .env — DO NOT COMMIT THIS FILE
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=healthcare-rag-capstone

# Add your OpenAI key when available
OPENAI_API_KEY=sk-...
```

> [!IMPORTANT]
> `.env` is already listed in `.gitignore`. Never remove it. Never paste credentials into any Python file directly.

### B.2 Install Required Libraries

```bash
pip install boto3 python-dotenv
```

### B.3 Verify Your Connection

Run this once after setup to confirm your credentials work:

```python
# config/verify_aws.py
import boto3, os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)

BUCKET = os.getenv("S3_BUCKET_NAME", "healthcare-rag-capstone")

try:
    response = s3.list_objects_v2(Bucket=BUCKET, Delimiter="/")
    prefixes = [p["Prefix"] for p in response.get("CommonPrefixes", [])]
    print(f"[OK] Connected to s3://{BUCKET}")
    print(f"[OK] Folders found: {prefixes}")
except Exception as e:
    print(f"[ERROR] {e}")
    print("Check: .env credentials, bucket name, IAM permissions.")
```

Run it:
```bash
python config/verify_aws.py
```

Expected output:
```
[OK] Connected to s3://healthcare-rag-capstone
[OK] Folders found: ['chunks/', 'config/', 'evaluation/', 'fhir_cache/', 'indexes/', 'processed/', 'raw/']
```

---

## S3 Path Constants — `config/settings.py`

**All three members import from here** so paths are never mistyped across scripts:

```python
# config/settings.py
import os
from dotenv import load_dotenv
load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "healthcare-rag-capstone")

# Member A
S3_RAW_PREFIX       = "raw/cms_manuals/"
S3_EXTRACTED_PREFIX = "processed/extracted_text/"
S3_CLEANED_PREFIX   = "processed/cleaned_text/"
S3_CHUNKS_KEY       = "chunks/chunks.json"
S3_MOCK_CHUNKS_KEY  = "chunks/mock_chunks.json"

# Member B
S3_FAISS_INDEX_KEY  = "indexes/faiss/faiss_index.bin"
S3_BM25_INDEX_KEY   = "indexes/bm25/bm25_index.pkl"
S3_CHUNK_ID_MAP_KEY = "config/chunk_id_map.json"
S3_EMBEDDINGS_PREFIX = "embeddings/"

# Member C
S3_FHIR_CACHE_PREFIX = "fhir_cache/"

# Shared
S3_EVAL_DATASET_KEY    = "evaluation/eval_dataset.json"
S3_EVAL_RESULTS_PREFIX = "evaluation/results/"
```

Usage in any script:
```python
from config.settings import S3_BUCKET, S3_CHUNKS_KEY
```

---

## Security Rules — All Members

| Rule | Why |
|------|-----|
| Keep `.env` out of Git (it's in `.gitignore`) | Secrets must never be in version control |
| Never paste AWS keys into Python files or Jupyter notebooks | Notebooks get committed; secrets leak |
| Never share keys in GitHub issues, PRs, or Discord channels | Public repos are public |
| Rotate the access key at the end of the project | Good hygiene after a shared key is used |

---

## Quick Reference

| Item | Value |
|------|-------|
| Console login | [console.aws.amazon.com](https://console.aws.amazon.com/) |
| IAM username | `project2` |
| IAM group | `freeloaders` |
| S3 permissions | `AmazonS3FullAccess` |
| Bucket name | `healthcare-rag-capstone` |
| Region | `us-east-1` |
| Verify script | `python config/verify_aws.py` |
| Path constants | `config/settings.py` |

---

*Next: Azure account setup → Databricks workspace → GitHub collaboration strategy → OpenSpec task tracking*
