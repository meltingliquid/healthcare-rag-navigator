# Databricks notebook source
# MAGIC %md
# MAGIC # AI/ML Pipeline - Day 3
# MAGIC **Task:** Hybrid Retriever (FAISS + BM25 + RRF Fusion)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install Dependencies

# COMMAND ----------

# %pip install boto3 python-dotenv openai faiss-cpu numpy rank_bm25 --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Imports and Environment Setup

# COMMAND ----------

import io
import json
import os
import sys
import pickle
import string
import tempfile
import pathlib
from collections import defaultdict

import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError
from openai import OpenAI

# Resolve root to import config
NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path().resolve()
_candidate = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name in ['notebooks', 'retrieval'] else NOTEBOOK_DIR

# Databricks Apps mounts source at /app/python/source_code — detect this explicitly
_DATABRICKS_APPS_ROOT = pathlib.Path("/app/python/source_code")
if _DATABRICKS_APPS_ROOT.exists():
    PROJECT_ROOT = _DATABRICKS_APPS_ROOT
else:
    PROJECT_ROOT = _candidate

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import variables from settings
from config.settings import (
    S3_BUCKET,
    S3_CHUNKS_KEY,
    S3_FAISS_INDEX_KEY,
    S3_BM25_INDEX_KEY,
    S3_CHUNK_ID_MAP_KEY,
)

EMBED_MODEL = "text-embedding-3-small"
DIMENSIONS  = 1536

print("✅ Imports and Environment Setup OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. S3 Client Setup (Hybrid)

# COMMAND ----------

def get_s3_client():
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        aws_access_key = dbutils.secrets.get(scope="capstone_scope", key="aws_access_key")
        aws_secret_key = dbutils.secrets.get(scope="capstone_scope", key="aws_secret_key")
    else:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / '.env')
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name="ap-south-1",
    )

s3 = get_s3_client()
print('✅ S3 Client Initialized')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Load Indexes and Chunks from S3

# COMMAND ----------

# Module-level state — loaded once, reused on every retrieve() call
_faiss_index  = None
_chunk_id_map = None
_bm25         = None
_bm25_ids     = None
_chunks_dict  = {}

def _load_all():
    """Lazy-load all indexes and chunk data from S3 into memory."""
    global _faiss_index, _chunk_id_map, _bm25, _bm25_ids, _chunks_dict
    if _faiss_index is not None:
        return  # Already loaded

    print("🔧 Loading indexes and chunks from S3...")

    # FAISS index: faiss.read_index() requires a file path, so use a temp file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        s3.download_fileobj(S3_BUCKET, S3_FAISS_INDEX_KEY, tmp)
        tmp_path = tmp.name
    _faiss_index = faiss.read_index(tmp_path)
    os.unlink(tmp_path)

    # chunk_id_map: int FAISS index → chunk_id string
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_CHUNK_ID_MAP_KEY)
    _chunk_id_map = json.loads(obj["Body"].read())

    # BM25 index
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_BM25_INDEX_KEY)
    data      = pickle.loads(obj["Body"].read())
    _bm25     = data["bm25"]
    _bm25_ids = data["chunk_ids"]

    # chunks.json for metadata lookup
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_CHUNKS_KEY)
    chunks = json.loads(obj["Body"].read().decode("utf-8"))
    _chunks_dict = {c["chunk_id"]: c for c in chunks}

    print(f"✅ FAISS: {_faiss_index.ntotal} vecs | BM25: {len(_bm25_ids)} docs | Chunks: {len(_chunks_dict)}")

# Trigger load
_load_all()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Tokenizer, FAISS Search, BM25 Search

# COMMAND ----------

def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation + whitespace split. Must match build_bm25_index.py."""
    if not text:
        return []
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def _faiss_search(query: str, top_n: int = 20) -> list[tuple[str, float]]:
    """Embed query with OpenAI and search FAISS index. Returns [(chunk_id, score)]."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(q_vec)

    distances, indices = _faiss_index.search(q_vec, min(top_n, _faiss_index.ntotal))
    return [
        (_chunk_id_map[str(idx)], float(score))
        for score, idx in zip(distances[0], indices[0])
        if str(idx) in _chunk_id_map
    ]

def _bm25_search(query: str, top_n: int = 20) -> list[tuple[str, float]]:
    """Search BM25 index for exact term matches. Returns [(chunk_id, score)]."""
    scores = _bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [(_bm25_ids[i], float(scores[i])) for i in top_idx]

print("✅ Search helper functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Hybrid Retrieve Function (RRF Fusion)

# COMMAND ----------

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid retrieval: FAISS top-20 + BM25 top-20 → RRF-60 → top-15 → top_k.

    Returns list[dict] with keys:
      chunk_id, chunk_text, score, manual_id, chapter_title, section_title, page_num
    """
    _load_all()

    faiss_results = _faiss_search(query)   # top-20 semantic candidates
    bm25_results  = _bm25_search(query)    # top-20 keyword candidates

    # Reciprocal Rank Fusion (RRF-60):
    # rrf_score(d) = Σ 1 / (k + rank(d))  where k=60 is the standard constant
    rrf: dict[str, float] = defaultdict(float)
    for rank, (cid, _) in enumerate(faiss_results, start=1):
        rrf[cid] += 1.0 / (60 + rank)
    for rank, (cid, _) in enumerate(bm25_results, start=1):
        rrf[cid] += 1.0 / (60 + rank)

    # Merge + deduplicate → top-15, then slice to top_k
    top15 = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:15]

    return [
        {
            "chunk_id":      cid,
            "chunk_text":    _chunks_dict.get(cid, {}).get("chunk_text", ""),
            "score":         round(rrf[cid], 6),
            "manual_id":     _chunks_dict.get(cid, {}).get("manual_id",     ""),
            "chapter_title": _chunks_dict.get(cid, {}).get("chapter_title", ""),
            "section_title": _chunks_dict.get(cid, {}).get("section_title", ""),
            "page_num":      _chunks_dict.get(cid, {}).get("page_num",       0),
            "source_url":    _chunks_dict.get(cid, {}).get("source_url",     ""),
        }
        for cid in top15[:top_k]
    ]

print("✅ retrieve() function defined and ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Validation Tests (3 Query Types)

# COMMAND ----------

# test_queries = [
#     ("semantic", "home health coverage"),
#     ("keyword",  "§40.1 HCPCS G0179"),
#     ("mixed",    "home health §40.1 coverage requirements"),
# ]

# for qtype, q in test_queries:
#     print("\n" + "=" * 65)
#     print(f"  [{qtype.upper()}]  \"{q}\"")
#     print("=" * 65)
#     results = retrieve(q, top_k=5)
#     for i, r in enumerate(results, start=1):
#         print(f"  {i}. score={r['score']:.5f}  chunk_id={r['chunk_id']}")
#         print(f"     {r['manual_id']} › {r['chapter_title']} › {r['section_title']}  (p.{r['page_num']})")
#         snippet = r["chunk_text"][:100].replace("\n", " ")
#         print(f"     {snippet}…")

# print("\n✅ All 3 query type tests complete")

# COMMAND ----------
