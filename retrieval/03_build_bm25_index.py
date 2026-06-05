# Databricks notebook source
# MAGIC %md
# MAGIC # AI/ML Pipeline - Day 3
# MAGIC **Task:** Build BM25 Index for exact term matching and Store in S3

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install Dependencies

# COMMAND ----------

# %pip install boto3 python-dotenv rank_bm25 --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Imports and Environment Setup

# COMMAND ----------

import os
import sys
import json
import string
import pickle
import pathlib
import boto3
from botocore.exceptions import ClientError
from rank_bm25 import BM25Okapi

# Resolve root to import config
NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name in ['notebooks', 'retrieval'] else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import variables from settings
from config.settings import (
    S3_BUCKET,
    S3_CHUNKS_KEY,
    S3_BM25_INDEX_KEY,
)

# Constants & Paths
BM25_INDEX_PATH = PROJECT_ROOT / "retrieval" / "bm25_index.pkl"

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
# MAGIC ### 4. Helper logic & Tokenization

# COMMAND ----------

def tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation + split. Must match search-time tokenisation."""
    if not text:
        return []
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def upload_to_s3(local_path: pathlib.Path, bucket: str, s3_key: str):
    try:
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"☁️   Uploaded → s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"❌  S3 upload failed for {s3_key}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Build BM25 Index

# COMMAND ----------

print("🚀 Starting BM25 Index Build...")

print(f"☁️  Downloading s3://{S3_BUCKET}/{S3_CHUNKS_KEY} ...")
try:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_CHUNKS_KEY)
    chunks = json.loads(obj["Body"].read().decode("utf-8"))
    print(f"✅  {len(chunks)} chunks loaded")
except Exception as e:
    print(f"❌ Failed to download chunks: {e}")
    sys.exit(1)

# Tokenize corpus
chunk_ids = [c["chunk_id"] for c in chunks]
tokenized_corpus = [tokenize(c.get("chunk_text", "")) for c in chunks]
print(f"⚙️  Tokenized {len(tokenized_corpus)} chunks")

# Build BM25Okapi index
bm25 = BM25Okapi(tokenized_corpus)
print("✅ BM25Okapi index built")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Save & Upload Index

# COMMAND ----------

# Save locally
with open(BM25_INDEX_PATH, "wb") as f:
    pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
print(f"💾 Index saved → {BM25_INDEX_PATH.name}")

# Upload to S3
print("\n⬆️  Uploading files to S3...")
upload_to_s3(BM25_INDEX_PATH, S3_BUCKET, S3_BM25_INDEX_KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Test Retrieval (Top-20)

# COMMAND ----------

print("\n🔍  Top-20 retrieval test:")
for query in ["§40.1 HCPCS G0179", "home health coverage"]:
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    # Sort scores descending
    top_n = min(20, len(scores))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    
    print(f"\n  Query: \"{query}\"")
    print(f"  {'Rank':<5} {'chunk_id':<40} {'BM25 Score':>10}")
    print("  " + "─" * 60)
    
    for rank, idx in enumerate(top_indices, start=1):
        marker = " ⭐" if scores[idx] > 0 else ""
        print(f"  {rank:<5} {chunk_ids[idx]:<40} {scores[idx]:>10.4f}{marker}")

print("\n✅ Test Complete")

# COMMAND ----------
