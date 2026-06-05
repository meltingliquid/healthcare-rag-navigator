# Databricks notebook source
# MAGIC %md
# MAGIC # AI/ML Pipeline - Day 3
# MAGIC **Task:** Build FAISS Index from Embeddings and Store in S3

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install Dependencies

# COMMAND ----------

# %pip install boto3 python-dotenv faiss-cpu numpy --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Imports and Environment Setup

# COMMAND ----------

import os
import sys
import json
import pathlib
import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError

# Resolve root to import config
NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name in ['notebooks', 'retrieval'] else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import variables from settings
from config.settings import (
    S3_BUCKET,
    S3_FAISS_INDEX_KEY,
    S3_CHUNK_ID_MAP_KEY,
)

# Constants & Paths
DIMENSIONS = 1536
EMBEDDINGS_PATH = PROJECT_ROOT / "retrieval" / "embeddings.npy"
CHUNK_MAP_PATH = PROJECT_ROOT / "retrieval" / "chunk_id_map.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "retrieval" / "faiss_index.bin"

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
# MAGIC ### 4. Helper function to upload

# COMMAND ----------

def upload_to_s3(local_path: pathlib.Path, bucket: str, s3_key: str):
    try:
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"☁️   Uploaded → s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"❌  S3 upload failed for {s3_key}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Build FAISS Index

# COMMAND ----------

print("🚀 Starting FAISS Index Build...")

if not EMBEDDINGS_PATH.exists() or not CHUNK_MAP_PATH.exists():
    print("❌ embeddings.npy or chunk_id_map.json not found — please run embed_chunks first.")
    sys.exit(1)

# Load embeddings
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

# Load chunk map
with open(CHUNK_MAP_PATH, "r") as f:
    chunk_id_map = json.load(f)

print(f"📄 Loaded embeddings {embeddings.shape} | {len(chunk_id_map)} chunk IDs")

# Normalize for Cosine Similarity
faiss.normalize_L2(embeddings)
print("✅ Vectors normalized (L2)")

# Build faiss.IndexFlatIP
index = faiss.IndexFlatIP(DIMENSIONS)
index.add(embeddings)
print(f"✅ faiss.IndexFlatIP built ({index.ntotal} vectors)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Save & Upload Index

# COMMAND ----------

# Save locally
faiss.write_index(index, str(FAISS_INDEX_PATH))
print(f"💾 Index saved → {FAISS_INDEX_PATH.name}")

# Upload Faiss index and chunk map to S3
print("\n⬆️  Uploading files to S3...")
upload_to_s3(FAISS_INDEX_PATH, S3_BUCKET, S3_FAISS_INDEX_KEY)
upload_to_s3(CHUNK_MAP_PATH, S3_BUCKET, S3_CHUNK_ID_MAP_KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Test Retrieval (Top-20)

# COMMAND ----------

print("\n🔍 Running Top-20 Retrieval Test (Random Query Vector)...")

rng = np.random.default_rng(seed=99)
query = rng.standard_normal((1, DIMENSIONS)).astype("float32")
faiss.normalize_L2(query)

k = min(20, index.ntotal)
distances, indices = index.search(query, k)

print(f"\n  {'Rank':<5} {'chunk_id':<40} {'Score':>8}")
print("  " + "─" * 55)
for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
    cid = chunk_id_map.get(str(idx), "UNKNOWN")
    print(f"  {rank:<5} {cid:<40} {score:>8.4f}")

print("\n✅ Test Complete")

# COMMAND ----------
