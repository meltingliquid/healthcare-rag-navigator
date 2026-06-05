# Databricks notebook source
# MAGIC %md
# MAGIC # AI/ML Pipeline - Day 2
# MAGIC **Task:** Embed Chunks using OpenAI and Store in S3 (Backup Locally)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install Dependencies

# COMMAND ----------

# %pip install boto3 python-dotenv openai numpy --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Imports and Environment Setup

# COMMAND ----------

import os
import sys
import json
import time
import pathlib
import boto3
import numpy as np
from botocore.exceptions import ClientError, BotoCoreError
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# Resolve root to import config
NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name in ['notebooks', 'retrieval'] else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import variables from settings
from config.settings import (
    S3_BUCKET,
    S3_CHUNKS_KEY,
    S3_EMBEDDINGS_PREFIX,
    S3_CHUNK_ID_MAP_KEY,
)

# Constants
MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 100
BATCH_DELAY = 1

# Cost estimate constants
COST_PER_1M_TOKENS = 0.02
AVG_TOKENS_PER_CHUNK = 500

# Local Paths
EMBEDDINGS_PATH = PROJECT_ROOT / "retrieval" / "embeddings.npy"
CHUNK_MAP_PATH = PROJECT_ROOT / "retrieval" / "chunk_id_map.json"
PROGRESS_PATH = PROJECT_ROOT / "retrieval" / ".embed_progress.json"

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
# MAGIC ### 4. Helper Functions

# COMMAND ----------

def download_chunks() -> list[dict]:
    """Download chunks.json from S3 into memory."""
    try:
        print(f"☁️   Downloading s3://{S3_BUCKET}/{S3_CHUNKS_KEY} ...")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_CHUNKS_KEY)
        chunks = json.loads(obj["Body"].read().decode("utf-8"))
        print(f"✅  {len(chunks)} chunks downloaded")
        return chunks
    except Exception as e:
        print(f"❌  S3 download error: {e}")
        sys.exit(1)

def upload_file(local_path: str, s3_key: str) -> None:
    """Upload a local file to S3."""
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"☁️   Uploaded → s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"⚠️   S3 upload failed ({s3_key}): {e}")

def save_progress(batch_idx: int, total: int, count: int) -> None:
    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "last_batch": batch_idx, 
            "total_batches": total,
            "embedded": count, 
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)

def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return None

def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    res = client.embeddings.create(model=MODEL, input=texts)
    return [x.embedding for x in res.data]

print("✅ Helper functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Embedding Pipeline & Resume Handling

# COMMAND ----------

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY missing from .env")
    sys.exit(1)

chunks = download_chunks()
n = len(chunks)

# Cost estimate
est_tokens = n * AVG_TOKENS_PER_CHUNK
est_cost = (est_tokens / 1_000_000) * COST_PER_1M_TOKENS
print(f"\n💰  Est. cost: ${est_cost:.4f} (~{est_tokens:,} tokens @ ${COST_PER_1M_TOKENS}/1M tokens)")

total_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
print(f"🗂️   {total_batches} batches × {BATCH_SIZE} chunks, {BATCH_DELAY}s delay")

start_batch = 0
all_embeddings = []

prog = load_progress()
if prog and EMBEDDINGS_PATH.exists():
    start_batch = prog["last_batch"] + 1
    all_embeddings = np.load(EMBEDDINGS_PATH).tolist()
    print(f"🔄  Resuming from batch {start_batch + 1}/{total_batches} ({len(all_embeddings)} embeddings done)")
else:
    print("ℹ️   No progress file — starting from scratch")

client = OpenAI(api_key=api_key)
failed = []

print("\n🚀 Starting Embedding Process...")
for idx in range(start_batch, total_batches):
    batch = chunks[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
    texts = [c.get("chunk_text", " ") for c in batch]

    try:
        embs = embed_batch(client, texts)
        all_embeddings.extend(embs)
        print(f"  ✅  Batch {idx+1:>4}/{total_batches}: {len(texts)} chunks")

        # Save after every batch locally for backup
        np.save(EMBEDDINGS_PATH, np.array(all_embeddings, dtype=np.float32))
        save_progress(idx, total_batches, len(all_embeddings))

    except Exception as e:
        print(f"  ❌  Batch {idx+1} failed: {e}")
        failed.append(idx)
        time.sleep(30) # slight backoff on error
    
    if idx < total_batches - 1:
        time.sleep(BATCH_DELAY)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Save & Upload Results

# COMMAND ----------

# Final save
arr = np.array(all_embeddings, dtype=np.float32)
np.save(EMBEDDINGS_PATH, arr)
print(f"\n💾  embeddings.npy saved (shape: {arr.shape})")

# Save map
chunk_map = {str(i): c["chunk_id"] for i, c in enumerate(chunks)}
with open(CHUNK_MAP_PATH, "w") as f:
    json.dump(chunk_map, f, indent=2)
print(f"💾  chunk_id_map.json saved ({len(chunk_map)} entries)")

# Upload to S3
print("\n⬆️  Uploading results to S3...")
upload_file(str(EMBEDDINGS_PATH), f"{S3_EMBEDDINGS_PREFIX}embeddings.npy")
upload_file(str(CHUNK_MAP_PATH), S3_CHUNK_ID_MAP_KEY)

# Clear progress backup
if not failed and PROGRESS_PATH.exists():
    os.remove(PROGRESS_PATH)

print("\n✅ Embedding Pipeline Complete")
if failed:
    print(f"⚠️ {len(failed)} batches failed. Check errors and resume.")

# COMMAND ----------