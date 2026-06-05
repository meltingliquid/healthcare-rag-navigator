# Databricks notebook source
# MAGIC %md
# MAGIC # Data Engineering Pipeline - Day 3
# MAGIC **Task:** Hierarchical Chunking of Cleaned Text and S3 Upload

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install Dependencies

# COMMAND ----------

# %pip install boto3 python-dotenv tqdm tiktoken --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Imports and Environment Setup

# COMMAND ----------

import os
import sys
import json
import re
import random
import pathlib
import boto3
from botocore.exceptions import ClientError
from typing import Dict, List, Tuple

# Attempt to load tiktoken
try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(ENCODER.encode(text))
except ImportError:
    print("⚠️  WARNING: tiktoken not installed. Using len(words)*1.3 fallback.")
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

# Resolve root to import config
NOTEBOOK_DIR = pathlib.Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import variables from settings
from config.settings import S3_BUCKET, S3_CHUNKS_KEY

# Local directories
CLEANED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned_text"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

print('✅ Imports and Environment Setup OK')
print(f'   PROJECT_ROOT: {PROJECT_ROOT}')
print(f'   S3_BUCKET: {S3_BUCKET}')
print(f'   S3_CHUNKS_KEY: {S3_CHUNKS_KEY}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. S3 Client Setup (Hybrid)

# COMMAND ----------

def get_s3_client():
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        # Running in Databricks
        aws_access_key = dbutils.secrets.get(scope="capstone_scope", key="aws_access_key")
        aws_secret_key = dbutils.secrets.get(scope="capstone_scope", key="aws_secret_key")
    else:
        # Running locally
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
# MAGIC ### 4. Chunking Logic & Helpers

# COMMAND ----------

# Regex for section headers (e.g., "10.1 - Requirements")
SECTION_REGEX = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*-\s*(.+)$")

def split_by_sentences(text: str) -> List[str]:
    """Splits text into sentences using punctuation boundaries."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text_with_overlap(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """Chunks text preserving sentences with target token size and overlap."""
    sentences = split_by_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            overlap_chunk = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                s_toks = count_tokens(s)
                if overlap_tokens + s_toks > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_tokens += s_toks
                
            current_chunk = overlap_chunk
            current_tokens = overlap_tokens
            
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Process Files and Generate Hierarchical Chunks

# COMMAND ----------

def process_file(filepath: pathlib.Path) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        doc = json.load(f)
        
    meta = doc.get("metadata", {})
    manual_id = meta.get("manual_id", "Unknown")
    manual_title = meta.get("manual_title", "Unknown")
    chapter_num = str(meta.get("chapter_num", "0"))
    chapter_title = meta.get("chapter_title", "Unknown")
    source_url = meta.get("source_url", "")
    
    current_section_num = "0"
    current_section_title = "General"
    
    chunks_out = []
    seq_tracker = {}
    
    for page in doc.get("pages", []):
        page_num = page.get("page_num", 1)
        cleaned_text = page.get("cleaned_text", "")
        
        lines = cleaned_text.split('\n')
        section_buffers = []
        current_buffer = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            match = SECTION_REGEX.match(line_stripped)
            if match:
                if current_buffer:
                    section_buffers.append((current_section_num, current_section_title, " ".join(current_buffer)))
                current_section_num = match.group(1)
                current_section_title = match.group(2)[:100]
                current_buffer = [line_stripped]
            else:
                current_buffer.append(line_stripped)
                
        if current_buffer:
            section_buffers.append((current_section_num, current_section_title, " ".join(current_buffer)))
            
        # Process section buffers into chunks
        for sec_num, sec_title, sec_text in section_buffers:
            sub_chunks = chunk_text_with_overlap(sec_text, max_tokens=500, overlap=50)
            
            ch_clean = str(chapter_num).replace(" ", "_")
            parent_chunk_id = f"{manual_id}_ch{ch_clean}_s{sec_num}"
            seq_key = f"{parent_chunk_id}_p{page_num}"
            
            for chunk_str in sub_chunks:
                tok_count = count_tokens(chunk_str)
                if tok_count < 15:
                    continue
                
                seq_tracker[seq_key] = seq_tracker.get(seq_key, 0) + 1
                seq = seq_tracker[seq_key]
                
                chunk_id = f"{seq_key}_{seq:03d}"
                
                chunks_out.append({
                    "chunk_id": chunk_id,
                    "parent_chunk_id": parent_chunk_id,
                    "manual_id": manual_id,
                    "manual_title": manual_title,
                    "chapter_num": chapter_num,
                    "chapter_title": chapter_title,
                    "section_title": sec_title,
                    "page_num": page_num,
                    "source_url": source_url,
                    "chunk_text": chunk_str,
                    "token_count": tok_count
                })
                
    return chunks_out

print("🚀 Starting Hierarchical Chunking...")

if not CLEANED_DIR.exists():
    print(f"❌ Missing directory: {CLEANED_DIR}")
    files = []
else:
    files = list(CLEANED_DIR.glob("*.json"))

print(f"📋 Found {len(files)} JSON files to process.")

all_chunks = []
for f in files:
    print(f"   Generating chunks for {f.name}...")
    all_chunks.extend(process_file(f))
    
# Sort chunks for deterministic output
all_chunks.sort(key=lambda x: x["chunk_id"])

local_chunks_path = CHUNKS_DIR / "chunks.json"
with open(local_chunks_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
print(f"\n✅ Saved {len(all_chunks)} chunks → {local_chunks_path.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Validate Chunk Quality

# COMMAND ----------

print("\n📋 TASK A3.2: VALIDATE CHUNK QUALITY")

if not all_chunks:
    print("❌ No chunks generated!")
else:
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    unique_ids = set(chunk_ids)
    
    if len(chunk_ids) == len(unique_ids):
        print("   ✅ No duplicate chunk_ids found.")
    else:
        print("   ❌ Duplicate chunk_ids found!")
        
    avg_tokens = sum(c["token_count"] for c in all_chunks) / len(all_chunks)
    print(f"   ✅ Average token count: {avg_tokens:.1f}")
    
    manuals = set(c["manual_id"] for c in all_chunks)
    print(f"   ✅ Coverage across {len(manuals)} manuals: {manuals}")
    
    print("\n🔍 Random Spot Checks (3 chunks):")
    samples = random.sample(all_chunks, min(3, len(all_chunks)))
    for s in samples:
        print(f"   - ID: {s['chunk_id']} | Section: {s['section_title']} | Tokens: {s['token_count']}")
        print(f"     Preview: {s['chunk_text'][:80]}...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Upload to S3

# COMMAND ----------

try:
    print(f"\n⬆️  Uploading chunks.json to S3...")
    s3.upload_file(str(local_chunks_path), S3_BUCKET, S3_CHUNKS_KEY)
    print(f"✅ Uploaded to s3://{S3_BUCKET}/{S3_CHUNKS_KEY}")
except ClientError as e:
    print(f"❌ S3 Upload Failed: {e}")

# COMMAND ----------