# Databricks notebook source
# MAGIC %md
# MAGIC # Data Engineering Pipeline - Day 1
# MAGIC **Task:** Download CMS source documents and upload to S3

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install Dependencies

# COMMAND ----------

# %pip install boto3 python-dotenv requests tqdm --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Imports and Environment Setup

# COMMAND ----------

import os
import sys
import json
import time
import requests
import pathlib
import boto3
from botocore.exceptions import ClientError

# Resolve root to import config
NOTEBOOK_DIR = pathlib.Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import variables from settings
from config.settings import S3_BUCKET, S3_RAW_PREFIX

# Local output directories
LOCAL_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

print('✅ Imports OK')
print(f'   PROJECT_ROOT: {PROJECT_ROOT}')
print(f'   S3_BUCKET: {S3_BUCKET}')
print(f'   S3_RAW_PREFIX: {S3_RAW_PREFIX}')

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
# MAGIC ### 4. Load Manifest

# COMMAND ----------

MANIFEST_PATH = PROJECT_ROOT / 'sources' / 'cms_download_manifest.json'

with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

CMS_CHAPTERS = []
for manual in manifest.get('manuals', []):
    manual_id = manual['manual_id']
    for chapter in manual.get('chapters', []):
        CMS_CHAPTERS.append({
            'manual_id': manual_id,
            'filename': chapter['local_filename'],
            'url': chapter['url']
        })

print(f'📋 Total documents to process: {len(CMS_CHAPTERS)}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Download Files Locally

# COMMAND ----------

DOWNLOAD_TIMEOUT_SEC = 120
HEADERS = {'User-Agent': 'Mozilla/5.0 (HealthcareRAG-Research/1.0)'}

def download_pdf(url: str, dest_path: pathlib.Path) -> bool:
    """Download a PDF to dest_path. Returns True on success."""
    if dest_path.exists():
        print(f'   ⏭️  Already exists locally: {dest_path.name}')
        return True
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT_SEC, stream=True)
        resp.raise_for_status()
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                
        print(f'   ✅ Downloaded: {dest_path.name}')
        return True
    except Exception as e:
        print(f'   ❌ Failed to download {url}: {e}')
        return False

download_results = []
for doc in CMS_CHAPTERS:
    local_path = LOCAL_RAW_DIR / doc['manual_id'] / doc['filename']
    print(f'\n⬇️  Processing: {doc["manual_id"]}/{doc["filename"]}')
    success = download_pdf(doc['url'], local_path)
    
    if success:
        download_results.append({
            'manual_id': doc['manual_id'],
            'filename': doc['filename'],
            'local_path': local_path,
            's3_key': f'{S3_RAW_PREFIX}{doc["manual_id"]}/{doc["filename"]}'
        })

print(f'\n📊 Downloaded successfully: {len(download_results)}/{len(CMS_CHAPTERS)}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Upload Files to S3

# COMMAND ----------

upload_count = 0
skip_count = 0

for doc in download_results:
    local_path = doc['local_path']
    s3_key = doc['s3_key']
    
    print(f'\n⬆️  Uploading: {doc["manual_id"]}/{doc["filename"]}')
    
    try:
        # Check if already uploaded to avoid redundant transfers
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        print('   ⏭️  Already in S3 — skipping')
        skip_count += 1
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)
            print(f'   ✅ Uploaded to s3://{S3_BUCKET}/{s3_key}')
            upload_count += 1
        else:
            print(f'   ❌ S3 Error: {e}')

print(f'\n📊 S3 Upload Summary:')
print(f'   Uploaded: {upload_count}')
print(f'   Skipped (already exists): {skip_count}')

# COMMAND ----------