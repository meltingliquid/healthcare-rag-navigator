# Databricks notebook source
# MAGIC %md
# MAGIC # Data Engineering Pipeline - Day 2
# MAGIC **Task:** Extract text from downloaded CMS PDFs, clean the text, and upload to S3.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install Dependencies

# COMMAND ----------

# %pip install boto3 python-dotenv tqdm pymupdf --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Imports and Environment Setup

# COMMAND ----------

import os
import sys
import json
import re
import pathlib
import boto3
import fitz  # PyMuPDF
from botocore.exceptions import ClientError
from typing import Dict, List

# Resolve root to import config
NOTEBOOK_DIR = pathlib.Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import variables from settings
from config.settings import S3_BUCKET, S3_EXTRACTED_PREFIX, S3_CLEANED_PREFIX

# Local directories
MANIFEST_PATH = PROJECT_ROOT / 'sources' / 'cms_download_manifest.json'
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
EXTRACTED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'extracted_text'
CLEANED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'cleaned_text'

EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

print('✅ Imports and Environment Setup OK')
print(f'   PROJECT_ROOT: {PROJECT_ROOT}')
print(f'   S3_BUCKET: {S3_BUCKET}')
print(f'   S3_EXTRACTED_PREFIX: {S3_EXTRACTED_PREFIX}')
print(f'   S3_CLEANED_PREFIX: {S3_CLEANED_PREFIX}')

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

def upload_to_s3(local_path: pathlib.Path, bucket: str, s3_key: str):
    """Uploads a file to S3 if not already present."""
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        print(f'   ⏭️  Already in S3 — skipping: {s3_key}')
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            s3.upload_file(str(local_path), bucket, s3_key)
            print(f'   ✅ Uploaded → s3://{bucket}/{s3_key}')
        else:
            print(f'   ❌ S3 Error for {s3_key}: {e}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Helper Functions for Metadata and Cleaning

# COMMAND ----------

def load_manifest(manifest_path: pathlib.Path) -> Dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_chapter_index(manifest: Dict) -> Dict[str, Dict]:
    index = {}
    for manual in manifest.get("manuals", []):
        for chapter in manual.get("chapters", []):
            index[chapter["local_filename"]] = {
                "manual_id": manual["manual_id"],
                "manual_title": manual["manual_title"],
                "chapter_num": str(chapter["chapter_num"]),
                "chapter_title": chapter["chapter_title"],
                "source_url": chapter["url"],
                "relevance": chapter.get("relevance", ""),
            }
    return index

CMS_HEADER_FOOTER_PATTERNS = [
    r"Medicare Benefit Policy Manual\s*",
    r"National Coverage Determinations Manual\s*",
    r"Medicare Claims Processing Manual\s*",
    r"Chapter \d+[\s\-–]+[^\n]*\n",
    r"Rev\.\s*\d+[,.]?\s*\d{2}-\d{2}-\d{2}[^\n]*\n",
    r"CMS Manual System\s*",
    r"Pub\.\s*100-\d+\s*",
    r"Table of Contents[\s\S]{0,200}(?=\n{2,})",
    r"\f",
]

def clean_text(raw_text: str) -> str:
    text = raw_text
    
    # Remove non-printable characters
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    
    # Remove headers/footers
    for pattern in CMS_HEADER_FOOTER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.I | re.M)
        
    # Fix broken words across lines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Collapse multiple spaces to single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Filter out noisy lines that are mostly numbers or separator characters
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if re.fullmatch(r"[\d\s]+", s):
            continue
        if re.fullmatch(r"[.\-_=\s]{5,}", s):
            continue
        lines.append(line)
        
    text = "\n".join(lines)
    # Replace more than 3 newlines with just 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()

print('✅ Helper functions defined')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Step 1: Extract Text from PDFs

# COMMAND ----------

def extract_text_from_pdf(pdf_path: pathlib.Path) -> List[Dict]:
    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            pages.append({
                "page_num": i + 1,
                "raw_text": text,
                "num_chars": len(text),
            })
        doc.close()
    except Exception as e:
        print(f"   [ERROR] Extracting from {pdf_path.name}: {e}")
    return pages

print("\n🚀 Starting PDF Text Extraction...")
manifest = load_manifest(MANIFEST_PATH)
chapter_idx = build_chapter_index(manifest)

pdfs = list(RAW_DIR.rglob("*.pdf"))
print(f"📋 Found {len(pdfs)} PDFs to process.")

for pdf in pdfs:
    print(f"\n📄 Extracting text from {pdf.name}...")
    
    pages = extract_text_from_pdf(pdf)
    metadata = chapter_idx.get(pdf.name, {})
    
    doc = {
        "metadata": metadata,
        "filename": pdf.name,
        "total_pages": len(pages),
        "pages": pages
    }
    
    out_name = f"{metadata.get('manual_id','unknown')}_{pdf.stem}.json"
    out_path = EXTRACTED_DIR / out_name
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        
    s3_key = f"{S3_EXTRACTED_PREFIX}{out_name}"
    upload_to_s3(out_path, S3_BUCKET, s3_key)

print("\n✅ PDF Extraction Complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Step 2: Clean Extracted Text

# COMMAND ----------

print("\n🚀 Starting Text Cleaning...")

extracted_files = list(EXTRACTED_DIR.glob("*.json"))
print(f"📋 Found {len(extracted_files)} files to clean.")

for file in extracted_files:
    print(f"\n🧹 Cleaning text for {file.name}...")
    
    with open(file, "r", encoding="utf-8") as f:
        doc = json.load(f)
        
    cleaned_pages = []
    for page in doc.get("pages", []):
        raw = page["raw_text"]
        clean = clean_text(raw)
        
        cleaned_pages.append({
            "page_num": page["page_num"],
            "raw_text": raw,
            "cleaned_text": clean,
            "num_chars": page["num_chars"],
            "num_chars_clean": len(clean)
        })
        
    doc["pages"] = cleaned_pages
    
    out_path = CLEANED_DIR / file.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        
    s3_key = f"{S3_CLEANED_PREFIX}{file.name}"
    upload_to_s3(out_path, S3_BUCKET, s3_key)

print("\n✅ Text Cleaning Complete.")

# COMMAND ----------