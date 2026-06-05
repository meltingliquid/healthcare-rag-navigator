"""
Audit Logger — api/audit_logger.py

Implements JSONL-based audit logging for the backend. Logs all interactions
strictly utilizing the masked/sanitized variants of queries. NEVER logs raw PHI
or complete prompts per docs/02_security_compliance.md.

Logs are stored locally in 'logs/audit_log.jsonl' and can be synchronized
periodically to AWS S3.
"""

import json
import logging
import os
import time
import uuid
import boto3
from dotenv import load_dotenv

load_dotenv()

# The logs directory should be strictly at the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "audit_log.jsonl")

# S3 Configuration from the .env file locally, or Databricks secrets in cloud
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_LOGS_PREFIX = os.getenv("S3_LOGS_PREFIX", "logs/audit_log.jsonl")

# Ensure logs output directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def log_interaction(
    query_masked: str,
    chunks_retrieved: list,
    confidence_score: float,
    response_draft: str,
    patient_context_used: bool,
    session_id: str = None
) -> str:
    """
    Logs the interaction to a local JSONL file efficiently.
    Returns the unique request ID assigned to the log entry.
    """
    request_id = str(uuid.uuid4())
    
    # Securely restrict the response field to a tiny preview to save analytical DB space natively
    if response_draft:
        response_preview = (response_draft[:200] + "...") if len(response_draft) > 200 else response_draft
    else:
        response_preview = ""
    
    entry = {
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_masked": query_masked,
        "chunks_retrieved": chunks_retrieved,
        "confidence_score": confidence_score,
        "response_preview": response_preview,
        "patient_context_used": patient_context_used,
        "session_id": session_id or "anonymous"
    }
    
    # Write to local JSONL
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except IOError as e:
        print(f"⚠️ Failed to write to robust audit log: {e}")
        
    return request_id

def upload_logs_to_s3():
    """Periodically call this to sync local audit logs safely to S3."""
    if not S3_BUCKET_NAME:
        print("⚠️ No S3 bucket config (S3_BUCKET_NAME). Skipping S3 upload.")
        return False
        
    if not os.path.exists(LOG_FILE):
        print("⚠️ No local logs found to upload.")
        return False
        
    try:
        # Load credentials implicitly via boto3 standard chaining (Env vars -> IAM roles)
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        s3.upload_file(LOG_FILE, S3_BUCKET_NAME, S3_LOGS_PREFIX)
        print(f"✅ Successfully uploaded audit logs to s3://{S3_BUCKET_NAME}/{S3_LOGS_PREFIX}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to upload audit logs to S3: {e}")
        return False

# ==============================================================================
# Testing Block
# ==============================================================================
if __name__ == "__main__":
    print("--- Running Audit Logger Validations ---")
    
    # 1. Test Writing Logs
    r_id = log_interaction(
        query_masked="Check Medicare provisions for [MASKED_DISEASE].",
        chunks_retrieved=["doc1", "doc2"],
        confidence_score=0.92,
        response_draft="Under Medicare Part B, you have complete coverage depending on co-pay variables.",
        patient_context_used=False
    )
    
    print(f"✅ Logged entry with Request ID: {r_id} to {LOG_FILE}")
    print("\nFile Preview:")
    with open(LOG_FILE, "r") as f:
        print(f.readlines()[-1].strip())
        
    # 2. Test S3 Synchronization Trigger
    print("\nS3 Test:")
    upload_logs_to_s3()
