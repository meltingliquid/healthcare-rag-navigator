# test_scripts/verify_aws.py
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