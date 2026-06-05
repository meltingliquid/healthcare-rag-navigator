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