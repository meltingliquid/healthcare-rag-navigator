# Databricks notebook source
# MAGIC %md
# MAGIC # AI/ML Pipeline - Day 4
# MAGIC **Task:** Cross-Encoder Reranker

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install Dependencies

# COMMAND ----------

# %pip install sentence-transformers numpy --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Imports and Environment Setup

# COMMAND ----------

import os
import sys
import time
import pathlib
import warnings

# Suppress Hugging Face symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import CrossEncoder

# Resolve root to import config
NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name in ['notebooks', 'retrieval'] else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("✅ Imports and Environment Setup OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Load Cross-Encoder Model

# COMMAND ----------

_MODEL_NAME = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

print(f"⏳ Loading reranker model: {_MODEL_NAME}...")
try:
    _reranker_model = CrossEncoder(_MODEL_NAME, max_length=512)
    print("✅ Model loaded successfully.")
except Exception as _e:
    _reranker_model = None
    print(f"⚠️ WARNING: Model load failed ({_e}). Will fall back to RRF scores.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Rerank Function

# COMMAND ----------

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-scores candidates against the query using a cross-encoder and returns top_k.

    Args:
        query      : Raw user query string.
        candidates : Output of hybrid_retriever.retrieve(query, top_k=15).
                     Each dict must contain 'chunk_text' and 'score' (RRF).
        top_k      : Number of final chunks to return.

    Returns:
        list[dict] — same schema as input, 'score' replaced with cross-encoder logit,
                     sorted descending. Falls back to RRF ordering on any failure.
    """
    if not candidates:
        return []

    # Fallback: model unavailable
    if _reranker_model is None:
        return candidates[:top_k]

    pairs = [(query, doc.get("chunk_text", "")) for doc in candidates]

    try:
        scores = _reranker_model.predict(pairs)  # numpy array
        
        # Create shallow copies to avoid mutating original dictionary references unexpectedly
        reranked_candidates = []
        for doc, score in zip(candidates, scores):
            new_doc = doc.copy()
            new_doc["score"] = float(score)
            reranked_candidates.append(new_doc)
            
        return sorted(reranked_candidates, key=lambda d: d["score"], reverse=True)[:top_k]

    except Exception as e:
        # Fallback: inference failed (OOM, extreme length, etc.)
        print(f"⚠️ [reranker] Inference failed ({e}). Falling back to RRF scores.")
        return candidates[:top_k]

print("✅ rerank() function defined and ready")

# COMMAND ----------