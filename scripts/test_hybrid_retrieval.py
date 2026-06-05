"""
Interactive test script for the Hybrid Retriever (FAISS + BM25 + RRF).
Loads indexes locally, embeds your query with OpenAI, runs both searches
and shows the fused RRF-ranked results.

Usage:
    python scripts/test_hybrid_retrieval.py
"""

import os
import sys
import json
import pickle
import string
import tempfile
import pathlib
from collections import defaultdict

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Resolve paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')

# Local paths
FAISS_INDEX_PATH = PROJECT_ROOT / "retrieval" / "faiss_index.bin"
CHUNK_MAP_PATH   = PROJECT_ROOT / "retrieval" / "chunk_id_map.json"
BM25_INDEX_PATH  = PROJECT_ROOT / "retrieval" / "bm25_index.pkl"
CHUNKS_PATH      = PROJECT_ROOT / "data" / "chunks" / "chunks.json"

EMBED_MODEL = "text-embedding-3-small"
DIMENSIONS  = 1536

# ─── Load all indexes once ────────────────────────────────────────────────────

def load_indexes():
    missing = [p for p in [FAISS_INDEX_PATH, CHUNK_MAP_PATH, BM25_INDEX_PATH] if not p.exists()]
    if missing:
        print("❌ Missing local index files. Build them first:")
        for p in missing:
            print(f"   - {p}")
        sys.exit(1)

    print("\n⏳ Loading indexes...")

    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(CHUNK_MAP_PATH, "r") as f:
        chunk_id_map = json.load(f)

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25     = bm25_data["bm25"]
    bm25_ids = bm25_data["chunk_ids"]

    chunks_data = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks_list = json.load(f)
            chunks_data = {c["chunk_id"]: c for c in chunks_list}

    print(f"✅ FAISS: {faiss_index.ntotal} vecs | BM25: {len(bm25_ids)} docs | Chunks: {len(chunks_data)}")
    return faiss_index, chunk_id_map, bm25, bm25_ids, chunks_data

# ─── Tokenizer (must match build_bm25_index.py) ──────────────────────────────

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

# ─── Individual searchers ─────────────────────────────────────────────────────

def faiss_search(query: str, faiss_index, chunk_id_map, api_key: str, top_n: int = 20):
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(q_vec)

    distances, indices = faiss_index.search(q_vec, min(top_n, faiss_index.ntotal))
    return [
        (chunk_id_map[str(idx)], float(score))
        for score, idx in zip(distances[0], indices[0])
        if str(idx) in chunk_id_map
    ]

def bm25_search(query: str, bm25, bm25_ids, top_n: int = 20):
    scores = bm25.get_scores(tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [(bm25_ids[i], float(scores[i])) for i in top_idx]

# ─── Hybrid retrieve (RRF) ────────────────────────────────────────────────────

def retrieve(query: str, faiss_index, chunk_id_map, bm25, bm25_ids, chunks_data, api_key: str, top_k: int = 5):
    print(f"\n🔍 Running hybrid retrieval for: \"{query}\"")

    faiss_results = faiss_search(query, faiss_index, chunk_id_map, api_key)
    bm25_results  = bm25_search(query, bm25, bm25_ids)

    # RRF-60 fusion
    rrf: dict[str, float] = defaultdict(float)
    for rank, (cid, _) in enumerate(faiss_results, start=1):
        rrf[cid] += 1.0 / (60 + rank)
    for rank, (cid, _) in enumerate(bm25_results, start=1):
        rrf[cid] += 1.0 / (60 + rank)

    top_merged = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:15]

    results = [
        {
            "chunk_id":      cid,
            "chunk_text":    chunks_data.get(cid, {}).get("chunk_text", ""),
            "score":         round(rrf[cid], 6),
            "manual_id":     chunks_data.get(cid, {}).get("manual_id",     ""),
            "chapter_title": chunks_data.get(cid, {}).get("chapter_title", ""),
            "section_title": chunks_data.get(cid, {}).get("section_title", ""),
            "page_num":      chunks_data.get(cid, {}).get("page_num",       0),
        }
        for cid in top_merged[:top_k]
    ]

    # Display
    print(f"\n  {'Rank':<5} {'RRF Score':>10}   {'chunk_id':<35} {'Source'}")
    print("  " + "─" * 110)
    for i, r in enumerate(results, start=1):
        source = f"{r['manual_id']} › {r['chapter_title'][:30]} (p.{r['page_num']})"
        print(f"  {i:<5} {r['score']:>10.6f}   {r['chunk_id']:<35} {source}")
        snippet = r["chunk_text"][:80].replace("\n", " ")
        print(f"        Preview: {snippet}…\n")

    return results

# ─── Main interactive loop ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("   HYBRID RETRIEVAL TEST  (FAISS + BM25 + RRF)")
    print("="*55)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY missing from .env")
        sys.exit(1)

    faiss_idx, cid_map, bm25_model, bm25_ids, chunks_data = load_indexes()

    while True:
        try:
            user_query = input("\nEnter query (or 'quit' to exit): ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['quit', 'q', 'exit']:
                break

            top_k_input = input("How many results? (default=5): ").strip()
            top_k = int(top_k_input) if top_k_input.isdigit() else 5

            retrieve(user_query, faiss_idx, cid_map, bm25_model, bm25_ids, chunks_data, api_key, top_k=top_k)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
