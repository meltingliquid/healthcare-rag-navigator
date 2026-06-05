import os
import sys
import json
import string
import pickle
import pathlib
from dotenv import load_dotenv

# Resolve paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / '.env')

# Paths
BM25_INDEX_PATH = PROJECT_ROOT / "retrieval" / "bm25_index.pkl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def load_index():
    if not BM25_INDEX_PATH.exists():
        print(f"❌ Index file missing. Have you built the BM25 index?\nLooked for:\n- {BM25_INDEX_PATH}")
        sys.exit(1)
        
    print("\n⏳ Loading BM25 Index...")
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
        bm25 = data["bm25"]
        chunk_ids = data["chunk_ids"]
        
    chunks_data = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks_list = json.load(f)
            chunks_data = {c["chunk_id"]: c for c in chunks_list}
            
    print(f"✅ Loaded BM25 Index ({len(chunk_ids)} chunks).")
    return bm25, chunk_ids, chunks_data

def test_query(query: str, bm25, chunk_ids, chunks_data, k: int = 20):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    # Sort descending
    top_n = min(k, len(scores))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    
    print(f"\n🔍 Searching for top-{k} results...")
    print(f"\n  {'Rank':<5} {'Score':>8}   {'chunk_id':<35} {'Preview'}")
    print("  " + "─" * 105)
    
    for rank, idx in enumerate(top_idx, start=1):
        cid = chunk_ids[idx]
        score = scores[idx]
        if score == 0.0:
            continue
            
        preview = ""
        if cid in chunks_data:
            text = chunks_data[cid].get("chunk_text", "").replace("\n", " ").strip()
            preview = text[:60] + "..." if len(text) > 60 else text
            
        print(f"  {rank:<5} {score:>8.4f}   {cid:<35} {preview}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("     BM25 EXACT MATCH RETRIEVAL TEST SCRIPT")
    print("="*50)

    # Load into memory once
    bm25_model, c_ids, c_data = load_index()
    
    while True:
        try:
            user_query = input("\nEnter search query (or 'quit' to exit): ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['quit', 'q', 'exit']:
                break
                
            test_query(user_query, bm25_model, c_ids, c_data, k=20)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
