import os
import sys
import json
import pathlib
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Resolve paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / '.env')

# Paths
FAISS_INDEX_PATH = PROJECT_ROOT / "retrieval" / "faiss_index.bin"
CHUNK_MAP_PATH = PROJECT_ROOT / "retrieval" / "chunk_id_map.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"

DIMENSIONS = 1536
MODEL = "text-embedding-3-small"

def load_index_and_map():
    if not FAISS_INDEX_PATH.exists() or not CHUNK_MAP_PATH.exists():
        print(f"❌ Requisite files missing. Have you built the FAISS index?\nLooked for:\n- {FAISS_INDEX_PATH}\n- {CHUNK_MAP_PATH}")
        sys.exit(1)
        
    print("\n⏳ Loading FAISS Index and Chunk Map...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CHUNK_MAP_PATH, "r") as f:
        chunk_map = json.load(f)
        
    chunks_data = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks_list = json.load(f)
            chunks_data = {c["chunk_id"]: c for c in chunks_list}
            
    print(f"✅ Loaded Index ({index.ntotal} vectors) and Map ({len(chunk_map)} entries).")
    return index, chunk_map, chunks_data

def get_embedding(query: str, api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
        
    print(f"🤖 Embedding query: '{query}'")
    res = client.embeddings.create(model=MODEL, input=query)
    embed = np.array(res.data[0].embedding, dtype="float32").reshape(1, -1)
    
    # Normalize for inner-product cosine similarity
    faiss.normalize_L2(embed)
    return embed

def test_query(query: str, index, chunk_map, chunks_data, api_key: str, k: int = 20):
    query_vector = get_embedding(query, api_key)
    
    print(f"\n🔍 Searching for top-{k} results...")
    distances, indices = index.search(query_vector, k)
    
    print(f"\n  {'Rank':<5} {'Score':>8}   {'chunk_id':<35} {'Preview'}")
    print("  " + "─" * 105)
    
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx == -1:  # Faiss returns -1 if not enough results
            continue
            
        cid = chunk_map.get(str(idx), "UNKNOWN")
        
        preview = ""
        if cid in chunks_data:
            text = chunks_data[cid].get("chunk_text", "").replace("\n", " ").strip()
            preview = text[:60] + "..." if len(text) > 60 else text
            
        print(f"  {rank:<5} {score:>8.4f}   {cid:<35} {preview}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("     FAISS SEMANTIC RETRIEVAL TEST SCRIPT")
    print("="*50)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY is missing from .env")
        sys.exit(1)
        
    # Load into memory once
    idx, cmap, cdata = load_index_and_map()
    
    while True:
        try:
            user_query = input("\nEnter search query (or 'quit' to exit): ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['quit', 'q', 'exit']:
                break
                
            test_query(user_query, idx, cmap, cdata, api_key, k=20)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
