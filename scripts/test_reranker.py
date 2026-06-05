"""
Interactive test script for the Cross-Encoder Reranker.
Loads the hybrid retriever to fetch top 15 results, then passes them to the reranker.

Usage:
    python scripts/test_reranker.py
"""

import os
import sys
import time
import pathlib

# Resolve paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

def load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

print("\n" + "="*55)
print("   CROSS-ENCODER RERANKER TEST")
print("="*55)

print("\n⏳ Bootstrapping Hybrid Retriever and Reranker...")
hr_module = load_module("hybrid_retriever", "retrieval/04_hybrid_retriever.py")
rr_module = load_module("reranker", "retrieval/05_reranker.py")

retrieve = hr_module.retrieve
rerank = rr_module.rerank

test_queries = [
    "home health coverage",
    "§40.1 HCPCS G0179",
    "hypertension diagnosis 130 mmHg",
]

latencies = []

print("\n🔍 Running benchmark tests...")
for q in test_queries:
    print("\n" + "-" * 65)
    print(f"  QUERY: '{q}'")

    # Fetch 15 broad candidates
    broad = retrieve(query=q, top_k=15)
    print(f"  Retrieved {len(broad)} candidates via Hybrid FAISS+BM25.")

    # Time the reranker
    t0 = time.perf_counter()
    top5 = rerank(query=q, candidates=broad, top_k=5)
    latency_ms = (time.perf_counter() - t0) * 1000
    latencies.append(latency_ms)

    print(f"  Reranked to Top-5 in {latency_ms:.1f} ms")
    print("-" * 65)
    for i, r in enumerate(top5, 1):
        print(f"  {i}. score={r['score']:8.4f} | {r['chunk_id']}")
        snip = r['chunk_text'][:90].replace('\n', ' ')
        print(f"     Preview: {snip}…")

if latencies:
    avg = sum(latencies) / len(latencies)
    print("\n" + "=" * 65)
    print(f"  Avg rerank latency: {avg:.1f} ms  (target < 200 ms)")
    if avg > 200:
        print("  ⚠️ NOTE: CPU execution may exceed GPU SLA. Acceptable for local MVP.")

# Interactive loop
print("\n" + "="*55)
print("   INTERACTIVE TEST")
print("="*55)
while True:
    try:
        user_query = input("\nEnter query (or 'quit' to exit): ").strip()
        if not user_query:
            continue
        if user_query.lower() in ['quit', 'q', 'exit']:
            break

        print("\n⏳ Fetching candidates via hybrid retrieval...")
        broad = retrieve(query=user_query, top_k=15)
        
        t0 = time.perf_counter()
        top_k = rerank(query=user_query, candidates=broad, top_k=5)
        latency = (time.perf_counter() - t0) * 1000
        
        print(f"\n✅ Reranked {len(broad)} candidates to top-5 in {latency:.1f}ms")
        for i, r in enumerate(top_k, 1):
            print(f"  {i}. [Score: {r['score']:8.4f}] {r['chunk_id']}")
            snip = r['chunk_text'][:80].replace('\n', ' ')
            print(f"     Preview: {snip}…\n")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
