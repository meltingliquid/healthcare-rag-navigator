"""
MVP Query Pipeline
==================
Flow:
  user query
    → sanitize & mask PHI
    → classify (general vs patient-specific)
    → [general]          → retrieve → generate → show result
    → [patient-specific] → extract identifiers → FHIR fetch
                         → summarise FHIR JSON → combine with masked query
                         → retrieve → generate → show result
"""

import os
import sys
import json
import pathlib
from typing import TypedDict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path().resolve()
_candidate = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR

# Databricks Apps mounts source at /app/python/source_code — detect this explicitly
_DATABRICKS_APPS_ROOT = pathlib.Path("/app/python/source_code")
if _DATABRICKS_APPS_ROOT.exists():
    PROJECT_ROOT = _DATABRICKS_APPS_ROOT
else:
    PROJECT_ROOT = _candidate

print(f"  📁 PROJECT_ROOT resolved to: {PROJECT_ROOT}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# ── Your existing API modules (unchanged) ─────────────────────────────────────
from api.input_sanitizer import sanitize, check_injection
from api.phi_masker import mask_phi
from api.fhir_client import fetch_patient_context
from api.audit_logger import log_interaction

# ── Retrieval (Member B) ──────────────────────────────────────────────────────
import importlib.util

# Debug: show what's actually deployed in the container
print(f"  📂 Top-level dirs in PROJECT_ROOT: {[p.name for p in PROJECT_ROOT.iterdir() if p.is_dir()]}")
_retrieval_dir = PROJECT_ROOT / "retrieval"
if _retrieval_dir.exists():
    print(f"  📂 retrieval/ contents: {[p.name for p in _retrieval_dir.iterdir()]}")
else:
    print(f"  ❌ retrieval/ folder NOT FOUND at {_retrieval_dir}")

def _load_module(name: str, relative_path: str):
    """Load a Python module by file path.
    Falls back to extension-less path since Databricks Workspace strips .py
    from notebook-style files when deploying via Apps."""
    import types
    full_path = PROJECT_ROOT / relative_path
    if not full_path.exists():
        # Databricks Apps strips .py from notebook-style files
        full_path = PROJECT_ROOT / relative_path.replace(".py", "")
    print(f"  📦 Loading {name} from: {full_path} (exists={full_path.exists()})")
    # Use compile+exec so we don't depend on .py extension for import machinery
    source = full_path.read_text(encoding="utf-8")
    mod = types.ModuleType(name)
    mod.__file__ = str(full_path)
    sys.modules[name] = mod
    exec(compile(source, str(full_path), "exec"), mod.__dict__)
    return mod

try:
    _hr = _load_module("hybrid_retriever", "retrieval/04_hybrid_retriever.py")
    retrieve_fn = _hr.retrieve
    print("✅ Retrieval module loaded")
except Exception as e:
    print(f"❌ Failed to load retrieval module: {e}")
    retrieve_fn = None

try:
    _reranker = _load_module("reranker", "retrieval/05_reranker.py")
    rerank_fn = _reranker.rerank
    print("✅ Reranker module loaded")
except Exception as e:
    print(f"❌ Failed to load reranker module: {e}")
    rerank_fn = None

# ── OpenAI client ─────────────────────────────────────────────────────────────
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    _openai_key = dbutils.secrets.get(scope="capstone_scope", key="openai_api_key")
else:
    _openai_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=_openai_key)


SYSTEM_PROMPT_PATH = PROJECT_ROOT / "retrieval" / "prompts" / "system_prompt.txt"
try:
    SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    SYSTEM_PROMPT = (
        "You are a helpful healthcare policy assistant. "
        "Return valid JSON with keys: 'answer', 'citations', 'confidence', 'patient_context_used'."
    )

# ── State ─────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    original_query:   str
    query:            str              # sanitised + masked, evolves through pipeline
    deanon_map:       dict
    is_injection:     bool
    query_type:       str              # "general" | "patient_specific"
    patient_id:       str
    patient_context:  dict
    retrieved_chunks: List[dict]
    chunks_relevant:  bool
    retries:          int
    final_response:   dict
    error:            str

# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_sanitize(state: GraphState) -> dict:
    """Sanitise input and block prompt injections."""
    q = state["original_query"]
    if check_injection(q):
        return {"is_injection": True, "error": "Prompt injection detected. Request blocked."}
    return {"is_injection": False, "query": sanitize(q), "error": ""}


def node_mask_phi(state: GraphState) -> dict:
    """Mask PHI using Presidio."""
    masked_query, deanon = mask_phi(state["query"])
    return {"query": masked_query, "deanon_map": deanon}


from api.query_classifier import classify_query

def node_classify(state: GraphState) -> dict:
    """
    Classify query as general or patient_specific using api module.
    """
    try:
        result = classify_query(
            masked_query=state["query"],
            raw_query=state["original_query"],
            session_patient_id=state.get("patient_id", "")
        )
        return {
            "query_type": result.get("query_type", "general"),
            "patient_id": result.get("patient_id", "")
        }
    except Exception as e:
        print(f"  ⚠️ Classify failed ({e}), defaulting to general")
        return {"query_type": "general", "patient_id": ""}


def node_fetch_fhir(state: GraphState) -> dict:
    """Fetch patient data from HAPI FHIR server."""
    print(f"  🏥 Fetching FHIR data for patient: {state['patient_id']}")
    patient_context = fetch_patient_context(state["patient_id"])
    return {"patient_context": patient_context}


def node_summarise_fhir(state: GraphState) -> dict:
    """
    Summarise the FHIR JSON response into a short plain-text description.
    Then append it to the masked query so retrieval has full context.
    """
    pc = state.get("patient_context", {})
    if not pc:
        return {}

    prompt = f"""Summarise the following FHIR patient data in 3-5 plain sentences.
Focus on: coverage type, active conditions, current medications, and any relevant demographics.
Do NOT include names, MBI, or any direct identifiers — use "the patient" instead.

FHIR data:
{json.dumps(pc, indent=2)[:3000]}"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        fhir_summary = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️ FHIR summarisation failed ({e}), using raw fields")
        fhir_summary = (
            f"Coverage: {pc.get('coverage_type', 'Unknown')}. "
            f"Conditions: {', '.join(pc.get('conditions', []))}. "
            f"Medications: {', '.join(pc.get('medications', []))}."
        )

    # We no longer pollute state["query"] with the summary. 
    # Instead, we just store it in state["patient_context"]["summary"] for the generator to use.
    pc["summary"] = fhir_summary
    print(f"  📋 FHIR summary generated")
    return {"patient_context": pc}


def node_retrieve(state: GraphState) -> dict:
    """Hybrid retrieval (FAISS + BM25)."""
    print("  🔍 Retrieving...")
    if retrieve_fn:
        # Increase top_k for better candidate pool for reranking
        chunks = retrieve_fn(query=state["query"], top_k=20)
        print(f"     Retrieved {len(chunks)} chunks")
    else:
        print("     ⚠️ Retrieve module unavailable")
        chunks = []
    return {"retrieved_chunks": chunks}


def node_rerank(state: GraphState) -> dict:
    """Cross-encoder reranking for top precision."""
    print("  🚀 Reranking chunks...")
    chunks = state.get("retrieved_chunks", [])
    if not chunks or not rerank_fn:
        return {"retrieved_chunks": chunks}
    
    # Reranker expects candidates to rank. We take top 100 for re-ranking if available
    ranked = rerank_fn(query=state["query"], candidates=chunks, top_k=5)
    print(f"     Reranked top {len(ranked)} chunks")
    return {"retrieved_chunks": ranked}


def node_check(state: GraphState) -> dict:
    """Relevance check: ask gpt-4o-mini if retrieved chunks answer the query."""
    print("  ✓ Checking chunk relevance...")
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"chunks_relevant": False}

    context_preview = "\n\n".join(
        [f"[Chunk {i+1}] {c.get('chunk_text', '')[:400]}" for i, c in enumerate(chunks[:5])]
    )
    prompt = (
        f"You are checking if retrieved healthcare policy chunks are relevant to a user query.\n"
        f"Query: {state['query']}\n\n"
        f"Retrieved Context:\n{context_preview}\n\n"
        f"Instructions: Reply YES if ANY of the chunks contain rules or concepts relevant to answering the query. "
        f"Reply NO only if ALL chunks are completely unrelated.\n"
        f"Reply strictly with YES or NO."
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        answer = resp.choices[0].message.content.strip().upper()
        relevant = "YES" in answer
        print(f"     Relevance: {answer}")
        return {"chunks_relevant": relevant}
    except Exception as e:
        print(f"     ⚠️ Relevance check failed: {e}. Defaulting to relevant.")
        return {"chunks_relevant": True}


def node_reformulate(state: GraphState) -> dict:
    """Reformulate query using broader CMS terminology if retrieval failed."""
    retries = state.get("retries", 0) + 1
    print(f"  🔄 Reformulating query (retry {retries})...")
    
    try:
        prompt = (
            f"The query '{state['query']}' failed to retrieve relevant healthcare policy documents. "
            f"Please rewrite the query to use broader Medicare/CMS terminology (e.g., 'blood glucose monitor' instead of 'CGM', 'NCD rules', etc). "
            f"Output strictly the rewritten query and nothing else."
        )
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        new_query = resp.choices[0].message.content.strip()
    except Exception:
        new_query = state["query"] + " Medicare policy requirements"
        
    print(f"     New query: {new_query}")
    return {"query": new_query, "retries": retries}


def node_generate(state: GraphState) -> dict:
    """Generate final answer from retrieved chunks."""
    print("  📝 Generating answer...")
    chunks = state.get("retrieved_chunks", [])

    context_str = ""
    for i, c in enumerate(chunks, 1):
        context_str += (
            f"\n--- CHUNK {i} ---\n"
            f"Source: Manual {c.get('manual_id')}, Chapter: {c.get('chapter_title')}, "
            f"Section: {c.get('section_title')}, Page {c.get('page_num')}\n"
            f"URL: {c.get('source_url', 'N/A')}\n"
            f"Text: {c.get('chunk_text')}\n"
        )

    # Prepare Patient Context for the LLM if available
    pc = state.get("patient_context", {})
    patient_context_str = ""
    if pc and pc.get("summary"):
        patient_context_str = f"Patient Context (masked):\n{pc['summary']}\n\n"

    # Use the original clean policy question for the LLM query field.
    # Patient context is passed separately so the LLM personalises the answer
    # rather than restricting it to the patient's specific conditions.
    display_query = state.get("original_query") or state["query"]

    user_prompt = (
        f"Query: {display_query}\n\n"
        f"{patient_context_str}"
        f"Retrieval Context:\n{context_str if context_str else 'No context retrieved.'}"
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return {"final_response": parsed}
    except Exception as e:
        print(f"  ❌ Generation error: {e}")
        return {
            "final_response": {
                "answer": "Generation failed. Please try again.",
                "citations": [],
                "confidence": {"level": "Low", "reasoning": str(e)},
                "patient_context_used": False,
            }
        }


def node_log(state: GraphState) -> dict:
    """HIPAA-compliant audit log — masked query only, never raw PHI."""
    is_injection = state.get("is_injection", False)
    query_masked = "[BLOCKED]" if is_injection else state.get("query", "")
    final_response = state.get("final_response", {})
    chunks = state.get("retrieved_chunks", [])

    req_id = log_interaction(
        query_masked=query_masked,
        chunks_retrieved=[c.get("chunk_id", "unknown") for c in chunks],
        confidence_score=final_response.get("confidence", {}).get("retrieval_score", 0.0),
        response_draft=final_response.get("answer", state.get("error", "")),
        patient_context_used=state.get("query_type") == "patient_specific",
        session_id="anonymous",
    )

    if final_response:
        final_response["request_id"] = req_id

    return {"final_response": final_response}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("sanitize",       node_sanitize)
    graph.add_node("mask_phi",       node_mask_phi)
    graph.add_node("classify",       node_classify)
    graph.add_node("fetch_fhir",     node_fetch_fhir)
    graph.add_node("summarise_fhir", node_summarise_fhir)
    graph.add_node("retrieve",       node_retrieve)
    graph.add_node("rerank",         node_rerank)
    graph.add_node("check",          node_check)
    graph.add_node("reformulate",    node_reformulate)
    graph.add_node("generate",       node_generate)
    graph.add_node("log",            node_log)

    # Routing
    def route_after_sanitize(state: GraphState) -> str:
        return "log" if state["is_injection"] else "mask_phi"

    def route_after_classify(state: GraphState) -> str:
        return "fetch_fhir" if state["query_type"] == "patient_specific" else "retrieve"

    def route_after_check(state: GraphState) -> str:
        if state.get("chunks_relevant"):
            return "generate"
        if state.get("retries", 0) >= 2:
            return "generate"
        return "reformulate"

    # Edges
    graph.add_edge(START, "sanitize")
    graph.add_conditional_edges("sanitize",  route_after_sanitize)
    graph.add_edge("mask_phi",               "classify")
    graph.add_conditional_edges("classify",  route_after_classify)

    # Patient-specific path: FHIR → summarise → retrieve
    graph.add_edge("fetch_fhir",             "summarise_fhir")
    graph.add_edge("summarise_fhir",         "retrieve")

    # Retrieval loop
    graph.add_edge("retrieve",               "rerank")
    graph.add_edge("rerank",                 "check")
    graph.add_conditional_edges("check",     route_after_check)
    graph.add_edge("reformulate",            "retrieve")

    # End
    graph.add_edge("generate",               "log")
    graph.add_edge("log",                    END)

    return graph.compile()



app = build_graph()
print("✅ MVP pipeline compiled")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_query(query: str) -> dict:
    """Run a single query through the MVP pipeline and return the final response."""
    initial_state = {
        "original_query":   query,
        "query":            "",
        "deanon_map":       {},
        "is_injection":     False,
        "query_type":       "",
        "patient_id":       "",
        "patient_context":  {},
        "retrieved_chunks": [],
        "final_response":   {},
        "error":            "",
    }

    final_state = {}
    for step in app.stream(initial_state):
        for node_name, state_update in step.items():
            print(f"  → [{node_name}]")
            if state_update:
                final_state.update(state_update)
                if state_update.get("error"):
                    print(f"    ⚠️  {state_update['error']}")
                for k, v in state_update.items():
                    if k == "retrieved_chunks":
                        print(f"      [DEBUG] {k}: {len(v)} chunks")
                    elif isinstance(v, dict) and k != "patient_context" and k != "final_response":
                        print(f"      [DEBUG] {k}: {{...}} ({len(v)} keys)")
                    else:
                        v_str = str(v)
                        if len(v_str) > 200:
                            v_str = v_str[:197] + "..."
                        print(f"      [DEBUG] {k}: {v_str}")

    return final_state.get("final_response", {})


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tests for Patient 131597951 (Aayush) - Cardiac/Cardiovascular Focus
    tests = {
        "F-01: Home Health Skilled Nursing": (
            "Patient ID 131597951. Based on this patient's heart failure and hypertension, "
            "do they qualify for Medicare home health skilled nursing for "
            "medication management and cardiac monitoring?"
        ),
        "F-02: Homebound Status": (
            "Patient ID 131597951. Given this patient's cardiovascular conditions and limited mobility, "
            "what criteria determine whether they are considered homebound "
            "for Medicare home health coverage?"
        ),
        "F-03: Skilled Nursing Observation": (
            "Patient ID 131597951. For this patient managing diabetes and hypertension, what skilled "
            "nursing observation and assessment services can be covered under "
            "Medicare home health?"
        ),
        "F-04: SNF Requirements": (
            "Patient ID 131597951. If this patient with heart failure is discharged from a hospital "
            "after a cardiac event, what are the Medicare SNF coverage requirements "
            "for continued skilled nursing care?"
        ),
        "F-05: Ambulance Emergency": (
            "Patient ID 131597951. Under Medicare, when is ambulance transport covered for a patient "
            "experiencing an acute cardiac event like a hypertensive crisis?"
        ),
        "F-06: CVD Treatments (NCD)": (
            "Patient ID 131597951. What does Medicare's National Coverage Determination policy say "
            "about coverage for cardiovascular disease treatment and monitoring?"
        ),
        "F-07: Hospice Election": (
            "Patient ID 131597951. If this patient with advanced heart failure has a terminal prognosis, "
            "what Medicare hospice services would be covered and what conditions "
            "must be met for election?"
        ),
    }

    for label, q in tests.items():
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  Query: {q}")
        print(f"{'='*60}")
        result = run_query(q)
        print(f"\n  Answer:     {result.get('answer', 'N/A')[:300]}")
        print(f"  Citations:  {result.get('citations', [])}")
        print(f"  Confidence: {result.get('confidence', {})}")
        print(f"  Patient ctx used: {result.get('patient_context_used', False)}")
        print(f"  Request ID: {result.get('request_id', 'N/A')}")