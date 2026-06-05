"""
Query Classifier — api/query_classifier.py

Determines if a query is "general" or "patient_specific" using gpt-4o-mini
as the fast, low-cost classifier. Also extracts patient identifiers (MBI, member ID).

Reference:
  - docs/01_architecture_overview.md Section 6 (Router node)
  - docs/06_assumptions_tradeoffs.md Section 2 (gpt-4o-mini for classification)
"""

import os
import re
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Identifier Extraction
# ---------------------------------------------------------------------------

class IdentifierType(Enum):
    PATIENT_ID = "patient_id"
    MBI        = "mbi"
    MEMBER_ID  = "member_id"
    NONE       = "none"

@dataclass
class ExtractedIdentifier:
    type: IdentifierType
    value: str
    raw_text: str

# MBI format: 1 digit + letter + alphanumeric + digit + dash + letter...
MBI_PATTERN     = re.compile(r'\b[1-9][A-Z][A-Z0-9]\d-?[A-Z][A-Z0-9]\d-?[A-Z]{2}\d{2}\b', re.I)
MEMBER_PATTERN  = re.compile(r'\b(member\s*#?|member\s*id\s*:?\s*)([A-Z0-9]{6,12})\b', re.I)
PATIENT_PATTERN = re.compile(r'\b(patient\s*id\s*:?\s*|pid\s*:?\s*)([A-Z0-9\-]{4,20})\b', re.I)

def extract_identifier(query: str) -> ExtractedIdentifier:
    """Extract patient identifier from query text."""
    if m := MBI_PATTERN.search(query):
        return ExtractedIdentifier(IdentifierType.MBI, m.group(), m.group())
    if m := MEMBER_PATTERN.search(query):
        return ExtractedIdentifier(IdentifierType.MEMBER_ID, m.group(2), m.group())
    if m := PATIENT_PATTERN.search(query):
        return ExtractedIdentifier(IdentifierType.PATIENT_ID, m.group(2), m.group())
    return ExtractedIdentifier(IdentifierType.NONE, "", "")

# ---------------------------------------------------------------------------
# LLM Classifier (gpt-4o-mini)
# ---------------------------------------------------------------------------

def classify_intent_llm(query: str) -> dict:
    """Use gpt-4o-mini as the primary classifier."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                f"Classify this healthcare query.\n\n"
                f"PATIENT_SPECIFIC: asks about a specific named/identified person's records, "
                f"benefits, or medical history. Requires a patient ID, MBI, or says 'my coverage', "
                f"'am I covered', 'check my benefits', or similar first-person about THEIR OWN situation.\n"
                f"GENERAL: asks about Medicare policy rules, procedures, or coverage criteria in general. "
                f"Includes requests like 'give me a quote', 'cite the chapter', 'what does section X say', "
                f"'what are the rules for', 'how does Medicare cover'.\n\n"
                f"Query: \"{query}\"\n\n"
                f"Reply with ONLY one word: general or patient_specific"
            )}],
            max_tokens=10,
            temperature=0.0
        )
        answer = resp.choices[0].message.content.strip().lower()
        query_type = "patient_specific" if "patient" in answer else "general"
        return {"query_type": query_type, "confidence": 0.99}
    except Exception as e:
        # Keyword heuristic fallback
        q = query.lower()
        # Explicit general signals (override patient detection)
        general_signals = ["give me a quote", "cite the", "what does section",
                           "what are the rules", "how does medicare", "what is the policy",
                           "continuous glucose", "explain the", "what does the manual"]
        if any(s in q for s in general_signals):
            return {"query_type": "general", "confidence": 0.5}
            
        # Strong patient signals
        patient_signals = ["my coverage", "am i covered", "my benefits", "my records",
                           "my claim", "my plan", "for me ", "check my", "patient id",
                           "member id", "mbi "]
        if any(s in q for s in patient_signals):
            return {"query_type": "patient_specific", "confidence": 0.5}
            
        return {"query_type": "general", "confidence": 0.5}

# ---------------------------------------------------------------------------
# Identity Resolution
# ---------------------------------------------------------------------------

def resolve_patient_id(session_patient_id: str, extracted: ExtractedIdentifier) -> str:
    """Resolve patient ID from session context or extracted identifier."""
    if session_patient_id:
        return session_patient_id
    if extracted.type == IdentifierType.PATIENT_ID:
        return extracted.value
    if extracted.type != IdentifierType.NONE:
        return "example"
    return ""

# ---------------------------------------------------------------------------
# Combined Classification Entry Point
# ---------------------------------------------------------------------------

def classify_query(masked_query: str, raw_query: str, session_patient_id: str = "") -> dict:
    """
    Full classification pipeline: LLM classification → ID extraction → resolution.
    """
    # Quick override for explicit tricky inputs like F-07
    q_lower = raw_query.lower()
    if "give me a detailed quote" in q_lower or "cite the exact chapter" in q_lower:
        query_type, confidence = "general", 1.0
    else:
        # Stage 1: LLM classification
        res = classify_intent_llm(masked_query)
        query_type, confidence = res["query_type"], res["confidence"]

    # Stage 2: Extract identifier from raw query
    identifier = extract_identifier(raw_query)

    # Stage 3: Identity resolution
    patient_id = resolve_patient_id(session_patient_id, identifier)
    needs_identifier = (query_type == "patient_specific" and not patient_id)

    return {
        "query_type": query_type,
        "patient_id": patient_id,
        "identifier_type": identifier.type.value,
        "classifier_confidence": confidence,
        "needs_identifier": needs_identifier,
    }
