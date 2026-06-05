"""
Session Manager — api/session_manager.py

Implements an in-memory, TTL-based session store as required by the MVP architecture.
This module is explicitly for SECURITY and COMPLIANCE:
- Stores query history and patient context.
- Sessions automatically expire and are DELETED from memory after 15 minutes of inactivity.
- No data is ever persisted to disk.

Reference:
- docs/01_architecture_overview.md Section 1 (Decision #5)
- docs/06_assumptions_tradeoffs.md Section 2 (Ephemeral storage)
"""

import time
import uuid
from typing import Dict, Any

# 15 minutes in seconds
SESSION_TTL_SECONDS = 15 * 60

# In-memory dictionary to hold all active sessions.
# Format: { session_id (str): {"last_accessed": float, "history": list, "patient_context": dict} }
_SESSIONS: Dict[str, Dict[str, Any]] = {}

def _now() -> float:
    return time.time()

def purge_expired_sessions() -> int:
    """
    Scans the in-memory dictionary and securely deletes any sessions 
    that have exceeded the TTL limit.
    Returns the number of sessions deleted.
    """
    current_time = _now()
    expired_ids = []
    
    for session_id, data in _SESSIONS.items():
        if current_time - data["last_accessed"] > SESSION_TTL_SECONDS:
            expired_ids.append(session_id)
            
    for session_id in expired_ids:
        # Del explicitly removes the reference from memory
        del _SESSIONS[session_id]
        
    return len(expired_ids)

def create_session() -> str:
    """
    Creates a new empty session and returns its unique ID.
    Always purges expired sessions first to maintain memory hygiene.
    """
    purge_expired_sessions()
    
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "last_accessed": _now(),
        "history": [],
        "patient_context": {}
    }
    return session_id

def get_session(session_id: str) -> Dict[str, Any]:
    """
    Retrieves a session if it exists and hasn't expired.
    Updates the 'last_accessed' timestamp for TTL extension.
    Returns None if the session does not exist or has expired.
    """
    purge_expired_sessions()
    
    session = _SESSIONS.get(session_id)
    if session:
        # Extend the TTL
        session["last_accessed"] = _now()
        return session
    
    return None

def update_session(session_id: str, history_entry: dict = None, patient_context: dict = None) -> bool:
    """
    Updates the session with new conversational history or patient context.
    Returns True if successful, False if session doesn't exist/expired.
    """
    session = get_session(session_id)
    if not session:
        return False
        
    if history_entry:
        session["history"].append(history_entry)
        
    if patient_context is not None:
        session["patient_context"] = patient_context
        
    return True

def delete_session(session_id: str) -> bool:
    """
    Explicitly force-deletes a session and its data from memory.
    Useful for explicit logouts or end-of-conversations.
    """
    if session_id in _SESSIONS:
        del _SESSIONS[session_id]
        return True
    return False


# ==============================================================================
# Testing Block (Comment out later)
# ==============================================================================
if __name__ == "__main__":
    print("--- Running Session Manager Tests ---")
    
    # Test 1: Creation and Retrieval
    sid = create_session()
    sess = get_session(sid)
    print(f"\n[Test 1] Session created: {sid}")
    print(f"Session data: {sess}")
    assert sess is not None
    assert sess["history"] == []
    
    # Test 2: Updating Session
    update_session(sid, history_entry={"query": "Am I covered?", "response": "Yes"}, patient_context={"id": "example", "conditions": ["Diabetes"]})
    sess_updated = get_session(sid)
    print(f"\n[Test 2] Session updated:")
    print(f"History: {sess_updated['history']}")
    print(f"Patient Context: {sess_updated['patient_context']}")
    
    # Test 3: Purge Mechanism
    # Temporarily override TTL to test purge
    original_ttl = SESSION_TTL_SECONDS
    SESSION_TTL_SECONDS = 2  # 2 second TTL for testing
    
    sid2 = create_session()
    print(f"\n[Test 3] Session 2 created. Simulating wait of 3 seconds...")
    time.sleep(3)
    
    purged_count = purge_expired_sessions()
    print(f"Sessions purged: {purged_count} (Expected at least 1)")
    
    missing_sess = get_session(sid2)
    print(f"Session 2 retrieval after wait: {missing_sess} (Expected None)")
    
    # Cleanup overriding
    SESSION_TTL_SECONDS = original_ttl
    print("\n--- All tests completed successfully ---")
