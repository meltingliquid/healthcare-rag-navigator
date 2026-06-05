"""
FHIR API Client — api/fhir_client.py

Fetches patient data from the public HAPI FHIR server and strips all PHI,
returning ONLY clinical context (conditions, medications, coverage_type, age_range).

Reference:
  - docs/02_security_compliance.md Section 2 (FHIR stripping rules)
  - docs/01_architecture_overview.md Section 3 (how FHIR data is used)

Env var required:
  HAPI_FHIR_BASE_URL=https://hapi.fhir.org/baseR4
"""

import os
import json
import requests
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HAPI_FHIR_BASE_URL = os.getenv("HAPI_FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
TIMEOUT_SECONDS = 10
FHIR_SAMPLES_DIR = Path(__file__).parent.parent / "data" / "fhir_samples"


# ---------------------------------------------------------------------------
# Internal helpers — PHI extraction (keep) and dropping (discard)
# ---------------------------------------------------------------------------

def _extract_conditions(bundle: dict) -> list[str]:
    """Extract condition display names from a FHIR Condition bundle."""
    conditions = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        code = resource.get("code", {})
        # Prefer the human-readable text; fall back to first coding display
        display = code.get("text") or ""
        if not display:
            for coding in code.get("coding", []):
                display = coding.get("display", "")
                if display:
                    break
        if display:
            conditions.append(display)
    return conditions


def _extract_medications(bundle: dict) -> list[str]:
    """Extract medication display names from a FHIR MedicationRequest bundle."""
    medications = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "MedicationRequest":
            continue
        med = resource.get("medicationCodeableConcept", {})
        display = med.get("text") or ""
        if not display:
            for coding in med.get("coding", []):
                display = coding.get("display", "")
                if display:
                    break
        if display:
            medications.append(display)
    return medications


def _extract_coverage_type(bundle: dict) -> str:
    """Extract the coverage type (e.g. 'Medicare Part A') from a FHIR Coverage bundle."""
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Coverage":
            continue
        coverage_type = resource.get("type", {})
        display = coverage_type.get("text") or ""
        if not display:
            for coding in coverage_type.get("coding", []):
                display = coding.get("display", "")
                if display:
                    break
        if display:
            return display
    return "Unknown"


def _extract_age_range(patient: dict) -> str:
    """
    Convert exact DOB to an age RANGE (e.g. '65-70'), never the exact DOB.
    Per doc 02: exact DOB is PHI; age range is not.
    """
    birth_date_str = patient.get("birthDate", "")
    if not birth_date_str:
        return "Unknown"
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - birth_date.year - (
            (today.month, today.day) < (birth_date.month, birth_date.day)
        )
        # Round down to nearest 5-year bracket
        lower = (age // 5) * 5
        return f"{lower}-{lower + 5}"
    except ValueError:
        return "Unknown"


# ---------------------------------------------------------------------------
# Low-level FHIR fetch — handles timeout + server status
# ---------------------------------------------------------------------------

def _fhir_get(url: str) -> tuple[dict | None, str]:
    """
    GET a FHIR resource/bundle. Returns (data_dict, status_message).
    status_message is 'ok' on success, or a human-readable error on failure.
    """
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS, headers={"Accept": "application/fhir+json"})
        response.raise_for_status()
        return response.json(), "ok"
    except requests.exceptions.Timeout:
        return None, f"FHIR server timeout after {TIMEOUT_SECONDS}s — server may be slow or down"
    except requests.exceptions.ConnectionError:
        return None, "FHIR server unreachable — check network or server status"
    except requests.exceptions.HTTPError as e:
        return None, f"FHIR server HTTP error: {e.response.status_code}"
    except Exception as e:
        return None, f"Unexpected FHIR error: {str(e)}"


# ---------------------------------------------------------------------------
# Offline fallback — load from cached samples
# ---------------------------------------------------------------------------

def _load_offline_sample(patient_id: str) -> dict:
    """
    Load cached FHIR responses from data/fhir_samples/ for offline development.
    Falls back to patient-001 samples if patient_id-specific files don't exist.
    """
    # Try patient-specific file first, then fall back to the generic sample
    suffix = patient_id if (FHIR_SAMPLES_DIR / f"patient_{patient_id}.json").exists() else "001"

    def _load(filename):
        path = FHIR_SAMPLES_DIR / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    patient   = _load(f"patient_{suffix}.json")
    conditions = _load(f"conditions_{suffix}.json")
    medications = _load(f"medications_{suffix}.json")
    coverage  = _load(f"coverage_{suffix}.json")

    conditions_list  = _extract_conditions(conditions)
    medications_list = _extract_medications(medications)
    coverage_type    = _extract_coverage_type(coverage)
    age_range        = _extract_age_range(patient)

    return {
        "conditions": conditions_list,
        "medications": medications_list,
        "coverage_type": coverage_type,
        "age_range": age_range,
        "source": "offline_cache",
        "server_status": "offline — using cached sample data",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_patient_context(patient_id: str) -> dict:
    """
    Fetch patient data from HAPI FHIR and return ONLY clinical context.

    PHI DROPPED (per docs/02_security_compliance.md):
      - name, DOB (converted to age range), SSN, address, phone, MRN

    PHI KEPT / RETURNED (clinical context for policy lookup):
      - conditions (diagnosis names)
      - medications (drug names)
      - coverage_type (e.g. 'Medicare Part A')
      - age_range  (e.g. '65-70')

    Returns:
        dict with keys: conditions, medications, coverage_type, age_range,
                        server_status, source
    """
    base = HAPI_FHIR_BASE_URL.rstrip("/")

    # --- Fetch all four resources ---
    patient_data,    patient_status    = _fhir_get(f"{base}/Patient/{patient_id}")
    conditions_data, conditions_status = _fhir_get(f"{base}/Condition?patient={patient_id}")
    medications_data,medications_status= _fhir_get(f"{base}/MedicationRequest?patient={patient_id}")
    coverage_data,   coverage_status   = _fhir_get(f"{base}/Coverage?patient={patient_id}")

    # --- Detect server failure (E-10: FHIR server down) ---
    server_ok = patient_status == "ok"

    if not server_ok:
        # Return offline cache plus a clear status message for the pipeline
        result = _load_offline_sample(patient_id)
        result["server_status"] = patient_status   # e.g. "FHIR server timeout after 10s"
        result["source"] = "offline_cache"
        return result

    # --- Extract ONLY clinical context — all PHI dropped here ---
    conditions  = _extract_conditions(conditions_data  or {})
    medications = _extract_medications(medications_data or {})
    coverage_type = _extract_coverage_type(coverage_data or {})
    age_range   = _extract_age_range(patient_data or {})

    return {
        "conditions":    conditions,
        "medications":   medications,
        "coverage_type": coverage_type,
        "age_range":     age_range,
        "source":        "live_fhir",
        "server_status": "ok",
        # PHI fields (name, DOB, SSN, address, MRN) are intentionally absent
    }


# ==============================================================================
# Testing Block (Comment out later)
# ==============================================================================
if __name__ == "__main__":
    print("--- FHIR Client Tests ---\n")

    # Test 1: Offline fallback using cached sample (patient-001)
    print("[Test 1] Offline cache — patient-001")
    result_offline = _load_offline_sample("001")
    print(f"  conditions   : {result_offline['conditions']}")
    print(f"  medications  : {result_offline['medications']}")
    print(f"  coverage_type: {result_offline['coverage_type']}")
    print(f"  age_range    : {result_offline['age_range']}")
    print(f"  source       : {result_offline['source']}")

    # Test 2: Verify PHI is NOT in the returned dict
    print("\n[Test 2] PHI check — no name/DOB/SSN/address in output")
    phi_keys = {"name", "birthDate", "ssn", "address", "telecom", "mrn"}
    leaked = phi_keys.intersection(result_offline.keys())
    if leaked:
        print(f"  FAIL — PHI found in output: {leaked}")
    else:
        print(f"  PASS — No PHI keys in returned dict")

    # Test 3: Age range conversion (not exact DOB)
    print("\n[Test 3] Age range conversion")
    mock_patient = {"birthDate": "1955-03-15"}
    age_range = _extract_age_range(mock_patient)
    print(f"  DOB: 1955-03-15 → age_range: {age_range} (should be a 5-yr bracket)")
    assert "-" in age_range and age_range != "Unknown", "Age range format unexpected"
    print(f"  PASS")

    # Test 4: Live FHIR call (E-10 — timeout/server down handled gracefully)
    print("\n[Test 4] Live FHIR fetch — patient 'example' (public HAPI test patient)")
    result_live = fetch_patient_context("example")
    print(f"  server_status: {result_live['server_status']}")
    print(f"  source       : {result_live['source']}")
    print(f"  conditions   : {result_live['conditions']}")
    print(f"  medications  : {result_live['medications']}")
    print(f"  coverage_type: {result_live['coverage_type']}")
    print(f"  age_range    : {result_live['age_range']}")

    # Test 5: Verify E-10 — simulate timeout by patching the module global directly
    print("\n[Test 5] E-10 — Simulated server down (bad URL)")
    import sys
    _real_url = HAPI_FHIR_BASE_URL
    # Patch the module-level global in the current module's namespace
    current_module = sys.modules[__name__]
    current_module.HAPI_FHIR_BASE_URL = "http://localhost:9999"
    result_down = fetch_patient_context("patient-001")
    current_module.HAPI_FHIR_BASE_URL = _real_url   # Restore
    print(f"  server_status: {result_down['server_status']}")
    print(f"  source       : {result_down['source']}")
    assert result_down["source"] == "offline_cache", "Should fall back to cache"
    print(f"  PASS — gracefully fell back to offline cache")
