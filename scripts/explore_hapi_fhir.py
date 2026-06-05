"""
Phase 2 of FHIR discovery — try known good patient IDs from the HAPI R4 sandbox
that are documented to have conditions, medications, and coverage linked to them.

Sources:
  - HAPI's own test data: IDs like 'example', '592776', '1837602', etc.
  - Synthea-generated patients that HAPI hosts
"""

import requests

BASE = "https://hapi.fhir.org/baseR4"
TIMEOUT = 10
HEADERS = {"Accept": "application/fhir+json"}

# Known candidate IDs to try on HAPI public R4 sandbox
# These are well-documented in HAPI's own examples and Synthea uploads
CANDIDATE_IDS = [
    "example",          # FHIR spec example patient
    "pat1",             # FHIR spec example
    "pat2",             # FHIR spec example
    "1",
    "592776",
    "1837602",
    "2694101",          # Commonly populated Synthea patient on public HAPI
    "17976",
    "smart-1288992",    # SMART on FHIR sandbox patient
    "patient-example",
]


def get_json(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def resource_count(resource_type, patient_id):
    data, err = get_json(
        f"{BASE}/{resource_type}",
        params={"patient": patient_id, "_summary": "count"}
    )
    if err or data is None:
        return 0
    return data.get("total", 0)


print("=" * 60)
print("  HAPI R4 — Known Patient ID Probe")
print("=" * 60)
print(f"\n  {'ID':<25} {'Conditions':>12} {'Medications':>13} {'Coverage':>10}")
print("  " + "-" * 65)

good = []
for pid in CANDIDATE_IDS:
    cond  = resource_count("Condition", pid)
    meds  = resource_count("MedicationRequest", pid)
    cov   = resource_count("Coverage", pid)
    flag = "  ← USABLE" if (cond + meds) > 0 else ""
    print(f"  {pid:<25} {cond:>12} {meds:>13} {cov:>10}{flag}")
    if (cond + meds) > 0:
        good.append((pid, cond, meds, cov))

print()
if good:
    best = sorted(good, key=lambda x: x[1] + x[2], reverse=True)[0]
    print(f"[RESULT] Best patient_id to use: '{best[0]}'")
    print(f"         Conditions={best[1]}, Medications={best[2]}, Coverage={best[3]}")
else:
    print("[RESULT] None of the candidate IDs had linked data.")
    print("         The public HAPI sandbox may have been reset.")
    print("         → Use the offline sample cache. It's reliable.")
