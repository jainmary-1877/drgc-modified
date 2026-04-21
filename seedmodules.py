"""
Seed fb_module names into ChromaDB for semantic search.
IDs are fetched live from DB by matching module names.
Run: python3 seedmodules.py
"""

from tools.vector_store import module_retriever
from core.database import db_manager

# Clean module names only — test/duplicate entries excluded
CLEAN_MODULE_NAMES = [
    "Audit Portfolio Approval",
    "Audit Portfolio Rejection",
    "Inspection Portfolio Approval",
    "Inspection Portfolio Rejection",
    "Audit Schedule Approval",
    "Audit Schedule Rejection",
    "Audit Reschedule Request",
    "Audit Form",
    "Audit Corrective Action Close With Deferred",
    "Audit Corrective Action Completion",
    "Audit Corrective Action Adequacy Rejection",
    "Audit Corrective Action Implementation Approval",
    "Audit Corrective Action Adequacy Approval",
    "Audit Corrective Action Progress Tracking",
    "Audit Corrective Action Implementation Rejection",
    "Audit Corrective Action Entry",
    "Inspection Mitigative Action Completion",
    "Inspection Report Approval",
    "Inspection Mitigative Action Rejection",
    "Inspection Report Rejection",
    "Inspection Mitigative Action Entry",
    "Inspection Mitigative Action PR Entry",
    "Inspection Mitigative Action Close with Deferred",
    "Inspection Report Final Approval Before Closeout",
    "Inspection Form",
    "Inspection Report Remarks Entry",
    "Audit Report Remarks Entry",
    "Inspection Corrective Action Progress Tracking",
    "Audit Plan Approval",
    "Audit Plan Rejection",
    "Audit Plan Remarks",
    "Inspection Schedule Cancellation Request",
    "Inspection Schedule Cancellation Rejection",
    "Inspection Schedule Cancellation Approval",
    "Audit Report Closeout",
    "Inspection Mitigative Action Client Details Entry",
    "first party - acc",
    "Statutory FLS Audit",
    "QHSE Audit",
    "QHSE Internal audit",
    "QHSE Internal audit FTZ",
    "Internal Accomodation",
    "External Specialist",
    "Statutory - Specialist",
    "external-stores",
    "Quality Audit",
    "FM Audit",
    "Second Party - Sustainability",
    "Internal-Subcontractors",
]


def fetch_module_ids(names: list) -> list:
    """
    Fetch module IDs from DB by matching names.
    Returns list of {id, name} dicts.
    """
    found = []
    not_found = []

    for name in names:
        sql = f"SELECT id, name FROM fb_modules WHERE name ILIKE '{name}' LIMIT 1;"
        result, error, _ = db_manager.execute_query(sql, timeout=10)

        if error or not result:
            not_found.append(name)
            continue

        row = dict(result[0]._mapping)
        found.append({"id": str(row["id"]), "name": row["name"]})

    if not_found:
        print(f"\n⚠ Not found in DB ({len(not_found)}):")
        for n in not_found:
            print(f"   - {n}")

    return found


def auto_seed_modules():
    """Called on startup — seeds only if module collection is empty."""
    try:
        existing = module_retriever.search("test", k=1)
        if existing:
            return False
    except Exception:
        pass

    print("Module collection empty — auto-seeding...")
    modules = fetch_module_ids(CLEAN_MODULE_NAMES)
    module_retriever.clear()
    module_retriever.add_modules(modules)
    print(f"Auto-seeded {len(modules)} modules.")
    return True


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv

    print("Fetching module IDs from DB...\n")
    modules = fetch_module_ids(CLEAN_MODULE_NAMES)

    print(f"\nFound {len(modules)} modules in DB.")

    if force:
        print("Clearing old module collection...")
        module_retriever.clear()

    module_retriever.add_modules(modules)
    print(f"\nSeeded {len(modules)} modules successfully.")
    print("Done.")