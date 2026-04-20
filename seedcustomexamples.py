"""
Seed domain-specific SQL examples into ChromaDB.
Built using schema_introspector.py knowledge — enum values, FK paths, business rules.
Run: python3 seedcustomexamples.py --force
Add new failures here and reseed whenever agent gets a query wrong.
"""

from tools.vector_store import few_shot_retriever
from core.database import db_manager

examples = [

    # ═══════════════════════════════════════════════════════════
    # FORM BUILDER — fb_forms
    # ═══════════════════════════════════════════════════════════
    {
        "question": "how many forms are there",
        "sql": "SELECT COUNT(*) AS total_forms FROM fb_forms;",
        "explanation": "Simple count, no JSONB needed",
        "complexity": "simple"
    },
    {
        "question": "list all forms",
        "sql": "SELECT id, name, status, active FROM fb_forms ORDER BY name;",
        "explanation": "Direct select from fb_forms",
        "complexity": "simple"
    },
    {
        "question": "how many forms are published",
        "sql": "SELECT COUNT(*) AS published_form_count FROM fb_forms WHERE status = 'PUBLISHED';",
        "explanation": "Status values are always UPPERCASE",
        "complexity": "simple"
    },
    {
        "question": "how many forms are not published",
        "sql": "SELECT COUNT(*) AS non_published_form_count FROM fb_forms WHERE status != 'PUBLISHED';",
        "explanation": "Use != PUBLISHED for negation, never list individual statuses",
        "complexity": "simple"
    },
    {
        "question": "list all forms that are not published",
        "sql": """SELECT id, name, status
FROM fb_forms
WHERE status != 'PUBLISHED'
ORDER BY status, name;""",
        "explanation": "Use != for negation not OR with individual values",
        "complexity": "simple"
    },
    {
        "question": "list forms by status",
        "sql": """SELECT status, COUNT(*) AS form_count
FROM fb_forms
GROUP BY status
ORDER BY form_count DESC;""",
        "explanation": "Group by status to get counts per status",
        "complexity": "simple"
    },
    {
        "question": "list all active forms",
        "sql": "SELECT id, name FROM fb_forms WHERE active = true ORDER BY name;",
        "explanation": "active is boolean, use = true not = 'true'",
        "complexity": "simple"
    },
    {
        "question": "list all inactive forms",
        "sql": "SELECT id, name FROM fb_forms WHERE active = false ORDER BY name;",
        "explanation": "active is boolean, use = false",
        "complexity": "simple"
    },
    {
        "question": "list all draft forms",
        "sql": "SELECT id, name FROM fb_forms WHERE status = 'DRAFT' ORDER BY name;",
        "explanation": "Status is always UPPERCASE: DRAFT, PUBLISHED, CANCELLED, DELETED, CONFLICT",
        "complexity": "simple"
    },
    {
        "question": "list all cancelled forms",
        "sql": "SELECT id, name FROM fb_forms WHERE status = 'CANCELLED' ORDER BY name;",
        "explanation": "Status is always UPPERCASE",
        "complexity": "simple"
    },

    # ═══════════════════════════════════════════════════════════
    # FORM BUILDER — QUESTIONS (JSONB pattern)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "count all questions across all forms",
        "sql": """SELECT COUNT(*) AS question_count
FROM fb_translation_json,
     jsonb_array_elements(translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION';""",
        "explanation": "Cross join jsonb_array_elements on fb_translation_json",
        "complexity": "medium"
    },
    {
        "question": "list the form with the most number of questions",
        "sql": """SELECT f.name AS form_name, COUNT(*) AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count DESC
LIMIT 1;""",
        "explanation": "JSONB unpack to count questions per form, order descending",
        "complexity": "complex"
    },
    {
        "question": "which form has the least number of questions",
        "sql": """SELECT f.name AS form_name, COUNT(*) AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count ASC
LIMIT 1;""",
        "explanation": "Same JSONB pattern but ORDER BY ASC for minimum",
        "complexity": "complex"
    },
    {
        "question": "how many questions does each form have",
        "sql": """SELECT f.name AS form_name, COUNT(*) AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count DESC;""",
        "explanation": "JSONB unpack to count questions grouped by form",
        "complexity": "complex"
    },
    {
        "question": "list all question labels in a specific form",
        "sql": """SELECT elem->>'translatedText' AS label
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE f.name ILIKE '%FORM_NAME%'
  AND elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
ORDER BY label;""",
        "explanation": "Always unpack JSONB with jsonb_array_elements, use ILIKE",
        "complexity": "complex"
    },
    {
        "question": "list forms with more than 5 questions",
        "sql": """SELECT f.name AS form_name, COUNT(*) AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
HAVING COUNT(*) > 5
ORDER BY question_count DESC;""",
        "explanation": "Use HAVING for post-aggregation filter on question count",
        "complexity": "complex"
    },

    # ═══════════════════════════════════════════════════════════
    # MODULES
    # ═══════════════════════════════════════════════════════════
    {
        "question": "list all modules",
        "sql": "SELECT id, name FROM fb_modules ORDER BY name;",
        "explanation": "Simple select from fb_modules, no JSONB",
        "complexity": "simple"
    },
    {
        "question": "how many modules are there",
        "sql": "SELECT COUNT(*) AS module_count FROM fb_modules;",
        "explanation": "Simple count of modules",
        "complexity": "simple"
    },
    {
        "question": "list modules related to inspection",
        "sql": "SELECT id, name FROM fb_modules WHERE name ILIKE '%inspection%' ORDER BY name;",
        "explanation": "Always use ILIKE for text matching never LIKE",
        "complexity": "simple"
    },

    # ═══════════════════════════════════════════════════════════
    # INSPECTION REPORT — scores
    # From introspector: status enums = DRAFT, SUBMITTED, CLOSED,
    # UNDER_REVIEW, RETURN_FOR_MODIFICATION
    # Business rule: always exclude DRAFT when querying scores
    # ═══════════════════════════════════════════════════════════
    {
        "question": "what is the average inspection score",
        "sql": """SELECT AVG(inspection_score) AS avg_score
FROM inspection_report
WHERE status != 'DRAFT';""",
        "explanation": "Always exclude DRAFT when querying inspection_score",
        "complexity": "simple"
    },
    {
        "question": "show the last 10 inspection scores",
        "sql": """SELECT inspection_score, status, submitted_on
FROM inspection_report
WHERE status != 'DRAFT'
ORDER BY submitted_on DESC
LIMIT 10;""",
        "explanation": "Exclude DRAFT, order by submitted_on DESC",
        "complexity": "simple"
    },
    {
        "question": "average inspection score by type",
        "sql": """SELECT it.name AS type_name, AVG(ir.inspection_score) AS avg_score
FROM inspection_report ir
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE ir.status != 'DRAFT'
GROUP BY it.name
ORDER BY avg_score DESC;""",
        "explanation": "Join inspection_type for names, exclude DRAFT",
        "complexity": "medium"
    },
    {
        "question": "which inspector has the most inspection hours",
        "sql": """SELECT u.first_name || ' ' || u.last_name AS inspector_name,
       SUM(ir.total_inspection_hours) AS total_hours
FROM inspection_report ir
JOIN users u ON ir.inspector_user_id = u.id
WHERE ir.status != 'DRAFT'
GROUP BY u.first_name, u.last_name
ORDER BY total_hours DESC
LIMIT 1;""",
        "explanation": "Join users to resolve inspector UUID, never show raw UUID",
        "complexity": "medium"
    },
    {
        "question": "how many inspections per type",
        "sql": """SELECT it.name AS type_name, COUNT(*) AS inspection_count
FROM inspection_report ir
JOIN inspection_type it ON ir.inspection_type_id = it.id
GROUP BY it.name
ORDER BY inspection_count DESC;""",
        "explanation": "Join inspection_type for readable names",
        "complexity": "medium"
    },
    {
        "question": "list all completed inspections",
        "sql": """SELECT inspection_id, inspection_score, submitted_on
FROM inspection_report
WHERE status = 'CLOSED'
ORDER BY submitted_on DESC
LIMIT 100;""",
        "explanation": "completed/done/finished = status CLOSED per vocabulary rules",
        "complexity": "simple"
    },
    {
        "question": "list all inspections under review",
        "sql": """SELECT inspection_id, inspection_score, submitted_on
FROM inspection_report
WHERE status = 'UNDER_REVIEW'
ORDER BY submitted_on DESC
LIMIT 100;""",
        "explanation": "pending review/under review/awaiting review = UNDER_REVIEW",
        "complexity": "simple"
    },
    {
        "question": "list returned inspections",
        "sql": """SELECT inspection_id, inspection_score, submitted_on
FROM inspection_report
WHERE status = 'RETURN_FOR_MODIFICATION'
ORDER BY submitted_on DESC
LIMIT 100;""",
        "explanation": "returned/sent back/rejected = RETURN_FOR_MODIFICATION",
        "complexity": "simple"
    },
    {
        "question": "inspections at a specific facility",
        "sql": """SELECT ir.inspection_id, ir.inspection_score, ir.status, fac.name AS facility_name
FROM inspection_report ir
JOIN facility fac ON ir.facility_id = fac.id
WHERE fac.name ILIKE '%FACILITY_NAME%'
  AND ir.status != 'DRAFT'
ORDER BY ir.submitted_on DESC
LIMIT 100;""",
        "explanation": "Join facility to resolve UUID, use ILIKE, exclude DRAFT",
        "complexity": "medium"
    },
    {
        "question": "inspections per cycle",
        "sql": """SELECT ic.id, ic.start_date, ic.end_date, COUNT(ir.id) AS report_count
FROM inspection_cycle ic
LEFT JOIN inspection_report ir ON ir.cycle_id = ic.id
GROUP BY ic.id, ic.start_date, ic.end_date
ORDER BY ic.start_date DESC;""",
        "explanation": "LEFT JOIN to include cycles with zero inspections",
        "complexity": "medium"
    },
    {
        "question": "cycles for safety inspections",
        "sql": """SELECT ic.id, ic.start_date, ic.end_date, COUNT(ir.id) AS report_count
FROM inspection_cycle ic
LEFT JOIN inspection_report ir ON ir.cycle_id = ic.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE it.name ILIKE '%safety%'
GROUP BY ic.id, ic.start_date, ic.end_date
ORDER BY ic.start_date DESC;""",
        "explanation": "Type names are on inspection_type table, go through inspection_report",
        "complexity": "complex"
    },

    # ═══════════════════════════════════════════════════════════
    # CORRECTIVE ACTIONS
    # From introspector: status enums = OPEN, CLOSED, OVERDUE,
    # CLOSE_WITH_DEFERRED
    # Vocabulary: pending/outstanding = NOT IN (CLOSED, CLOSE_WITH_DEFERRED)
    # ═══════════════════════════════════════════════════════════
    {
        "question": "show all open corrective actions",
        "sql": """SELECT corrective_action_id, cause, corrective_action,
       responsible, status, progress_stage
FROM inspection_corrective_action
WHERE status = 'OPEN'
LIMIT 100;""",
        "explanation": "open = status OPEN exactly per vocabulary rules",
        "complexity": "simple"
    },
    {
        "question": "list all pending corrective actions",
        "sql": """SELECT corrective_action_id, cause, corrective_action,
       responsible, status, progress_stage
FROM inspection_corrective_action
WHERE status NOT IN ('CLOSED', 'CLOSE_WITH_DEFERRED')
LIMIT 100;""",
        "explanation": "pending/outstanding/unresolved = NOT IN CLOSED and CLOSE_WITH_DEFERRED",
        "complexity": "simple"
    },
    {
        "question": "list overdue corrective actions",
        "sql": """SELECT corrective_action_id, cause, responsible,
       target_close_out_date, age
FROM inspection_corrective_action
WHERE status = 'OVERDUE'
   OR (target_close_out_date < CURRENT_DATE AND completed_on IS NULL)
LIMIT 100;""",
        "explanation": "overdue = status OVERDUE or past target date and not completed",
        "complexity": "simple"
    },
    {
        "question": "total capex and opex for corrective actions",
        "sql": """SELECT SUM(capex) AS total_capex, SUM(opex) AS total_opex
FROM inspection_corrective_action;""",
        "explanation": "Simple SUM aggregation on cost columns",
        "complexity": "simple"
    },
    {
        "question": "show corrective actions with their causes",
        "sql": """SELECT corrective_action_id, cause, corrective_action,
       responsible, status
FROM inspection_corrective_action
WHERE status NOT IN ('CLOSED', 'CLOSE_WITH_DEFERRED')
LIMIT 100;""",
        "explanation": "Show cause column, exclude closed actions",
        "complexity": "simple"
    },
    {
        "question": "how many corrective actions are closed",
        "sql": """SELECT COUNT(*) AS closed_count
FROM inspection_corrective_action
WHERE status IN ('CLOSED', 'CLOSE_WITH_DEFERRED');""",
        "explanation": "closed = CLOSED or CLOSE_WITH_DEFERRED per vocabulary",
        "complexity": "simple"
    },
    {
        "question": "how many corrective actions are not closed",
        "sql": """SELECT COUNT(*) AS open_count
FROM inspection_corrective_action
WHERE status NOT IN ('CLOSED', 'CLOSE_WITH_DEFERRED');""",
        "explanation": "not closed = NOT IN both closed statuses",
        "complexity": "simple"
    },
    {
        "question": "corrective actions at a specific facility",
        "sql": """SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status, fac.name AS facility_name
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN facility fac ON ir.facility_id = fac.id
WHERE fac.name ILIKE '%FACILITY_NAME%'
LIMIT 100;""",
        "explanation": "Go through inspection_report to reach facility — it is the hub table",
        "complexity": "complex"
    },
    {
        "question": "corrective actions for safety inspections",
        "sql": """SELECT ica.corrective_action_id, ica.cause, ica.corrective_action,
       ica.responsible, ica.status, it.name AS inspection_type
FROM inspection_corrective_action ica
JOIN inspection_report ir ON ica.inspection_id = ir.id
JOIN inspection_type it ON ir.inspection_type_id = it.id
WHERE it.name ILIKE '%safety%'
LIMIT 100;""",
        "explanation": "Go through inspection_report hub to reach inspection_type",
        "complexity": "complex"
    },

    # ═══════════════════════════════════════════════════════════
    # INSPECTION SCHEDULE
    # From introspector: status enums = PENDING, ONGOING,
    # COMPLETED, OVERDUE, CANCELLED
    # ═══════════════════════════════════════════════════════════
    {
        "question": "inspection schedule for this month",
        "sql": """SELECT *
FROM inspection_schedule
WHERE schedule_date >= date_trunc('month', CURRENT_DATE)
  AND schedule_date < date_trunc('month', CURRENT_DATE) + INTERVAL '1 month'
LIMIT 100;""",
        "explanation": "Use date_trunc for month boundaries",
        "complexity": "medium"
    },
    {
        "question": "list all overdue scheduled inspections",
        "sql": """SELECT *
FROM inspection_schedule
WHERE status = 'OVERDUE'
LIMIT 100;""",
        "explanation": "Schedule status enum: PENDING, ONGOING, COMPLETED, OVERDUE, CANCELLED",
        "complexity": "simple"
    },
    {
        "question": "list all pending scheduled inspections",
        "sql": """SELECT *
FROM inspection_schedule
WHERE status = 'PENDING'
LIMIT 100;""",
        "explanation": "Schedule status enum is UPPERCASE",
        "complexity": "simple"
    },
]


# ═══════════════════════════════════════════════════════════════
# Validation — uses EXPLAIN to check SQL before seeding
# ═══════════════════════════════════════════════════════════════

def validate_examples(examples):
    """Dry-run each SQL via EXPLAIN before seeding."""
    valid = []
    invalid = []

    for ex in examples:
        # Skip examples with placeholder values like FORM_NAME, FACILITY_NAME
        if any(p in ex["sql"] for p in ["FORM_NAME", "FACILITY_NAME", "FACILITY_NAME%"]):
            valid.append(ex)
            print(f"⏭  SKIPPED (template): {ex['question']}")
            continue

        explain_sql = f"EXPLAIN {ex['sql']}"
        try:
            result, error, _ = db_manager.execute_query(explain_sql, timeout=10)
            if error:
                invalid.append({"question": ex["question"], "reason": error})
                print(f"❌ INVALID: {ex['question']}\n   Reason: {error}\n")
            else:
                valid.append(ex)
                print(f"✅ VALID:   {ex['question']}")
        except Exception as e:
            invalid.append({"question": ex["question"], "reason": str(e)})
            print(f"❌ ERROR:   {ex['question']}\n   Reason: {e}\n")

    print(f"\nSummary: {len(valid)} valid/skipped, {len(invalid)} invalid")
    return valid, invalid


def auto_seed():
    """Called on startup — seeds only if vector store is empty."""
    try:
        existing = few_shot_retriever.retrieve("test", k=1)
        if existing:
            return False
    except Exception:
        pass

    print("Vector store empty — auto-seeding...")
    valid_examples, _ = validate_examples(examples)
    few_shot_retriever.add_examples_batch(valid_examples)
    print(f"Auto-seeded {len(valid_examples)} examples.")
    return True


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    strict = "--strict" in sys.argv

    print("Validating examples against DB...\n")
    valid_examples, invalid_examples = validate_examples(examples)

    if invalid_examples and strict:
        print("Strict mode — aborting due to invalid examples.")
        sys.exit(1)

    if force:
        print("\nForce reseed — clearing old examples...")
        few_shot_retriever.clear()

    few_shot_retriever.add_examples_batch(valid_examples)
    print(f"\nSeeded {len(valid_examples)} examples successfully.")
    print("Done. Restart your agent to use updated examples.")