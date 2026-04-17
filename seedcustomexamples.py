"""
Seed custom few-shot examples specific to your database schema.
Run this ONCE after setup: python seed_custom_examples.py

This teaches the model your JSONB translation pattern so it generates
correct queries for your database structure.
"""

from tools import few_shot_retriever
from loguru import logger


# ─────────────────────────────────────────────────────────────
# ADD YOUR DATABASE-SPECIFIC EXAMPLES HERE
# These teach the model your JSONB translation pattern
# ─────────────────────────────────────────────────────────────

custom_examples = [
    {
        "question": "Which form has the most questions?",
        "sql": """SELECT f.name AS form_name, COUNT(elem->>'entityType') AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count DESC
LIMIT 1;""",
        "explanation": "Use jsonb_array_elements to expand the translations JSONB array, filter by language and entityType, then count and group by form name.",
        "complexity": "complex"
    },
    {
        "question": "Show all modules",
        "sql": """SELECT fb_modules.id,
                 fb_modules.name
          FROM   fb_modules;""",
        "explanation": "Simple select from fb_modules table.",
        "complexity": "simple"
    },
    {
        "question": "List all forms with their names in English",
        "sql": """SELECT f.id AS form_id, elem->>'value' AS form_name
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME';""",
        "explanation": "Join fb_forms with fb_translation_json and expand JSONB array to get English names.",
        "complexity": "medium"
    },
    {
        "question": "How many questions are in each form?",
        "sql": """SELECT f.name AS form_name, COUNT(elem->>'entityType') AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count DESC;""",
        "explanation": "Expand JSONB translations array and count QUESTION entityType per form.",
        "complexity": "medium"
    },
]

{
    "question": "which form has the most questions",
    "sql": """SELECT f.name AS form_name, COUNT(elem->>'entityType') AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count DESC
LIMIT 1;""",
    "explanation": "Expand JSONB translations array, filter entityType=QUESTION, count per form.",
    "complexity": "complex"
},
{
    "question": "form with maximum number of questions",
    "sql": """SELECT f.name AS form_name, COUNT(elem->>'entityType') AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count DESC
LIMIT 1;""",
    "explanation": "Same JSONB pattern, different question phrasing.",
    "complexity": "complex"
},
{
    "question": "list forms with their question counts",
    "sql": """SELECT f.name AS form_name, COUNT(elem->>'entityType') AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count DESC;""",
    "explanation": "All forms with question counts using JSONB expansion.",
    "complexity": "complex"
}
def main():
    logger.info("Seeding custom examples into vector store...")
    few_shot_retriever.add_examples_batch(custom_examples)
    logger.info(f"✓ Successfully seeded {len(custom_examples)} custom examples")
    logger.info("The model will now use these as reference for similar questions.")


if __name__ == "__main__":
    main()