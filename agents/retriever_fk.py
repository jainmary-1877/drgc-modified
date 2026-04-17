"""
Schema Linker Agent (Selector): Identifies relevant tables and columns.
Includes keyword pre-filtering and FK-based table expansion.
"""

from typing import List, Set
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from core.state import AgentState
from core.database import db_manager
from config import settings

# At the top of retriever_fk.py, after imports

SCHEMA_ANNOTATIONS = {
    "fb_translation_json": """
-- CRITICAL: fb_translation_json.translations is a JSONB ARRAY. Structure:
-- [{"language": "eng", "attribute": "NAME", "entityType": "QUESTION|PAGE|FORM",
--   "elementId": "uuid", "translatedText": "label text"}]
-- MANDATORY query pattern:
--   FROM fb_translation_json tj,
--        jsonb_array_elements(tj.translations) AS elem
--   WHERE elem->>'language' = 'eng'
--     AND elem->>'attribute' = 'NAME'
--     AND elem->>'entityType' = 'QUESTION'
-- NEVER use ->> directly on translations column without jsonb_array_elements first.
""",
"fb_forms": """
-- fb_forms columns: id, name, status, created_on, translations_id
-- translations_id is FK → fb_translation_json.id
-- Join pattern: JOIN fb_translation_json tj ON f.translations_id = tj.id
-- fb_forms.name stores the form name directly (no JSONB needed for form name)
-- CRITICAL: status values are ALWAYS UPPERCASE STRINGS
-- CORRECT:   WHERE status = 'PUBLISHED'
-- CORRECT:   WHERE status = 'DRAFT'
-- WRONG:     WHERE status = 'published'
-- WRONG:     WHERE status = 'draft'
""",
    "fb_modules": """
-- fb_modules columns: id, name
-- Simple table — SELECT id, name FROM fb_modules
""",
    "fb_question": """
-- fb_question has NO name/label/title column.
-- Question labels are in fb_translation_json.translations JSONB only.
""",
    "fb_page": """
-- fb_page has NO name/label/title column.
-- Page labels are in fb_translation_json.translations JSONB only.
"""
}

MANDATORY_SQL_PATTERNS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY SQL PATTERNS — use these exactly:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

-- List all modules:
SELECT id, name FROM fb_modules ORDER BY name LIMIT 100;

-- Count ALL questions across all forms:
SELECT COUNT(*) AS question_count
FROM fb_translation_json,
     jsonb_array_elements(translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION';

-- STATUS VALUES ARE CASE-SENSITIVE AND ALWAYS UPPERCASE:
-- 'DRAFT' | 'PUBLISHED' | 'CANCELLED' | 'DELETED' | 'CONFLICT'
-- Form with MOST questions:
SELECT f.name AS form_name, COUNT(*) AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
GROUP BY f.name
ORDER BY question_count DESC
LIMIT 1;

-- Count questions in a specific form:
SELECT COUNT(*) AS question_count
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE f.name ILIKE '%FORM_NAME%'
  AND elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION';

-- List question labels for a specific form:
SELECT elem->>'translatedText' AS label
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE f.name ILIKE '%FORM_NAME%'
  AND elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'QUESTION'
LIMIT 100;

-- List all forms with English names:
SELECT f.id, elem->>'translatedText' AS form_name
FROM fb_forms f
JOIN fb_translation_json tj ON f.translations_id = tj.id,
     jsonb_array_elements(tj.translations) AS elem
WHERE elem->>'language' = 'eng'
  AND elem->>'attribute' = 'NAME'
  AND elem->>'entityType' = 'FORM'
LIMIT 100;
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
class SchemaLinkerAgent:
    """
    Performs schema pruning to reduce context noise.
    Uses keyword pre-filter + FK expansion + LLM to select relevant tables.
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.ollama_model_fast,
            temperature=0,
            base_url=settings.ollama_base_url
        )

        self.table_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database schema expert. Identify which tables are relevant for the user's question.

Instructions:
1. Analyze the question and logical plan
2. Select ONLY tables that are strictly necessary
3. Be conservative - include a table only if clearly needed
4. Return ONLY a comma-separated list of table names (no explanations)

Example:
Question: "What is the average order value by customer segment?"
Available Tables: customers, orders, products, invoices, shipments, employees
Response: customers, orders"""),
            ("user", """Question: {question}

Plan: {plan}

Available Tables: {all_tables}

Return comma-separated table names only:""")
        ])

    def _keyword_prefilter(self, question: str, plan: str, all_tables: List[str], max_candidates: int = 20) -> List[str]:
        """Pre-filter tables by keyword matching to avoid overwhelming the LLM."""
        question_lower = (question + " " + plan).lower()

        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "shall", "can",
                     "all", "show", "me", "get", "list", "find", "what", "which",
                     "how", "many", "much", "from", "where", "and", "or", "in",
                     "of", "to", "for", "with", "by", "on", "at", "as", "into"}

        question_words = set(question_lower.replace(",", " ").split()) - stopwords

        scored = []
        for table in all_tables:
            table_lower = table.lower()
            table_parts = set(table_lower.split("_"))

            score = 0
            for word in question_words:
                if word in table_lower:
                    score += 3
                for part in table_parts:
                    if len(part) > 2 and (word in part or part in word):
                        score += 1

            scored.append((table, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = [t for t, s in scored if s > 0][:max_candidates]

        if len(candidates) < 5:
            candidates = [t for t, s in scored[:max_candidates]]

        logger.info(f"Keyword pre-filter: {len(all_tables)} → {len(candidates)} candidate tables")
        return candidates

    def _expand_with_fk_tables(self, selected_tables: List[str], all_tables: List[str], max_depth: int = 1) -> List[str]:
        """
        Expand selected tables by following foreign key relationships.
        This ensures related tables (like fb_translation_json) are included
        even if they don't match keywords in the question.
        """
        expanded: Set[str] = set(selected_tables)

        for table in list(selected_tables):
            try:
                metadata = db_manager.get_table_metadata(table)
                fks = metadata.get("foreign_keys", [])

                for fk in fks:
                    referred_table = fk.get("referred_table")
                    if referred_table and referred_table in all_tables:
                        if referred_table not in expanded:
                            logger.info(f"FK expansion: {table} → {referred_table}")
                            expanded.add(referred_table)
            except Exception as e:
                logger.warning(f"Could not expand FKs for {table}: {e}")

        result = list(expanded)
        logger.info(f"FK expansion: {len(selected_tables)} → {len(result)} tables")
        return result

    def select_tables(self, question: str, plan: str, all_tables: List[str]) -> List[str]:
        """Select relevant tables using keyword pre-filter + LLM + FK expansion."""
        try:
            # ── NEW: domain overrides for JSONB schema ──────────────────
            DOMAIN_OVERRIDES = [
                ({"form", "question"},  ["fb_forms", "fb_translation_json"]),
                ({"form", "name"},      ["fb_forms", "fb_translation_json"]),
                ({"form", "element"},   ["fb_forms", "fb_translation_json"]),
                ({"module"},            ["fb_modules"]),
            ]
            question_words = set(question.lower().split())
            forced_tables = []
            for keywords, tables in DOMAIN_OVERRIDES:
                if keywords.issubset(question_words):
                    forced_tables.extend(tables)
                    logger.info(f"Domain override triggered for keywords {keywords}: adding {tables}")

            # Step 1: keyword pre-filter
            candidates = self._keyword_prefilter(question, plan, all_tables)

            # Merge forced tables in
            for t in forced_tables:
                if t in all_tables and t not in candidates:
                    candidates.append(t)
            # ────────────────────────────────────────────────────────────

            # Step 2: LLM picks from filtered candidates
            chain = self.table_selection_prompt | self.llm
            response = chain.invoke({
                "question": question,
                "plan": plan,
                "all_tables": ", ".join(candidates)
            })

            selected = [t.strip() for t in response.content.split(",")]
            selected = [t for t in selected if t in all_tables]

            # Ensure forced tables survive LLM pruning
            for t in forced_tables:       # ← NEW
                if t in all_tables and t not in selected:
                    selected.append(t)

            if not selected:
                selected = candidates[:5]

            logger.info(f"LLM selected {len(selected)} tables: {', '.join(selected)}")

            # Step 3: expand with FK-related tables
            expanded = self._expand_with_fk_tables(selected, all_tables)
            return expanded

        except Exception as e:
            logger.error(f"Table selection error: {e}")
            return self._keyword_prefilter(question, plan, all_tables, max_candidates=5)


    def retrieve_schema(self, state: AgentState) -> dict:
        """Retrieve and prune schema information to only relevant tables."""
        logger.info("SCHEMA LINKER: Retrieving relevant tables and schema")

        question = state["question"]
        plan = state.get("plan", "")

        try:
            all_tables = db_manager.get_all_table_names()
            logger.info(f"Database has {len(all_tables)} tables")

            if plan:
                selected_tables = self.select_tables(question, plan, all_tables)
            else:
                selected_tables = self._keyword_prefilter(question, "", all_tables, max_candidates=5)

            schema_context = db_manager.get_schema_for_tables(selected_tables)
            annotations = []
            #temporary
            logger.debug(f"ANNOTATIONS INJECTED: {list(SCHEMA_ANNOTATIONS.keys())}")
            logger.debug(f"fb_translation_json in selected: {'fb_translation_json' in selected_tables}")
            for table in selected_tables:
                if table in SCHEMA_ANNOTATIONS:
                    annotations.append(SCHEMA_ANNOTATIONS[table])

            if annotations:
                schema_context = "\n".join(annotations) + "\n" + schema_context

            # Always inject mandatory patterns so LLM knows the correct queries
            schema_context = schema_context + "\n" + MANDATORY_SQL_PATTERNS
            # ────────────────────────────────────────────────────────────

            schema_metadata = {}
            for table in selected_tables:
                metadata = db_manager.get_table_metadata(table)
                schema_metadata[table] = metadata

            logger.info(f"Final selected {len(selected_tables)} tables: {', '.join(selected_tables)}")

            return {
                "relevant_tables": selected_tables,
                "schema_context": schema_context,
                "schema_metadata": schema_metadata
            }

        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            return {
                "error": f"Schema retrieval failed: {str(e)}",
                "should_retry": False
            }

def schema_linker_node(state: AgentState) -> dict:
    """LangGraph node wrapper for SchemaLinkerAgent."""
    agent = SchemaLinkerAgent()
    return agent.retrieve_schema(state)