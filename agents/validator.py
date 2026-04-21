"""
SQL Validator: Lightweight post-generation check before execution.
Catches common LLM mistakes without needing a DB round-trip.
"""

import re
from loguru import logger
from core.state import AgentState


VALIDATION_RULES = [
    {
        "name": "hardcoded_negation",
        "pattern": r"status\s*=\s*'(DRAFT|CANCELLED|DELETED|CONFLICT)'",
        "message": "Use status != 'PUBLISHED' instead of listing individual non-published statuses",
        "suggestion": lambda sql: re.sub(
            r"WHERE\s+.*status\s*=\s*'(DRAFT|CANCELLED|DELETED|CONFLICT)'.*?(OR\s+status\s*=\s*'\w+')*",
            "WHERE status != 'PUBLISHED'",
            sql, flags=re.IGNORECASE
        )
    },
    {
        "name": "lowercase_status",
        "pattern": r"status\s*[!=<>]+\s*'(published|draft|cancelled|deleted|conflict|open|closed|overdue)'",
        "message": "Status values must be UPPERCASE",
        "suggestion": lambda sql: re.sub(
            r"(status\s*[!=<>]+\s*)'(\w+)'",
            lambda m: m.group(0).replace(m.group(2), m.group(2).upper()),
            sql, flags=re.IGNORECASE
        )
    },
    {
        "name": "like_instead_of_ilike",
        "pattern": r"\bLIKE\b",
        "message": "Use ILIKE instead of LIKE for case-insensitive matching",
        "suggestion": lambda sql: re.sub(r'\bLIKE\b', 'ILIKE', sql, flags=re.IGNORECASE)
    },
    {
        "name": "missing_jsonb_unpack",
        "pattern": r"fb_translation_json(?!.*jsonb_array_elements)",
        "message": "fb_translation_json must use jsonb_array_elements() to unpack translations",
        "suggestion": None
    },
    {
        "name": "select_star",
        "pattern": r"SELECT\s+\*",
        "message": "Avoid SELECT * — specify columns explicitly",
        "suggestion": None
    },
    {
    "name": "wrong_date_column_report_date",
    "pattern": r"\breport_date\b",
    "message": "report_date does not exist — use submitted_on",
    "suggestion": lambda sql: re.sub(r'\breport_date\b', 'submitted_on', sql)
    },
    {
        "name": "wrong_date_column_inspection_date",
        "pattern": r"\binspection_date\b",
        "message": "inspection_date does not exist — use submitted_on or start_date_time",
        "suggestion": lambda sql: re.sub(r'\binspection_date\b', 'submitted_on', sql)
    },
    {
        "name": "wrong_date_column_created_date",
        "pattern": r"\bcreated_date\b",
        "message": "created_date does not exist — use created_on",
        "suggestion": lambda sql: re.sub(r'\bcreated_date\b', 'created_on', sql)
    },
    {
        "name": "deleted_not_filtered",
        "pattern": r"FROM inspection_report(?!.*deleted\s*=\s*false)",
        "message": "Consider filtering deleted = false for inspection_report queries",
        "suggestion": None
    },
    {
        "name": "hardcoded_negation",
        "pattern": r"status\s*=\s*'(DRAFT|CANCELLED|DELETED|CONFLICT)'",
        "message": "Use status != 'PUBLISHED' instead of listing individual non-published statuses",
        "suggestion": lambda sql: re.sub(
            r"WHERE\s+.*status\s*=\s*'(DRAFT|CANCELLED|DELETED|CONFLICT)'.*?(OR\s+status\s*=\s*'\w+')*",
            "WHERE status != 'PUBLISHED'",
            sql, flags=re.IGNORECASE
        )
    },
    {
        "name": "lowercase_status",
        "pattern": r"status\s*[!=<>]+\s*'(published|draft|cancelled|deleted|conflict|open|closed|overdue)'",
        "message": "Status values must be UPPERCASE",
        "suggestion": lambda sql: re.sub(
            r"(status\s*[!=<>]+\s*)'(\w+)'",
            lambda m: m.group(0).replace(m.group(2), m.group(2).upper()),
            sql, flags=re.IGNORECASE
        )
    },
    {
        "name": "like_instead_of_ilike",
        "pattern": r"\bLIKE\b",
        "message": "Use ILIKE instead of LIKE for case-insensitive matching",
        "suggestion": lambda sql: re.sub(r'\bLIKE\b', 'ILIKE', sql, flags=re.IGNORECASE)
    },
    {
        "name": "missing_jsonb_unpack",
        "pattern": r"fb_translation_json(?!.*jsonb_array_elements)",
        "message": "fb_translation_json must use jsonb_array_elements() to unpack translations",
        "suggestion": None
    },
    {
        "name": "select_star",
        "pattern": r"SELECT\s+\*",
        "message": "Avoid SELECT * — specify columns explicitly",
        "suggestion": None
    },
    # ── Date column fixes ────────────────────────────────────
    {
        "name": "wrong_date_column_report_date",
        "pattern": r"\breport_date\b",
        "message": "report_date does not exist — use submitted_on",
        "suggestion": lambda sql: re.sub(r'\breport_date\b', 'submitted_on', sql)
    },
    {
        "name": "wrong_date_column_inspection_date",
        "pattern": r"\binspection_date\b",
        "message": "inspection_date does not exist — use submitted_on or start_date_time",
        "suggestion": lambda sql: re.sub(r'\binspection_date\b', 'submitted_on', sql)
    },
    {
        "name": "wrong_date_column_created_date",
        "pattern": r"\bcreated_date\b",
        "message": "created_date does not exist — use created_on",
        "suggestion": lambda sql: re.sub(r'\bcreated_date\b', 'created_on', sql)
    },
    # ── Score column fixes ───────────────────────────────────
    {
        "name": "wrong_score_column_rating",
        "pattern": r"\bir\.rating\b|\brating\b",
        "message": "rating does not exist — use inspection_score or gp_score",
        "suggestion": lambda sql: re.sub(
            r'\bir\.rating\b|\brating\b', 'ir.inspection_score', sql
        )
    },
    {
    "name": "wrong_score_column_score",
    "pattern": r"\bir\.score\b|\bAVG\(score\)|\bSUM\(score\)|\bir\.score\s",
    "message": "ir.score does not exist — use ir.inspection_score or ir.gp_score",
    "suggestion": lambda sql: re.sub(
        r'\bir\.score\b', 'ir.inspection_score', sql
    )
    },
    {
        "name": "wrong_score_column_rating",
        "pattern": r"\bir\.rating\b|\bAVG\(ir\.rating\)|\bAVG\(rating\)",
        "message": "rating does not exist — use inspection_score or gp_score",
        "suggestion": lambda sql: re.sub(
            r'\bir\.rating\b|\brating\b', 'ir.inspection_score', sql
        )
    },
    {
    "name": "wrong_table_clients",
    "pattern": r"\bclients\b",
    "message": "Table 'clients' does not exist — use 'client'",
    "suggestion": lambda sql: re.sub(r'\bclients\b', 'client', sql)
    },
]


class SQLValidator:
    """Validates generated SQL before execution."""

    def validate(self, state: AgentState) -> dict:
        sql = state.get("sql_query", "")
        if not sql:
            return {}

        for rule in VALIDATION_RULES:
            if re.search(rule["pattern"], sql, re.IGNORECASE | re.DOTALL):
                logger.warning(f"SQL Validator caught: {rule['name']} — {rule['message']}")

                if rule["suggestion"]:
                    try:
                        fixed_sql = rule["suggestion"](sql)
                        logger.info(f"Auto-fixed: {rule['name']}")
                        logger.debug(f"Fixed SQL: {fixed_sql}")
                        return {"sql_query": fixed_sql}
                    except Exception as e:
                        logger.warning(f"Auto-fix failed for {rule['name']}: {e}")

        logger.info("SQL Validator: no issues found")
        return {}


def validator_node(state: AgentState) -> dict:
    """LangGraph node wrapper for SQLValidator."""
    agent = SQLValidator()
    return agent.validate(state)
def validate(self, state: AgentState) -> dict:
    sql = state.get("sql_query", "")
    logger.info(f"VALIDATOR INPUT SQL: {sql[:100]}")
    if not sql:
        return {}

    for rule in VALIDATION_RULES:
        if re.search(rule["pattern"], sql, re.IGNORECASE | re.DOTALL):
            logger.warning(f"VALIDATOR CAUGHT: {rule['name']}")
            if rule["suggestion"]:
                try:
                    fixed_sql = rule["suggestion"](sql)
                    logger.info(f"VALIDATOR FIXED SQL: {fixed_sql[:100]}")
                    return {"sql_query": fixed_sql}
                except Exception as e:
                    logger.warning(f"VALIDATOR FIX FAILED: {e}")

    logger.info("VALIDATOR: no issues found")
    return {}