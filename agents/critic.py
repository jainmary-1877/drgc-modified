"""
Critic Agent (Refiner): Validates, executes, and corrects SQL queries.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from core.state import AgentState
from core.database import db_manager
from config import settings
from agents.generator import SQLGeneratorAgent

class CriticAgent:
    """Validates, executes, and corrects SQL queries."""

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.ollama_model_reasoning,
            temperature=settings.ollama_temperature,
            base_url=settings.ollama_base_url
        )

        self.reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL debugging expert. A query failed and you must fix it.

CRITICAL — inspection_report VALID columns:
- Score: inspection_score, gp_score — NEVER use rating, score, grade
- Date: submitted_on, created_on, closed_on, start_date_time, end_date_time
- Hours: total_inspection_hours
- NEVER use: report_date, inspection_date, rating, score, grade

Follow STRICT steps:
1. Identify the exact error type from the error message
2. Locate the issue in the SQL query  
3. Cross-check with schema_context
4. Apply minimal correction
5. Ensure SQL is valid PostgreSQL syntax

IMPORTANT:
Think step-by-step internally, but output ONLY the final corrected SQL query.

LOGICAL PLAN (MUST FOLLOW):
Use this plan to guide your correction. Do NOT violate it.

Before finalizing:
- Ensure all columns exist in schema
- Ensure joins are valid
- Ensure no ambiguity

IMPORTANT:
Think step-by-step internally, but output ONLY the final corrected SQL query.

SCHEMA:
{schema_context}

ORIGINAL QUESTION:
{question}

FAILED SQL:
{sql_query}

ERROR MESSAGE:
{error}

LOGICAL PLAN:
{plan}

"Output ONLY the corrected SQL query. No explanations, no markdown, no text before or after the SQL."""),
            ("user", "Fix the query:")
        ])
    def execute_and_validate(self, state: AgentState) -> dict:
        """Execute SQL query and handle results/errors."""
        logger.info("CRITIC: Executing and validating SQL query")

        sql_query = state.get("sql_query")
        if not sql_query:
            return {"error": "No SQL query to execute", "should_retry": False}

        try:
            result, error, exec_time = db_manager.execute_query(
                sql_query,
                timeout=settings.query_timeout_seconds
            )

            if error:
                logger.warning(f"Query execution failed: {error}")
                error_type = self._classify_error(error)
                return {
                    "error": error,
                    "error_type": error_type,
                    "query_result": None,
                    "execution_time_ms": exec_time,
                    "should_retry": True
                }
            else:
                logger.info(f"Query executed successfully in {exec_time:.2f}ms")
                result_preview = self._format_result_preview(result)
                return {
                    "query_result": result,
                    "result_preview": result_preview,
                    "execution_time_ms": exec_time,
                    "error": None,
                    "should_retry": False
                }

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {"error": str(e), "error_type": "runtime", "should_retry": True}

    def reflect_and_fix(self, state: AgentState) -> dict:
        """Analyze error and generate corrected SQL."""
        logger.info("CRITIC: Reflecting on error and fixing SQL")

        iterations = state.get("iterations", 0)

        if iterations >= settings.max_iterations:
            logger.error(f"Max iterations ({settings.max_iterations}) reached")
            return {
                "should_retry": False,
                "error": f"Failed to generate valid SQL after {settings.max_iterations} attempts"
            }

        question = state["question"]
        plan = state.get("plan", "")
        schema_context = state.get("schema_context", "")
        sql_query = state.get("sql_query", "")
        error = state.get("error", "")

        try:
            chain = self.reflection_prompt | self.llm
            response = chain.invoke({
            "question": question,
            "plan": plan,
            "schema_context": schema_context,
            "sql_query": sql_query,
            "error": error,
            "error_type": state.get("error_type", "")
        })
            content = response.content
            
            fixed_sql = SQLGeneratorAgent()._clean_sql(content)

            if "SQL:" in content:
                fixed_sql = content.split("SQL:")[-1].strip()
            else:
                fixed_sql = content.strip()

            logger.info(f"Generated corrected SQL (iteration {iterations + 1})")
            logger.debug(f"Fixed SQL: {fixed_sql}")

            return {
                "sql_query": fixed_sql,
                "iterations": iterations + 1,
                "should_retry": True
            }

        except Exception as e:
            logger.error(f"Reflection error: {e}")
            return {"error": f"Failed to correct SQL: {str(e)}", "should_retry": False}
        
    def _classify_error(self, error_msg: str) -> str:
        error_lower = error_msg.lower()
        if "column" in error_lower and ("does not exist" in error_lower or "not found" in error_lower):
            return "column_not_found"
        elif "table" in error_lower and ("does not exist" in error_lower or "not found" in error_lower):
            return "table_not_found"
        elif "syntax" in error_lower:
            return "syntax_error"
        elif "ambiguous" in error_lower:
            return "ambiguous_column"
        elif "timeout" in error_lower:
            return "timeout"
        else:
            return "runtime_error"

    def _format_result_preview(self, result, max_rows: int = 5) -> str:
        if isinstance(result, str):
            return result
        if not result:
            return "Query returned no results"
        try:
            if hasattr(result[0], '_mapping'):
                rows = [dict(row._mapping) for row in result[:max_rows]]
                preview = f"Returned {len(result)} row(s). Preview:\n"
                for i, row in enumerate(rows, 1):
                    preview += f"Row {i}: {row}\n"
                if len(result) > max_rows:
                    preview += f"... ({len(result) - max_rows} more rows)"
                return preview
            else:
                return str(result[:max_rows])
        except Exception as e:
            logger.warning(f"Could not format result: {e}")
            return str(result)[:500]


def executor_node(state: AgentState) -> dict:
    """LangGraph node wrapper for execution."""
    agent = CriticAgent()
    return agent.execute_and_validate(state)


def reflector_node(state: AgentState) -> dict:
    """LangGraph node wrapper for reflection/correction."""
    agent = CriticAgent()
    return agent.reflect_and_fix(state)