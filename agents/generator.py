"""
SQL Generator Agent: Translates logical plans into SQL queries.
Uses sqlcoder model which is purpose-built for SQL generation.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from loguru import logger
from core.state import AgentState
from config import settings


class SQLGeneratorAgent:
    """Translates logical plans into valid SQL queries."""

    def __init__(self):
        # Use dedicated SQL model for generation
        self.llm = ChatOllama(
            model=settings.ollama_model_sql,
            temperature=settings.ollama_temperature,
            base_url=settings.ollama_base_url
        )
        self.system_prompt="""You are an expert PostgreSQL SQL engineer for a form-builder database.

CRITICAL RULES:
1. Use ONLY the provided schema — never hallucinate table or column names
2. Follow the logical plan exactly
3. Return ONLY the final SQL query — no markdown, no explanations

JSONB RULES (never violate):
- fb_question and fb_page have NO name/label/title columns
- The ONLY tables with a name column are fb_forms and fb_modules
- All question/page labels are in fb_translation_json.translations (JSONB array)
- ALWAYS use jsonb_array_elements() to unpack — never query translations directly
- Use ->> for text extraction, not ->
- JSONB keys are camelCase: translatedText, entityType, elementId, attribute, language
- Valid entityType values: QUESTION, PAGE, FORM — no others exist

SCHEMA (includes mandatory SQL patterns):
{schema_context}

LOGICAL PLAN:
{plan}

USER QUESTION:
{question}

Write the SQL query:"""

        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Generate the SQL query:")
        ])

    def generate(self, state: AgentState, few_shot_examples=None) -> dict:
        """Generate SQL query from plan and schema."""
        logger.info("SQL GENERATOR: Creating SQL query from plan")

        question = state["question"]
        plan = state.get("plan", "")
        schema_context = state.get("schema_context", "")

        if not schema_context:
            logger.error("No schema context available")
            return {"error": "Cannot generate SQL without schema context", "should_retry": False}

        try:
            if few_shot_examples and settings.enable_dynamic_few_shot:
                examples = [
                    {"question": ex.get("question", ""), "sql": ex.get("sql", "")}
                    for ex in few_shot_examples
                ]

                example_prompt = ChatPromptTemplate.from_messages([
                    ("human", "{question}"),
                    ("ai", "{sql}")
                ])

                few_shot_prompt = FewShotChatMessagePromptTemplate(
                    example_prompt=example_prompt,
                    examples=examples
                )

                full_prompt = ChatPromptTemplate.from_messages([
                    ("system", self.system_prompt),
                    few_shot_prompt,
                    ("user", "Generate the SQL query:")
                ])

                chain = full_prompt | self.llm
            else:
                chain = self.generation_prompt | self.llm
            logger.debug(f"SCHEMA CONTEXT BEING SENT:\n{schema_context}")
            logger.debug(f"PLAN BEING SENT:\n{plan}")
            response = chain.invoke({
                "question": question,
                "plan": plan,
                "schema_context": schema_context
            })

            sql = self._clean_sql(response.content)

            logger.info(f"Generated SQL ({len(sql)} characters)")
            logger.debug(f"SQL: {sql}")

            return {
                "sql_query": sql,
                "sql_explanation": response.content
            }

        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return {"error": f"SQL generation failed: {str(e)}", "should_retry": False}

    def _clean_sql(self, raw_sql: str) -> str:
        """Clean SQL output from LLM response."""
        # Remove markdown code blocks
        sql = raw_sql.replace("```sql", "").replace("```", "").strip()

        # Remove backtick-wrapped values (e.g. `'PUBLISHED'` → 'PUBLISHED')
        import re
        sql = re.sub(r'`([^`]*)`', r'\1', sql)

        # Find where SQL actually starts
        lines = sql.split("\n")
        sql_start_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if any(stripped.startswith(kw) for kw in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]):
                sql_start_idx = i
                break

        if sql_start_idx is not None:
            sql = "\n".join(lines[sql_start_idx:])

        # Cut off anything after the final semicolon
        if ";" in sql:
            sql = sql[:sql.rfind(";") + 1]

        return sql.strip()

def generator_node(state: AgentState) -> dict:
    """LangGraph node wrapper for SQLGeneratorAgent."""
    agent = SQLGeneratorAgent()
    few_shot = state.get("few_shot_examples", None)
    return agent.generate(state, few_shot_examples=few_shot)