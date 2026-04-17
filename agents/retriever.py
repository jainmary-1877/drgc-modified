"""
Schema Linker Agent (Selector): Identifies relevant tables and columns.
"""

from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from core.state import AgentState
from core.database import db_manager
from config import settings


class SchemaLinkerAgent:
    """
    Performs schema pruning to reduce context noise.
    Identifies only the relevant tables and columns needed for the query.
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
            # Handle common prefixes like fb_, tbl_, etc.
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

    def select_tables(self, question: str, plan: str, all_tables: List[str]) -> List[str]:
        """Select relevant tables using keyword pre-filter + LLM."""
        try:
            candidates = self._keyword_prefilter(question, plan, all_tables)

            chain = self.table_selection_prompt | self.llm
            response = chain.invoke({
                "question": question,
                "plan": plan,
                "all_tables": ", ".join(candidates)
            })

            selected = [t.strip() for t in response.content.split(",")]
            selected = [t for t in selected if t in all_tables]

            logger.info(f"Selected {len(selected)} tables: {', '.join(selected)}")
            return selected if selected else candidates[:5]

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