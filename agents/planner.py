"""Planner Agent: Breaks down complex questions into logical steps."""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from core.state import AgentState
from config import settings


class PlannerAgent:
    """Decomposes natural language questions into structured logical plans."""

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.ollama_model_reasoning,
            temperature=settings.ollama_temperature,
            base_url=settings.ollama_base_url
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data architect specializing in SQL query planning.

Your task: Decompose the user's question into clear logical steps.

Guidelines:
1. Identify the core intent (aggregation, comparison, trend analysis, joins)
2. Break down into atomic logical steps
3. Define metrics and formulas explicitly
4. Specify filters, groupings, and ordering needed

Output: A clear, numbered plan. Do NOT write SQL code.

Example:
Question: "What is the average order value by customer segment?"
Plan:
1. Join orders table with customers table
2. Calculate AVG(order_total) for each customer
3. Group by customer.segment
4. Order by average value descending"""),
            ("user", "{question}")
        ])

        self.chain = self.prompt | self.llm

    def plan(self, state: AgentState) -> dict:
        """Generate a logical plan for the question."""
        logger.info("PLANNER: Decomposing question into logical steps")
        question = state["question"]

        try:
            response = self.chain.invoke({"question": question})
            plan = response.content

            import re
            steps = re.findall(r'^\d+\..*$', plan, re.MULTILINE)
            logger.info(f"Generated plan with {len(steps)} steps")

            return {
                "plan": plan,
                "plan_steps": steps,
                "iterations": 0,
                "should_retry": True
            }
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return {"error": f"Planning failed: {str(e)}", "should_retry": False}


def planner_node(state: AgentState) -> dict:
    """LangGraph node wrapper for PlannerAgent."""
    agent = PlannerAgent()
    return agent.plan(state)