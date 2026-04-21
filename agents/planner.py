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
1. Identify the core intent (aggregation, comparison, filtering, joins)
2. Break down into atomic logical steps
3. Define metrics and formulas explicitly
4. Specify filters using EXACT logic:
   - For negation: use != or NOT IN, never list assumed values
   - For status filters: state the exact operator
     e.g. "WHERE status != 'PUBLISHED'" not "where status is draft or cancelled"
   - For counts: specify COUNT(*) or COUNT(DISTINCT col)
   - For text matching: always use ILIKE not LIKE
5. Specify groupings and ordering needed

CRITICAL FILTER RULES:
- "not published"   -> filter: status != 'PUBLISHED'
- "only published"  -> filter: status = 'PUBLISHED'
- "active"          -> filter: active = true
- "not active"      -> filter: active = false
- "completed"       -> filter: status = 'CLOSED'
- "pending"         -> filter: status NOT IN ('CLOSED','CLOSE_WITH_DEFERRED')
- "open"            -> filter: status = 'OPEN'
- "overdue"         -> filter: status = 'OVERDUE' OR target_close_out_date < CURRENT_DATE
- Never assume which values satisfy a negation - always use != or NOT IN
- Status values are ALWAYS UPPERCASE
EXACT TABLE NAMES (never use wrong names in plan):
- client (NOT clients)
- facility (NOT facilities)  
- inspection_report (NOT inspections, NOT inspection_reports)
- inspection_corrective_action (NOT corrective_actions)
- users (correct as-is)

INSPECTION_REPORT COLUMN RULES (never use wrong names in plan):
- To get inspection score: use inspection_score column - NEVER write AVG(score), AVG(rating), AVG(grade)
- To get gp score: use gp_score column
- To get hours: use total_inspection_hours column
- Date filtering: use submitted_on - NEVER write report_date, inspection_date
- Always write exact column names in the plan so the SQL generator uses them correctly

Output: A clear numbered plan with explicit filter conditions.
Do NOT write SQL code.

Example:
Question: "How many forms are not published?"
Plan:
1. Query fb_forms table
2. Filter using WHERE status != 'PUBLISHED'
3. Count all matching rows using COUNT(*)

Example:
Question: "Which inspector has the most hours?"
Plan:
1. Query inspection_report table
2. Filter WHERE status != 'DRAFT' to exclude unfinished inspections
3. JOIN users table on inspector_user_id to get inspector names
4. SUM(total_inspection_hours) grouped by inspector name
5. ORDER BY total DESC LIMIT 1

Example:
Question: "which inspector has the lowest average score"
Plan:
1. Query inspection_report table
2. Filter WHERE status != 'DRAFT' and deleted = false
3. JOIN users table on inspector_user_id to get inspector names
4. Calculate AVG(ir.inspection_score) grouped by inspector name
5. ORDER BY avg_score ASC LIMIT 1"""),
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