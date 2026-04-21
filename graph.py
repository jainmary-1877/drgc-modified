"""
LangGraph workflow orchestration for the Text-to-SQL agent.
Implements the DRGC (Decomposition-Retrieval-Generation-Correction) framework.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from loguru import logger
import time

from core.state import AgentState
from agents import (
    planner_node,
    schema_linker_node,
    generator_node,
    executor_node,
    reflector_node,
    validator_node
)
from tools import semantic_cache, few_shot_retriever
from tools.vector_store import auto_seed_if_empty
from config import settings

# Auto-seed once on startup if vector store is empty
auto_seed_if_empty()
def should_continue(state: AgentState) -> Literal["reflect", "end", "cache_success"]:
    """
    Determines the next step in the workflow after query execution.
    
    Decision flow:
    - If query succeeded: cache result and end
    - If max iterations reached: end with error
    - If self-correction disabled: end with error
    - Otherwise: attempt to fix the error
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name: "cache_success", "end", or "reflect"
    """
    # If there's no error, cache and end
    if state.get("error") is None:
        logger.info("✓ Query successful - caching and ending workflow")
        return "cache_success"
    
    # If max iterations reached, stop
    if state.get("iterations", 0) >= settings.max_iterations:
        logger.warning(f"✗ Max iterations ({settings.max_iterations}) reached - ending workflow")
        return "end"
    
    # If self-correction is disabled, stop
    if not settings.enable_self_correction:
        logger.warning("✗ Self-correction disabled - ending workflow")
        return "end"
    
    # If should_retry flag is False, stop
    if not state.get("should_retry", True):
        logger.warning("✗ Retry flag is False - ending workflow")
        return "end"
    
    # Otherwise, attempt reflection/correction
    logger.info(f"↻ Attempting correction (iteration {state.get('iterations', 0) + 1})")
    return "reflect"


def add_start_time(state: AgentState) -> dict:
    """Add timestamp at start of workflow."""
    return {"start_time": time.time()}


def check_cache_node(state: AgentState) -> dict:
    """
    Checks if we have a cached result for this question.
    Uses semantic similarity to find matching previous queries.
    """
    question = state["question"]
    cached = semantic_cache.get(question)
    
    if cached:
        logger.info("✓ Using cached result")
        return {
            **cached,
            "cache_hit": True,
            "should_retry": False
        }
    else:
        return {"cache_hit": False}


def retrieve_few_shot_node(state: AgentState) -> dict:
    """
    Retrieves similar SQL examples from vector store to guide generation.
    These examples help the LLM write better SQL queries.
    """
    if not settings.enable_dynamic_few_shot:
        return {}
    
    question = state["question"]
    examples = few_shot_retriever.retrieve(question)
    
    logger.info(f"Retrieved {len(examples)} few-shot examples")
    return {"few_shot_examples": examples}


def should_use_cache(state: AgentState) -> Literal["use_cache", "continue"]:
    """
    Decides whether to use cached result or continue with full workflow.
    """
    if state.get("cache_hit", False):
        return "use_cache"
    return "continue"


def cache_result_node(state: AgentState) -> dict:
    """
    Stores successful query results in semantic cache for future use.
    Only caches when query executed without errors.
    """
    # Only cache if query was successful
    if state.get("error") is None and state.get("sql_query"):
        result_to_cache = {
            "sql_query": state["sql_query"],
            "query_result": state.get("query_result"),
            "result_preview": state.get("result_preview"),
            "plan": state.get("plan"),
            "relevant_tables": state.get("relevant_tables")
        }
        
        semantic_cache.set(state["question"], result_to_cache)
    
    return {}


def build_graph() -> StateGraph:
    """
    Builds the LangGraph workflow for the Text-to-SQL agent.
    
    Workflow:
    1. Initialize timestamp
    2. Check cache (if hit, skip to end)
    3. Plan: Break down question into logical steps
    4. Retrieve: Get few-shot examples from vector store
    5. Schema Link: Find relevant tables/columns
    6. Generate: Write SQL query
    7. Execute: Run query and validate
    8. On error: Reflect and retry (up to max_iterations)
    9. On success: Cache result and end
    
    Returns:
        Configured StateGraph ready for compilation
    """
    logger.info("Building Text-to-SQL agent graph...")
    
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # === ADD NODES ===
    # Each node represents a step in the workflow
    workflow.add_node("init", add_start_time)  # Track execution time
    workflow.add_node("check_cache", check_cache_node)  # Try to use cached result
    workflow.add_node("planner", planner_node)  # Decompose question into steps
    workflow.add_node("retrieve_few_shot", retrieve_few_shot_node)  # Get example queries
    workflow.add_node("schema_retriever", schema_linker_node)  # Find relevant tables
    workflow.add_node("generator", generator_node)  # Generate SQL
    workflow.add_node("executor", executor_node)  # Execute and validate
    workflow.add_node("reflector", reflector_node)  # Fix SQL errors
    workflow.add_node("validator", validator_node)   # Pre-execution SQL validation if any
    workflow.add_node("cache_result", cache_result_node)  # Store successful result
    
    # === DEFINE WORKFLOW ===
    workflow.set_entry_point("init")
    
    # Always start with cache check
    workflow.add_edge("init", "check_cache")
    
    # If cache hit, skip to end; otherwise continue with planning
    workflow.add_conditional_edges(
        "check_cache",
        should_use_cache,
        {
            "use_cache": END,
            "continue": "planner"
        }
    )
    
    # Linear flow through the DRGC pipeline
    workflow.add_edge("planner", "retrieve_few_shot")
    workflow.add_edge("retrieve_few_shot", "schema_retriever")
    workflow.add_edge("schema_retriever", "generator")
    workflow.add_edge("generator", "validator")
    workflow.add_edge("validator", "executor")
    
    # After execution, decide: success (cache), error (reflect), or give up (end)
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "end": END,
            "cache_success": "cache_result",
            "reflect": "reflector"
        }
    )
    
    workflow.add_edge("cache_result", END)
    
    # After reflection, retry execution
    workflow.add_edge("reflector", "validator")
    
    logger.info("Graph built successfully")
    return workflow


def compile_graph():
    """
    Compile the workflow graph.
    
    Returns:
        Compiled graph ready for execution
    """
    workflow = build_graph()
    app = workflow.compile()
    logger.info("Graph compiled and ready")
    return app


# Create global graph instance
graph = compile_graph()


def run_agent(question: str) -> dict:
    """
    Execute the Text-to-SQL agent for a given question.
    
    Args:
        question: Natural language question
        
    Returns:
        Final state with results
    """
    logger.info(f"{'='*60}")
    logger.info(f"Running Text-to-SQL Agent")
    logger.info(f"Question: {question}")
    logger.info(f"{'='*60}")
    
    # Initialize state
    initial_state: AgentState = {
        "question": question,
        "plan": None,
        "plan_steps": None,
        "relevant_tables": None,
        "schema_context": None,
        "schema_metadata": None,
        "sql_query": None,
        "sql_explanation": None,
        "few_shot_examples": None,
        "query_result": None,
        "result_preview": None,
        "execution_time_ms": None,
        "error": None,
        "error_type": None,
        "iterations": 0,
        "should_retry": True,
        "messages": [],
        "start_time": None,
        "cache_hit": False
    }
    
    try:
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        # Calculate total latency
        if final_state.get("start_time"):
            total_time = (time.time() - final_state["start_time"]) * 1000
            logger.info(f"Total execution time: {total_time:.2f}ms")
            final_state["total_latency_ms"] = total_time
        
        # Log summary
        if final_state.get("error"):
            logger.error(f"✗ Agent failed: {final_state['error']}")
        else:
            logger.info(f"✓ Agent succeeded")
            logger.info(f"SQL: {final_state.get('sql_query', 'N/A')}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        return {
            **initial_state,
            "error": str(e),
            "should_retry": False
        }


async def run_agent_async(question: str) -> dict:
    """
    Asynchronous version of run_agent.
    
    Args:
        question: Natural language question
        
    Returns:
        Final state with results
    """
    logger.info(f"{'='*60}")
    logger.info(f"Running Text-to-SQL Agent (Async)")
    logger.info(f"Question: {question}")
    logger.info(f"{'='*60}")
    
    initial_state: AgentState = {
        "question": question,
        "plan": None,
        "plan_steps": None,
        "relevant_tables": None,
        "schema_context": None,
        "schema_metadata": None,
        "sql_query": None,
        "sql_explanation": None,
        "few_shot_examples": None,
        "query_result": None,
        "result_preview": None,
        "execution_time_ms": None,
        "error": None,
        "error_type": None,
        "iterations": 0,
        "should_retry": True,
        "messages": [],
        "start_time": None,
        "cache_hit": False
    }
    
    try:
        final_state = await graph.ainvoke(initial_state)
        
        if final_state.get("start_time"):
            total_time = (time.time() - final_state["start_time"]) * 1000
            final_state["total_latency_ms"] = total_time
        
        return final_state
        
    except Exception as e:
        logger.error(f"Async graph execution error: {e}")
        return {
            **initial_state,
            "error": str(e),
            "should_retry": False
        }
