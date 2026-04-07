"""
Workflow builder for SQL Agent graph
Implements strict graph flow matching the diagram:
start → schema_extraction → query_decomposer → sql_generator → executor_validator → [format/format_error] → end
Regenerate loop: executor_validator → sql_generator (only)
"""

from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from .state import SQLAgentState
from .nodes import (
    create_schema_extraction_node,
    create_query_decomposer_node,
    create_sql_generator_node,
    create_executor_validator_node,
    create_formatter_node,
    create_error_handler_node
)


def should_retry(state: SQLAgentState, config: Dict) -> str:
    """
    Determine next step based on validation
    Strict routing matching the graph diagram:
    - If validation passed → format (formatter node)
    - If max retries exceeded → format_error (error_handler node)
    - Otherwise → regenerate (back to sql_generator ONLY)
    """
    max_retries = config['retry'].get('max_retries', 3)

    if state.get("validation_status", False):
        return "format"
    elif state.get("regenerate_count", 0) >= max_retries:
        return "format_error"
    else:
        return "regenerate"


def build_sql_agent_graph(
    get_database_path_func,
    get_csv_paths_func,
    load_examples_func,
    embeddings_dir: Path,
    get_embeddings_func,
    get_decomposer_llm_func,
    get_decomposer_prompt_func,
    get_sql_generator_llm_func,
    get_sql_generator_prompt_func,
    extract_sql_func,
    get_formatter_llm_func,
    get_formatter_prompt_func,
    format_funcs: Dict,
    config: Dict
):
    """
    Build the SQL agent graph with strict flow matching the diagram

    Graph Flow:
    START
      ↓
    schema_extraction
      ↓
    query_decomposer (ALWAYS - no conditional skipping)
      ↓
    sql_generator
      ↓
    executor_validator
      ↓ (conditional edges)
      ├─→ regenerate (back to sql_generator only)
      ├─→ format (formatter node)
      └─→ format_error (error_handler node)
      ↓
    END
    """
    graph = StateGraph(SQLAgentState)

    # Create nodes using factory functions
    schema_node = create_schema_extraction_node(
        get_database_path_func=get_database_path_func,
        get_csv_paths_func=get_csv_paths_func,
        load_examples_func=load_examples_func,
        embeddings_dir=embeddings_dir,
        get_embeddings_func=get_embeddings_func,
        config=config
    )

    decomposer_node = create_query_decomposer_node(
        get_llm_func=get_decomposer_llm_func,
        get_prompt_func=get_decomposer_prompt_func
    )

    generator_node = create_sql_generator_node(
        get_llm_func=get_sql_generator_llm_func,
        get_prompt_func=get_sql_generator_prompt_func,
        extract_sql_func=extract_sql_func,
        config=config
    )

    validator_node = create_executor_validator_node(config=config)

    formatter_node = create_formatter_node(
        get_llm_func=get_formatter_llm_func,
        get_prompt_func=get_formatter_prompt_func,
        format_funcs=format_funcs
    )

    error_node = create_error_handler_node(
        format_error_func=format_funcs['format_error']
    )

    # Add nodes to graph
    graph.add_node("schema_extraction", schema_node)
    graph.add_node("query_decomposer", decomposer_node)
    graph.add_node("sql_generator", generator_node)
    graph.add_node("executor_validator", validator_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("error_handler", error_node)

    # Define strict linear workflow (matching diagram)
    graph.add_edge(START, "schema_extraction")
    graph.add_edge("schema_extraction", "query_decomposer")
    graph.add_edge("query_decomposer", "sql_generator")
    graph.add_edge("sql_generator", "executor_validator")

    # Conditional routing from executor_validator (matching diagram)
    graph.add_conditional_edges(
        "executor_validator",
        lambda state: should_retry(state, config),
        {
            "format": "formatter",
            "regenerate": "sql_generator",  # Loop back to sql_generator ONLY
            "format_error": "error_handler"
        }
    )

    # Terminal edges
    graph.add_edge("formatter", END)
    graph.add_edge("error_handler", END)

    return graph.compile()


def run_sql_agent(
    query: str,
    db_id: str,
    output_mode: str,
    graph,
    config: Dict
) -> Dict:
    """
    Run the SQL agent with specified configuration

    Args:
        query: User's natural language query
        db_id: Database identifier (required)
        output_mode: One of ["sql_only", "sql_with_results", "nlp_explanation"]
        graph: Compiled graph instance
        config: Configuration dictionary

    Returns:
        Complete agent state dictionary including formatted_response, regenerate_count, etc.
    """
    if not db_id:
        raise ValueError("db_id is required to identify the database")

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "db_id": db_id,
        "user_query": query,
        "output_mode": output_mode,
        "error_count": 0,
        "regenerate_count": 0,
        "confidence_score": 0.0,
        "execution_plan": [],
        "evidence_mapping": [],
        "decomposed_queries": []
    }

    # Run the agent
    result = graph.invoke(initial_state)

    return result  # Return full state instead of just formatted_response
