"""
State definition for SQL Agent workflow
"""

from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SQLAgentState(TypedDict):
    """State for SQL Agent graph workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    decomposed_queries: List[str]  # Legacy - kept for compatibility
    execution_plan: List[str]  # Step-by-step execution plan
    evidence_mapping: List[str]  # NL → DB element mappings
    schema_context: Dict[str, Any]
    relevant_tables: List[str]
    sql_query: str
    sql_results: Any
    formatted_response: str
    error_count: int
    validation_status: bool
    output_mode: Literal["sql_only", "sql_with_results", "nlp_explanation"]
    confidence_score: float
    db_id: str
    db_path: Optional[str]
    regenerate_count: int  # Track regeneration attempts
