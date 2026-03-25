"""
Logged SQL Agent Wrapper
Wraps execute_query to add agent execution logging
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SQL_Agent_2 import execute_query as _original_execute_query
from evaluation.agent_logger import AgentExecutionLogger
from typing import Optional


# Global logger instance
_logger: Optional[AgentExecutionLogger] = None


def set_logger(logger: AgentExecutionLogger):
    """Set the global logger instance"""
    global _logger
    _logger = logger


def get_logger() -> Optional[AgentExecutionLogger]:
    """Get the current logger instance"""
    return _logger


def execute_query_with_logging(
    query: str,
    db_id: str,
    output_mode: str = "sql_only",
    question_id: int = None
) -> str:
    """
    Execute query with agent logging enabled

    Args:
        query: Natural language question
        db_id: Database identifier
        output_mode: Output format mode
        question_id: Question ID for logging

    Returns:
        Formatted response from SQL agent
    """
    logger = get_logger()

    # If no logger set, run without logging
    if logger is None:
        return _original_execute_query(query, db_id, output_mode)

    # Start logging
    if question_id is not None:
        logger.start_query(question_id, db_id, query)

    try:
        # Execute query
        result = _original_execute_query(query, db_id, output_mode)

        # Log success
        if logger and question_id is not None:
            # Try to extract SQL from result
            final_sql = None
            if "```sql" in result:
                try:
                    final_sql = result.split("```sql")[1].split("```")[0].strip()
                except:
                    pass

            logger.end_query(success=True, final_sql=final_sql)

        return result

    except Exception as e:
        # Log failure
        if logger and question_id is not None:
            logger.end_query(success=False, error=str(e))
        raise


# For backward compatibility, export under original name
__all__ = ['execute_query_with_logging', 'set_logger', 'get_logger']
