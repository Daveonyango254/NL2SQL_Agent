"""
Prompt Templates for SQL Agent
Centralized prompts for consistency across all agent implementations
"""

from .query_decomposer import QUERY_DECOMPOSER_PROMPT, get_decomposer_prompt
from .sql_generator import SQL_GENERATOR_PROMPT, get_sql_generator_prompt, get_sql_generator_simple_prompt
from .formatter import (
    FORMATTER_PROMPT,
    get_formatter_prompt,
    format_sql_only,
    format_sql_with_results,
    format_nlp_explanation,
    format_error
)

__all__ = [
    'QUERY_DECOMPOSER_PROMPT',
    'SQL_GENERATOR_PROMPT',
    'FORMATTER_PROMPT',
    'get_decomposer_prompt',
    'get_sql_generator_prompt',
    'get_sql_generator_simple_prompt',
    'get_formatter_prompt',
    'format_sql_only',
    'format_sql_with_results',
    'format_nlp_explanation',
    'format_error'
]
