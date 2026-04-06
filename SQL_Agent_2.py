"""
SQL Agent with Modular Architecture
Reorganized codebase with strict graph flow and ablation support
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Import from reorganized modules
from api.src.utils.database import (
    discover_databases,
    get_database_path,
    get_database_csv_paths,
    load_database_examples,
    get_base_dir
)
from api.src.utils.llm_factory import LLMConfig
from api.src.utils.sql_extractor import extract_sql
from api.src.prompts import (
    get_decomposer_prompt,
    get_sql_generator_prompt,
    get_formatter_prompt,
    format_sql_only,
    format_sql_with_results,
    format_nlp_explanation,
    format_error
)
from api.src.graph.workflow import build_sql_agent_graph, run_sql_agent

load_dotenv()

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

BASE_DIR = get_base_dir()
CONFIG_PATH = BASE_DIR / "api" / "config.yaml"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Default configuration
DEFAULT_CONFIG = {
    "agents": {
        "schema_extractor": {"enabled": True},
        "query_decomposer": {"enabled": True},
        "sql_generator": {"enabled": True},
        "validator": {"enabled": True}
    },
    "primary_model_type": "openai",
    "openai": {
        "sql_generator_model": "gpt-4o",
        "query_decomposer_model": "gpt-4o",
        "fallback_model": "gpt-4o",
        "temperature": 0
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "sql_generator_model": "llama3.1:8b",
        "query_decomposer_model": "llama3.1:8b",
        "fallback_to_openai": True,
        "temperature": 0
    },
    "retry": {
        "max_retries": 3,
        "fallback_after_retry": 1
    },
    "features": {
        "enable_debug_output": False,
        "log_queries": False,
        "log_file": "sql_agent_logs.txt"
    }
}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load configuration from YAML file with defaults"""
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config:
                # Deep merge yaml_config into config
                for key, value in yaml_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value

            print(f"[OK] Loaded configuration from {config_path}")
        except Exception as e:
            print(f"[WARN] Could not load config.yaml: {e}. Using defaults.")
    else:
        print(f"[WARN] No config.yaml found at {config_path}. Using defaults.")

    return config


# Load configuration
CONFIG = load_config()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

# Global cache for agent and LLM config to avoid reloading embeddings
_AGENT_CACHE = None
_LLM_CONFIG_CACHE = None


def initialize_agent():
    """Initialize the SQL agent graph with all dependencies"""
    global _AGENT_CACHE, _LLM_CONFIG_CACHE

    # Return cached agent if already initialized
    if _AGENT_CACHE is not None:
        return _AGENT_CACHE

    # Create or reuse LLM config (with embeddings cache)
    if _LLM_CONFIG_CACHE is None:
        _LLM_CONFIG_CACHE = LLMConfig(CONFIG)
    llm_config = _LLM_CONFIG_CACHE

    # Format functions
    format_funcs = {
        'format_sql_only': format_sql_only,
        'format_sql_with_results': format_sql_with_results,
        'format_nlp_explanation': format_nlp_explanation,
        'format_error': format_error
    }

    # Build the graph
    _AGENT_CACHE = build_sql_agent_graph(
        get_database_path_func=get_database_path,
        get_csv_paths_func=get_database_csv_paths,
        load_examples_func=load_database_examples,
        embeddings_dir=EMBEDDINGS_DIR,
        get_embeddings_func=llm_config.get_embeddings,
        get_decomposer_llm_func=llm_config.get_decomposer_llm,
        get_decomposer_prompt_func=get_decomposer_prompt,
        get_sql_generator_llm_func=llm_config.get_sql_generator_llm,
        get_sql_generator_prompt_func=get_sql_generator_prompt,
        extract_sql_func=extract_sql,
        get_formatter_llm_func=llm_config.get_formatter_llm,
        get_formatter_prompt_func=get_formatter_prompt,
        format_funcs=format_funcs,
        config=CONFIG
    )

    return _AGENT_CACHE


def execute_query(
    query: str,
    db_id: str,
    output_mode: str = "nlp_explanation"
) -> str:
    """
    Execute a natural language query on the specified database

    Args:
        query: User's natural language query
        db_id: Database identifier
        output_mode: Output format - "sql_only", "sql_with_results", or "nlp_explanation"

    Returns:
        Formatted response string
    """
    if not db_id:
        raise ValueError("db_id is required to identify the database")

    # Initialize agent graph
    graph = initialize_agent()

    # Run the agent
    result = run_sql_agent(
        query=query,
        db_id=db_id,
        output_mode=output_mode,
        graph=graph,
        config=CONFIG
    )

    return result


# =============================================================================
# CLI EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python SQL_Agent_New.py <db_id> <question> <output_mode>")
        print("Example: python SQL_Agent_New.py california_schools \"List all charter schools\" sql_with_results")
        print(f"\nAvailable databases: {list(discover_databases().keys())}")
        print(f"\nUsing: {CONFIG.get('primary_model_type', 'openai')} models")
        sys.exit(1)

    db_id = sys.argv[1]
    question = sys.argv[2]
    output_mode = sys.argv[3]

    print(f"\n{'='*60}")
    print(f"SQL Agent - {CONFIG.get('primary_model_type', 'openai').upper()} Mode")
    print(f"Database: {db_id}")
    print(f"Question: {question}")
    print(f"Output Mode: {output_mode}")
    print(f"{'='*60}\n")

    results = execute_query(
        query=question,
        db_id=db_id,
        output_mode=output_mode
    )

    print(results)
