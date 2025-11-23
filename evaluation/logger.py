"""
Logging utility for evaluation scripts and SQL Agent source tracking
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Base directory (parent of evaluation folder)
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup logger with console and optional file output

    Args:
        name: Logger name
        log_file: Optional path to log file in output directory
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified, saves to output directory)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def setup_query_loggers():
    """
    Setup specialized loggers for query evaluation and source tracking.
    These log to files in the logs/ directory.

    Returns:
        Tuple of (eval_logger, source_logger)
    """
    # Query Evaluation Logger - tracks query execution results
    eval_logger = logging.getLogger('query_evaluation')
    eval_logger.setLevel(logging.INFO)

    if not eval_logger.handlers:  # Avoid duplicate handlers
        eval_handler = logging.FileHandler(
            LOGS_DIR / 'query_evaluation.log',
            mode='a',
            encoding='utf-8'
        )
        eval_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        eval_logger.addHandler(eval_handler)

    # Query Source Tracking Logger - tracks which model generated each query
    source_logger = logging.getLogger('query_source_tracking')
    source_logger.setLevel(logging.INFO)

    if not source_logger.handlers:  # Avoid duplicate handlers
        source_handler = logging.FileHandler(
            LOGS_DIR / 'query_source_tracking.log',
            mode='a',
            encoding='utf-8'
        )
        source_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        source_logger.addHandler(source_handler)

    return eval_logger, source_logger


def log_query_source(
    source_logger,
    db_id: str,
    query: str,
    model_provider: str,
    model_name: str,
    regenerate_count: int,
    is_fallback: bool,
    success: bool,
    sql_query: str = None
):
    """
    Log query source information in a structured format

    Args:
        source_logger: The source tracking logger
        db_id: Database identifier
        query: User's natural language query
        model_provider: "ollama" or "openai"
        model_name: Specific model used (e.g., "gpt-4o", "llama3.1:8b")
        regenerate_count: Number of regeneration attempts
        is_fallback: Whether fallback model was used
        success: Whether query executed successfully
        sql_query: The generated SQL query
    """
    source_logger.info(
        f"DB={db_id} | "
        f"PROVIDER={model_provider} | "
        f"MODEL={model_name} | "
        f"FALLBACK={is_fallback} | "
        f"RETRIES={regenerate_count} | "
        f"SUCCESS={success} | "
        f"QUERY={query[:100]}{'...' if len(query) > 100 else ''}"
    )

    if sql_query:
        # Log SQL on separate line for readability
        sql_oneline = ' '.join(sql_query.split())[:200]
        source_logger.info(f"  SQL={sql_oneline}{'...' if len(sql_oneline) >= 200 else ''}")


def log_query_evaluation(
    eval_logger,
    db_id: str,
    query: str,
    sql_query: str,
    success: bool,
    error_message: str = None,
    rows_returned: int = None,
    confidence_score: float = None
):
    """
    Log query evaluation results

    Args:
        eval_logger: The evaluation logger
        db_id: Database identifier
        query: User's natural language query
        sql_query: Generated SQL query
        success: Whether execution was successful
        error_message: Error message if failed
        rows_returned: Number of rows returned
        confidence_score: Model confidence score
    """
    status = "SUCCESS" if success else "FAILED"
    confidence_str = f"{confidence_score:.2f}" if confidence_score is not None else "N/A"
    rows_str = str(rows_returned) if rows_returned is not None else "N/A"
    eval_logger.info(
        f"{status} | DB={db_id} | "
        f"ROWS={rows_str} | "
        f"CONFIDENCE={confidence_str} | "
        f"QUERY={query[:80]}{'...' if len(query) > 80 else ''}"
    )

    if not success and error_message:
        eval_logger.error(f"  ERROR={error_message[:200]}")


# Pre-initialize loggers for import
_eval_logger, _source_logger = setup_query_loggers()


def get_eval_logger():
    """Get the query evaluation logger"""
    return _eval_logger


def get_source_logger():
    """Get the query source tracking logger"""
    return _source_logger
