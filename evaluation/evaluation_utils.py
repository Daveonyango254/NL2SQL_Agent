"""
Utility functions for SQL evaluation
Provides common functions for loading data, executing SQL, and formatting results
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load JSONL file (one JSON object per line) or regular JSON array
    Automatically detects format and loads accordingly

    Args:
        file_path: Path to JSONL or JSON file

    Returns:
        List of dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Try to load as regular JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # If that fails, try JSONL format (one JSON per line)
    data = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def load_json(file_path: str) -> List[Dict]:
    """
    Load JSON file (single array)

    Args:
        file_path: Path to JSON file

    Returns:
        List of dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_predictions(file_path: str) -> Dict[int, str]:
    """
    Load predicted SQL queries from JSON file
    Format: {question_id: predicted_sql}

    Args:
        file_path: Path to predictions file

    Returns:
        Dictionary mapping question_id to SQL query
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert string keys to int if needed
    return {int(k): v for k, v in data.items()}


def connect_db(sql_dialect: str, db_path: str):
    """
    Connect to database based on dialect

    Args:
        sql_dialect: Database type (SQLite, PostgreSQL, MySQL)
        db_path: Path to database file

    Returns:
        Database connection
    """
    if sql_dialect == "SQLite":
        return sqlite3.connect(db_path)
    elif sql_dialect == "PostgreSQL":
        # PostgreSQL would require psycopg2 and connection string parsing
        raise NotImplementedError("PostgreSQL support not yet implemented")
    elif sql_dialect == "MySQL":
        # MySQL would require mysql-connector-python
        raise NotImplementedError("MySQL support not yet implemented")
    else:
        raise ValueError(f"Unsupported SQL dialect: {sql_dialect}")


def execute_sql(
    predicted_sql: str,
    ground_truth: str,
    db_path: str,
    sql_dialect: str,
    metric_func: Callable = None
):
    """
    Execute SQL queries and compute metric

    Args:
        predicted_sql: Predicted SQL query
        ground_truth: Ground truth SQL query
        db_path: Path to database
        sql_dialect: Database type
        metric_func: Function to compute metric (e.g., calculate_ex)

    Returns:
        Metric result
    """
    conn = connect_db(sql_dialect, db_path)
    cursor = conn.cursor()

    # Execute predicted query
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()

    # Execute ground truth query
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()

    conn.close()

    # Apply metric function if provided
    if metric_func:
        return metric_func(predicted_res, ground_truth_res)

    return predicted_res, ground_truth_res


def package_sqls(
    sql_file_path: str,
    db_root_path: str,
    mode: str = "pred",
    ground_truth_path: str = None,
    question_ids: List[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Load SQL queries and corresponding database paths

    Args:
        sql_file_path: Path to SQL file (JSON or JSONL)
        db_root_path: Root directory containing databases
        mode: "pred" for predictions (dict format) or "gt" for ground truth (list format)
        ground_truth_path: Required when mode="pred" to get db_ids
        question_ids: Optional list of question_ids to extract in order (for mode="gt")

    Returns:
        Tuple of (sql_queries, db_paths)
    """
    db_root = Path(db_root_path)

    if mode == "pred":
        # Predictions format: {question_id: sql}
        predictions = load_predictions(sql_file_path)

        # Need ground truth to get db_ids and create db_paths
        if not ground_truth_path:
            raise ValueError("ground_truth_path is required when mode='pred'")

        # Load ground truth to get db_ids
        gt_data = load_json(ground_truth_path)

        # Create mapping from question_id to ground truth entry
        # This ensures correct alignment even when question_ids don't match list indices
        gt_map = {item['question_id']: item for item in gt_data}

        sql_queries = []
        db_paths = []

        for qid in sorted(predictions.keys()):
            sql_queries.append(predictions[qid])

            # Get db_id from ground truth using question_id mapping
            if qid not in gt_map:
                raise ValueError(f"Question ID {qid} not found in ground truth data")

            db_id = gt_map[qid].get('db_id', '')
            db_path = db_root / db_id / f"{db_id}.sqlite"
            db_paths.append(str(db_path))

        return sql_queries, db_paths

    else:
        # Ground truth format: list of dicts with SQL and db_id
        # Load data based on format
        if sql_file_path.endswith('.jsonl'):
            data = load_jsonl(sql_file_path)
        else:
            data = load_json(sql_file_path)

        # If question_ids provided, extract in that order (for alignment with predictions)
        if question_ids:
            # Create mapping from question_id to ground truth entry
            gt_map = {item['question_id']: item for item in data}

            sql_queries = []
            db_paths = []

            for qid in question_ids:
                if qid not in gt_map:
                    raise ValueError(f"Question ID {qid} not found in ground truth data")

                item = gt_map[qid]
                sql = item.get('SQL', '')
                db_id = item.get('db_id', '')

                sql_queries.append(sql)

                # Construct database path
                db_path = db_root / db_id / f"{db_id}.sqlite"
                db_paths.append(str(db_path))

            return sql_queries, db_paths

        # Otherwise return in file order (original behavior)
        sql_queries = []
        db_paths = []

        for item in data:
            sql = item.get('SQL', '')
            db_id = item.get('db_id', '')

            sql_queries.append(sql)

            # Construct database path
            db_path = db_root / db_id / f"{db_id}.sqlite"
            db_paths.append(str(db_path))

        return sql_queries, db_paths


def sort_results(results: List[Dict]) -> List[Dict]:
    """
    Sort evaluation results by sql_idx

    Args:
        results: List of result dictionaries with 'sql_idx' key

    Returns:
        Sorted list of results
    """
    return sorted(results, key=lambda x: x.get('sql_idx', 0))


def print_data(
    score_lists: List[float],
    count_lists: List[int],
    metric: str = "EX",
    result_log_file: str = None
):
    """
    Print evaluation results in formatted table
    Results are ALWAYS saved to result_log_file in output directory

    Args:
        score_lists: [simple, moderate, challenging, overall] scores
        count_lists: [simple, moderate, challenging, overall] counts
        metric: Metric name (EX or R-VES)
        result_log_file: Path in output directory to save results
    """
    simple_score, moderate_score, challenging_score, overall_score = score_lists
    simple_count, moderate_count, challenging_count, total_count = count_lists

    # Format output
    output = []
    output.append("=" * 80)
    output.append(f"{metric} EVALUATION RESULTS")
    output.append("=" * 80)
    output.append(f"{'Difficulty':<15} {'Count':<10} {metric:<10}")
    output.append("-" * 80)
    output.append(f"{'Simple':<15} {simple_count:<10} {simple_score:>6.2f}")
    output.append(f"{'Moderate':<15} {moderate_count:<10} {moderate_score:>6.2f}")
    output.append(f"{'Challenging':<15} {challenging_count:<10} {challenging_score:>6.2f}")
    output.append("-" * 80)
    output.append(f"{'Overall':<15} {total_count:<10} {overall_score:>6.2f}")
    output.append("=" * 80)

    result_text = "\n".join(output)
    print(result_text)

    # ALWAYS save to file in output directory
    if result_log_file:
        output_path = Path(result_log_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"\n[SAVED] Results saved to: {output_path}")
