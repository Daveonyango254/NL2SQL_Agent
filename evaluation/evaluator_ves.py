"""
VES (Valid Efficiency Score) Evaluator
Measures both correctness and query efficiency
Consolidated implementation + wrapper in one file
"""

import sys
import json
import numpy as np
import argparse
import multiprocessing as mp
import time
import math
from pathlib import Path
from func_timeout import func_timeout, FunctionTimedOut

from evaluation.evaluation_utils import (
    load_jsonl,
    package_sqls,
    sort_results,
    print_data,
    connect_db,
)
from evaluation.config import OUTPUT_DIR

# Global result collector
exec_result = []


def result_callback(result):
    """Callback for multiprocessing results"""
    exec_result.append(result)


def clean_abnormal(input):
    """
    Remove outliers from timing measurements using 3-sigma rule

    Args:
        input: List of timing measurements

    Returns:
        List with outliers removed
    """
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list


def execute_sql(sql, db_path, sql_dialect, return_time=False):
    """
    Execute SQL query and optionally return execution time

    Args:
        sql: SQL query
        db_path: Database path
        sql_dialect: SQL dialect
        return_time: If True, return execution time instead of results

    Returns:
        Query results or execution time
    """
    conn = connect_db(sql_dialect, db_path)
    start_time = time.time()
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    conn.close()
    exec_time = time.time() - start_time
    if return_time:
        return exec_time
    return res


def iterated_execute_sql(
    predicted_sql, ground_truth, db_path, iterate_num, sql_dialect
):
    """
    Execute queries multiple times and compute efficiency reward

    Args:
        predicted_sql: Predicted SQL query
        ground_truth: Ground truth SQL query
        db_path: Database path
        iterate_num: Number of timing iterations
        sql_dialect: SQL dialect

    Returns:
        Reward score based on efficiency (0 to 1.25)
    """
    diff_list = []
    predicted_res = execute_sql(predicted_sql, db_path, sql_dialect)
    ground_truth_res = execute_sql(ground_truth, db_path, sql_dialect)
    reward = 0
    time_ratio = 0

    # Only measure efficiency if results match        *******************************************************************************************************************
    if set(predicted_res) == set(ground_truth_res):
        for _ in range(iterate_num):
            predicted_time = execute_sql(
                predicted_sql, db_path, sql_dialect, return_time=True
            )
            ground_truth_time = execute_sql(
                ground_truth, db_path, sql_dialect, return_time=True
            )
            diff_list.append(ground_truth_time / predicted_time)

        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list) / len(processed_diff_list)

    # Assign reward based on time ratio
    if time_ratio == 0:
        reward = 0
    elif time_ratio >= 2:
        reward = 1.25
    elif time_ratio >= 1 and time_ratio < 2:
        reward = 1
    elif time_ratio >= 0.5 and time_ratio < 1:
        reward = 0.75
    elif time_ratio >= 0.25 and time_ratio < 0.5:
        reward = 0.5
    else:
        reward = 0.25

    return reward


def execute_model(
    predicted_sql, ground_truth, db_place, idx, iterate_num, meta_time_out, sql_dialect
):
    """
    Execute a single SQL query pair with timeout

    Args:
        predicted_sql: Predicted SQL query
        ground_truth: Ground truth SQL query
        db_place: Database path
        idx: Query index
        iterate_num: Number of timing iterations
        meta_time_out: Timeout in seconds
        sql_dialect: SQL dialect

    Returns:
        Dictionary with sql_idx and reward
    """
    try:
        reward = func_timeout(
            meta_time_out * iterate_num,
            iterated_execute_sql,
            args=(predicted_sql, ground_truth,
                  db_place, iterate_num, sql_dialect),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        reward = 0
    except Exception as e:
        result = [(f"error",)]
        reward = 0

    result = {"sql_idx": idx, "reward": reward}
    return result


def run_sqls_parallel(
    sqls,
    db_places,
    num_cpus=1,
    iterate_num=100,
    meta_time_out=30.0,
    sql_dialect="SQLite",
):
    """
    Execute SQL query pairs in parallel

    Args:
        sqls: List of (predicted_sql, ground_truth) tuples
        db_places: List of database paths
        num_cpus: Number of CPU cores
        iterate_num: Number of timing iterations
        meta_time_out: Timeout per query
        sql_dialect: SQL dialect
    """
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_places[i],
                i,
                iterate_num,
                meta_time_out,
                sql_dialect,
            ),
            callback=result_callback,
        )
    pool.close()
    pool.join()


def compute_ves(exec_results):
    """
    Compute VES score from execution results

    Args:
        exec_results: List of execution results with rewards

    Returns:
        VES score
    """
    num_queries = len(exec_results)

    # Handle empty results
    if num_queries == 0:
        return 0.0

    total_reward = 0
    count = 0

    for i, result in enumerate(exec_results):
        if result["reward"] != 0:
            count += 1
        total_reward += math.sqrt(result["reward"]) * 100

    ves = total_reward / num_queries
    return ves


def compute_ves_by_diff(exec_results, diff_json_path):
    """
    Compute VES broken down by difficulty level

    Args:
        exec_results: List of execution results
        diff_json_path: Path to difficulty labels file

    Returns:
        Tuple of (simple_ves, moderate_ves, challenging_ves, overall_ves, count_lists)
    """
    num_queries = len(exec_results)
    contents = load_jsonl(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    # Only iterate through the number of results we have
    for i in range(min(len(exec_results), len(contents))):
        content = contents[i]

        if content["difficulty"] == "simple":
            simple_results.append(exec_results[i])
        if content["difficulty"] == "moderate":
            moderate_results.append(exec_results[i])
        if content["difficulty"] == "challenging":
            challenging_results.append(exec_results[i])

    simple_ves = compute_ves(simple_results)
    moderate_ves = compute_ves(moderate_results)
    challenging_ves = compute_ves(challenging_results)
    all_ves = compute_ves(exec_results)

    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return simple_ves, moderate_ves, challenging_ves, all_ves, count_lists


def run_ves_evaluation(
    predicted_sql_path: str,
    ground_truth_path: str,
    db_root_path: str,
    diff_json_path: str,
    num_cpus: int = 1,
    iterate_num: int = 10,
    meta_time_out: float = 30.0,
    sql_dialect: str = "SQLite",
    output_log_path: str = None
) -> dict:
    """
    Run VES (Valid Efficiency Score) evaluation - HIGH-LEVEL INTERFACE

    Args:
        predicted_sql_path: Path to predicted SQL file
        ground_truth_path: Path to ground truth file
        db_root_path: Root directory for databases
        diff_json_path: Path to difficulty labels
        num_cpus: Number of CPU cores
        iterate_num: Number of timing iterations
        meta_time_out: Timeout per query
        sql_dialect: SQL dialect
        output_log_path: Path to save results in output directory

    Returns:
        Dictionary with evaluation results
    """
    # Ensure output directory exists
    if output_log_path is None:
        output_log_path = OUTPUT_DIR / "ves_results.txt"
    else:
        output_log_path = OUTPUT_DIR / Path(output_log_path).name

    # Clear previous results
    exec_result.clear()

    # Load predicted and ground truth SQLs
    # First load predictions to get the question_ids in sorted order
    from evaluation.evaluation_utils import load_predictions
    predictions = load_predictions(predicted_sql_path)
    question_ids_in_order = sorted(predictions.keys())

    # Load prediction queries
    pred_queries, db_paths = package_sqls(
        predicted_sql_path,
        db_root_path,
        mode='pred',
        ground_truth_path=ground_truth_path
    )

    # Load ground truth queries IN THE SAME ORDER as predictions
    # This ensures correct pairing: pred_queries[i] matches gt_queries[i]
    gt_queries, db_paths_gt = package_sqls(
        ground_truth_path,
        db_root_path,
        mode="gt",
        question_ids=question_ids_in_order
    )

    # Pair queries - now they are properly aligned by question_id
    query_pairs = list(zip(pred_queries, gt_queries))

    # Run parallel execution with timing
    run_sqls_parallel(
        query_pairs,
        db_places=db_paths_gt,
        num_cpus=num_cpus,
        iterate_num=iterate_num,
        meta_time_out=meta_time_out,
        sql_dialect=sql_dialect,
    )

    # Sort and compute metrics
    sorted_results = sort_results(exec_result)

    simple_ves, moderate_ves, challenging_ves, overall_ves, count_lists = compute_ves_by_diff(
        sorted_results, diff_json_path
    )

    score_lists = [simple_ves, moderate_ves, challenging_ves, overall_ves]

    # Print and save results to output directory
    print_data(
        score_lists,
        count_lists,
        metric="R-VES",
        result_log_file=str(output_log_path)
    )

    # Return structured results
    return {
        "overall_ves": overall_ves,
        "simple_ves": simple_ves,
        "moderate_ves": moderate_ves,
        "challenging_ves": challenging_ves,
        "counts": {
            "simple": count_lists[0],
            "moderate": count_lists[1],
            "challenging": count_lists[2],
            "total": count_lists[3]
        },
        "output_file": str(output_log_path)
    }


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--predicted_sql_path", type=str, required=True, default=""
    )
    args_parser.add_argument("--ground_truth_path",
                             type=str, required=True, default="")
    args_parser.add_argument("--db_root_path", type=str,
                             required=True, default="")
    args_parser.add_argument("--num_cpus", type=int, default=1)
    args_parser.add_argument("--meta_time_out", type=float, default=30.0)
    args_parser.add_argument("--diff_json_path", type=str, default="")
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--output_log_path", type=str, default="SQLite")
    args_parser.add_argument("--iterate_num", type=int, default=10)
    args = args_parser.parse_args()

    # Run evaluation
    results = run_ves_evaluation(
        predicted_sql_path=args.predicted_sql_path,
        ground_truth_path=args.ground_truth_path,
        db_root_path=args.db_root_path,
        diff_json_path=args.diff_json_path,
        num_cpus=args.num_cpus,
        iterate_num=args.iterate_num,
        meta_time_out=args.meta_time_out,
        sql_dialect=args.sql_dialect,
        output_log_path=args.output_log_path
    )

    print("=" * 80)
    print(f"Finished R-VES evaluation for {args.sql_dialect}")
    print("=" * 80)
