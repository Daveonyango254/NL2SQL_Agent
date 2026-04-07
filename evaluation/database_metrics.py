"""
Database-Grouped Metrics Calculator
Aggregates evaluation results by database to show per-database performance
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_predictions_with_metadata(predictions_file: Path) -> List[Dict]:
    """
    Load predictions file with metadata

    Args:
        predictions_file: Path to predictions JSON file

    Returns:
        List of prediction dictionaries with metadata
    """
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both dict format {question_id: sql} and list format
    if isinstance(data, dict):
        # Convert to list format for consistency
        return [{"question_id": qid, "predicted_sql": sql} for qid, sql in data.items()]
    return data


def group_results_by_database(
    predictions: List[Dict],
    ground_truth_file: Path,
    ex_results: List[Dict] = None,
    ves_results: List[Dict] = None,
    latency_data: Dict = None
) -> Dict[str, List[Dict]]:
    """
    Group all results by database ID

    Args:
        predictions: List of prediction dictionaries
        ground_truth_file: Path to ground truth JSON (contains db_id)
        ex_results: EX evaluation results (sorted by sql_idx)
        ves_results: VES evaluation results (sorted by sql_idx)
        latency_data: Latency tracker data

    Returns:
        Dictionary mapping db_id to list of query results
    """
    # Load ground truth to get db_id for each question
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # Create mapping: question_id -> db_id
    qid_to_dbid = {item['question_id']: item['db_id'] for item in ground_truth}

    # Group by database
    db_groups = defaultdict(list)

    # Build comprehensive per-query results
    for i, pred in enumerate(predictions):
        if isinstance(pred, dict):
            qid = pred.get('question_id', i)
        else:
            qid = i

        db_id = qid_to_dbid.get(qid, 'unknown')

        query_result = {
            'question_id': qid,
            'db_id': db_id,
            'prediction': pred,
        }

        # Add EX result if available
        if ex_results and i < len(ex_results):
            query_result['ex_result'] = ex_results[i]

        # Add VES result if available
        if ves_results and i < len(ves_results):
            query_result['ves_result'] = ves_results[i]

        # Add latency if available
        if latency_data and 'query_details' in latency_data:
            if str(qid) in latency_data['query_details']:
                query_result['latency'] = latency_data['query_details'][str(qid)]

        db_groups[db_id].append(query_result)

    return dict(db_groups)


def calculate_database_metrics(db_groups: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Calculate metrics for each database

    Args:
        db_groups: Dictionary mapping db_id to list of query results

    Returns:
        Dictionary mapping db_id to metrics dictionary
    """
    db_metrics = {}

    for db_id, queries in db_groups.items():
        metrics = {
            'db_id': db_id,
            'total_queries': len(queries),
            'slm_queries': 0,
            'llm_queries': 0,
            'ex_correct': 0,
            'ves_scores': [],
            'retries': [],
            'latencies': [],
            'timeouts': 0,
        }

        for query in queries:
            # EX accuracy
            if 'ex_result' in query and query['ex_result'].get('res') == 1:
                metrics['ex_correct'] += 1

            # VES scores (include 0 for failures - matches evaluator logic)
            if 'ves_result' in query:
                time_ratio = query['ves_result'].get('time_ratio', 0)
                # Calculate VES score (0 for failures/timeouts)
                ves_score = np.sqrt(time_ratio) * 100 if time_ratio > 0 else 0
                metrics['ves_scores'].append(ves_score)
            else:
                # No VES result means it wasn't evaluated
                metrics['ves_scores'].append(0)

            # Model usage (SLM vs LLM)
            if 'prediction' in query and isinstance(query['prediction'], dict):
                model_used = query['prediction'].get('model_used', 'SLM')
                if model_used == 'LLM':
                    metrics['llm_queries'] += 1
                else:
                    metrics['slm_queries'] += 1

                # Retry count
                retry_count = query['prediction'].get('retry_count', 0)
                metrics['retries'].append(retry_count)

            # Latency
            if 'latency' in query:
                lat_ms = query['latency'].get('latency_ms', 0)
                is_timeout = query['latency'].get('is_timeout', False)

                metrics['latencies'].append(lat_ms)
                if is_timeout:
                    metrics['timeouts'] += 1

        # Calculate aggregated metrics
        metrics['ex_accuracy'] = (metrics['ex_correct'] / metrics['total_queries'] * 100) if metrics['total_queries'] > 0 else 0
        metrics['ves_average'] = np.mean(metrics['ves_scores']) if metrics['ves_scores'] else 0
        metrics['avg_retries'] = np.mean(metrics['retries']) if metrics['retries'] else 0

        # Latency percentiles (recalculated from raw values)
        if metrics['latencies']:
            metrics['latency_mean'] = np.mean(metrics['latencies'])
            metrics['latency_p50'] = np.percentile(metrics['latencies'], 50)
            metrics['latency_p95'] = np.percentile(metrics['latencies'], 95)
        else:
            metrics['latency_mean'] = 0
            metrics['latency_p50'] = 0
            metrics['latency_p95'] = 0

        db_metrics[db_id] = metrics

    return db_metrics


def generate_database_report_table(db_metrics: Dict[str, Dict], output_path: Path = None) -> str:
    """
    Generate formatted database metrics table

    Args:
        db_metrics: Dictionary of per-database metrics
        output_path: Optional path to save report (both .txt and .csv)

    Returns:
        Formatted table string
    """
    # Calculate overall metrics
    overall = {
        'db_id': 'OVERALL',
        'total_queries': sum(m['total_queries'] for m in db_metrics.values()),
        'slm_queries': sum(m['slm_queries'] for m in db_metrics.values()),
        'llm_queries': sum(m['llm_queries'] for m in db_metrics.values()),
        'ex_correct': sum(m['ex_correct'] for m in db_metrics.values()),
        'ves_scores': [score for m in db_metrics.values() for score in m['ves_scores']],
        'retries': [r for m in db_metrics.values() for r in m['retries']],
        'latencies': [lat for m in db_metrics.values() for lat in m['latencies']],
        'timeouts': sum(m['timeouts'] for m in db_metrics.values()),
    }

    overall['ex_accuracy'] = (overall['ex_correct'] / overall['total_queries'] * 100) if overall['total_queries'] > 0 else 0
    overall['ves_average'] = np.mean(overall['ves_scores']) if overall['ves_scores'] else 0
    overall['avg_retries'] = np.mean(overall['retries']) if overall['retries'] else 0
    overall['latency_mean'] = np.mean(overall['latencies']) if overall['latencies'] else 0
    overall['latency_p50'] = np.percentile(overall['latencies'], 50) if overall['latencies'] else 0
    overall['latency_p95'] = np.percentile(overall['latencies'], 95) if overall['latencies'] else 0

    # Build table
    header = f"{'Database':<22} | {'Queries':>7} | {'EX (%)':>6} | {'VES':>6} | {'SLM':>3} | {'LLM':>3} | {'Avg Retries':>11} | {'Mean (ms)':>9} | {'p50 (ms)':>8} | {'p95 (ms)':>8} | {'Timeouts':>8}"
    separator = "-" * len(header)

    lines = [
        "",
        "DATABASE METRICS REPORT",
        "=" * len(header),
        header,
        separator
    ]

    # Sort databases alphabetically
    sorted_dbs = sorted(db_metrics.items(), key=lambda x: x[0])

    for db_id, metrics in sorted_dbs:
        line = (
            f"{db_id:<22} | "
            f"{metrics['total_queries']:>7} | "
            f"{metrics['ex_accuracy']:>6.1f} | "
            f"{metrics['ves_average']:>6.1f} | "
            f"{metrics['slm_queries']:>3} | "
            f"{metrics['llm_queries']:>3} | "
            f"{metrics['avg_retries']:>11.2f} | "
            f"{metrics['latency_mean']:>9,.0f} | "
            f"{metrics['latency_p50']:>8,.0f} | "
            f"{metrics['latency_p95']:>8,.0f} | "
            f"{metrics['timeouts']:>8}"
        )
        lines.append(line)

    # Add overall row
    lines.append(separator)
    overall_line = (
        f"{overall['db_id']:<22} | "
        f"{overall['total_queries']:>7} | "
        f"{overall['ex_accuracy']:>6.1f} | "
        f"{overall['ves_average']:>6.1f} | "
        f"{overall['slm_queries']:>3} | "
        f"{overall['llm_queries']:>3} | "
        f"{overall['avg_retries']:>11.2f} | "
        f"{overall['latency_mean']:>9,.0f} | "
        f"{overall['latency_p50']:>8,.0f} | "
        f"{overall['latency_p95']:>8,.0f} | "
        f"{overall['timeouts']:>8}"
    )
    lines.append(overall_line)
    lines.append("=" * len(header))
    lines.append("")

    table_str = "\n".join(lines)

    # Save to file if requested
    if output_path:
        # Save text version
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(table_str)

        # Save CSV version
        csv_path = output_path.with_suffix('.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Database,Queries,EX (%),VES,SLM,LLM,Avg Retries,Mean (ms),p50 (ms),p95 (ms),Timeouts\n")
            for db_id, metrics in sorted_dbs:
                f.write(
                    f"{db_id},"
                    f"{metrics['total_queries']},"
                    f"{metrics['ex_accuracy']:.1f},"
                    f"{metrics['ves_average']:.1f},"
                    f"{metrics['slm_queries']},"
                    f"{metrics['llm_queries']},"
                    f"{metrics['avg_retries']:.2f},"
                    f"{metrics['latency_mean']:.0f},"
                    f"{metrics['latency_p50']:.0f},"
                    f"{metrics['latency_p95']:.0f},"
                    f"{metrics['timeouts']}\n"
                )
            # Overall row
            f.write(
                f"{overall['db_id']},"
                f"{overall['total_queries']},"
                f"{overall['ex_accuracy']:.1f},"
                f"{overall['ves_average']:.1f},"
                f"{overall['slm_queries']},"
                f"{overall['llm_queries']},"
                f"{overall['avg_retries']:.2f},"
                f"{overall['latency_mean']:.0f},"
                f"{overall['latency_p50']:.0f},"
                f"{overall['latency_p95']:.0f},"
                f"{overall['timeouts']}\n"
            )

    return table_str


def generate_model_comparison_table(db_groups: Dict[str, List[Dict]], output_path: Path = None) -> str:
    """
    Generate SLM vs LLM comparison table

    Args:
        db_groups: Dictionary mapping db_id to list of query results
        output_path: Optional path to save report

    Returns:
        Formatted comparison table string
    """
    slm_metrics = {
        'queries': 0,
        'ex_correct': 0,
        'ves_scores': [],
        'retries': [],
        'latencies': []
    }

    llm_metrics = {
        'queries': 0,
        'ex_correct': 0,
        'ves_scores': [],
        'retries': [],
        'latencies': []
    }

    # Collect metrics by model type
    for db_id, queries in db_groups.items():
        for query in queries:
            model_used = 'SLM'
            if 'prediction' in query and isinstance(query['prediction'], dict):
                model_used = query['prediction'].get('model_used', 'SLM')

            target = slm_metrics if model_used == 'SLM' else llm_metrics
            target['queries'] += 1

            # EX
            if 'ex_result' in query and query['ex_result'].get('res') == 1:
                target['ex_correct'] += 1

            # VES (include 0 for failures - matches evaluator logic)
            if 'ves_result' in query:
                time_ratio = query['ves_result'].get('time_ratio', 0)
                ves_score = np.sqrt(time_ratio) * 100 if time_ratio > 0 else 0
                target['ves_scores'].append(ves_score)
            else:
                target['ves_scores'].append(0)

            # Retries
            if 'prediction' in query and isinstance(query['prediction'], dict):
                retry_count = query['prediction'].get('retry_count', 0)
                target['retries'].append(retry_count)

            # Latency
            if 'latency' in query:
                lat_ms = query['latency'].get('latency_ms', 0)
                target['latencies'].append(lat_ms)

    # If no LLM queries, skip this table
    if llm_metrics['queries'] == 0:
        return "\n[No LLM queries detected - fallback disabled or not triggered]\n"

    # Calculate aggregated metrics
    slm_ex = (slm_metrics['ex_correct'] / slm_metrics['queries'] * 100) if slm_metrics['queries'] > 0 else 0
    llm_ex = (llm_metrics['ex_correct'] / llm_metrics['queries'] * 100) if llm_metrics['queries'] > 0 else 0

    slm_ves = np.mean(slm_metrics['ves_scores']) if slm_metrics['ves_scores'] else 0
    llm_ves = np.mean(llm_metrics['ves_scores']) if llm_metrics['ves_scores'] else 0

    slm_retries = np.mean(slm_metrics['retries']) if slm_metrics['retries'] else 0
    llm_retries = np.mean(llm_metrics['retries']) if llm_metrics['retries'] else 0

    slm_latency_mean = np.mean(slm_metrics['latencies']) if slm_metrics['latencies'] else 0
    llm_latency_mean = np.mean(llm_metrics['latencies']) if llm_metrics['latencies'] else 0

    slm_latency_p95 = np.percentile(slm_metrics['latencies'], 95) if slm_metrics['latencies'] else 0
    llm_latency_p95 = np.percentile(llm_metrics['latencies'], 95) if llm_metrics['latencies'] else 0

    # Build table
    lines = [
        "",
        "MODEL COMPARISON REPORT",
        "=" * 80,
        f"{'Metric':<20} | {'SLM (llama3.1:8b)':>18} | {'LLM (gpt-4o)':>18} | {'Difference':>15}",
        "-" * 80,
        f"{'Total Queries':<20} | {slm_metrics['queries']:>18} | {llm_metrics['queries']:>18} | {'-':>15}",
        f"{'EX Accuracy (%)':<20} | {slm_ex:>18.1f} | {llm_ex:>18.1f} | {llm_ex - slm_ex:>+14.1f}%",
        f"{'VES Score':<20} | {slm_ves:>18.1f} | {llm_ves:>18.1f} | {llm_ves - slm_ves:>+15.1f}",
        f"{'Mean Latency (ms)':<20} | {slm_latency_mean:>18,.0f} | {llm_latency_mean:>18,.0f} | {llm_latency_mean - slm_latency_mean:>+13,.0f} ms",
        f"{'p95 Latency (ms)':<20} | {slm_latency_p95:>18,.0f} | {llm_latency_p95:>18,.0f} | {llm_latency_p95 - slm_latency_p95:>+13,.0f} ms",
        f"{'Avg Retries':<20} | {slm_retries:>18.2f} | {llm_retries:>18.2f} | {llm_retries - slm_retries:>+15.2f}",
        "=" * 80,
        ""
    ]

    table_str = "\n".join(lines)

    # Save to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(table_str)

    return table_str
