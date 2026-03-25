"""
Compare predictions vs ground truth to identify failure patterns
"""

import sys
import json
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import OUTPUT_DIR


def execute_sql(db_path: Path, sql: str, timeout: float = 5.0):
    """Execute SQL and return results"""
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def compare_predictions(experiment_name: str = "baseline", num_samples: int = 5):
    """Compare predictions with ground truth"""

    # Load files
    predictions_file = OUTPUT_DIR / f"{experiment_name}_predictions.json"
    metadata_file = OUTPUT_DIR / f"{experiment_name}_predictions_metadata.json"
    ground_truth_file = OUTPUT_DIR / "temp_sampled_dev.json"
    db_root = Path("data/bird/dev_databases")

    with open(predictions_file) as f:
        predictions = json.load(f)

    with open(metadata_file) as f:
        metadata = json.load(f)

    with open(ground_truth_file) as f:
        ground_truth = json.load(f)

    print("="*80)
    print("PREDICTION vs GROUND TRUTH COMPARISON")
    print("="*80)

    # Analyze failures
    failures = []
    successes = []

    for gt in ground_truth:
        qid = str(gt['question_id'])
        pred_sql = predictions.get(qid, "")
        gt_sql = gt['SQL']
        db_id = gt['db_id']

        if not pred_sql:
            failures.append({
                'qid': qid,
                'db_id': db_id,
                'question': gt['question'],
                'reason': 'No prediction generated',
                'difficulty': gt.get('difficulty', 'unknown')
            })
            continue

        # Try to execute both
        db_path = db_root / db_id / f"{db_id}.sqlite"

        pred_results, pred_error = execute_sql(db_path, pred_sql)
        gt_results, gt_error = execute_sql(db_path, gt_sql)

        # Compare
        match = False
        reason = ""

        if pred_error:
            reason = f"Prediction SQL error: {pred_error[:100]}"
        elif gt_error:
            reason = f"Ground truth SQL error: {gt_error[:100]}"
        elif pred_results == gt_results:
            match = True
            reason = "Results match"
        else:
            reason = f"Results differ (pred: {len(pred_results) if pred_results else 0} rows, gt: {len(gt_results) if gt_results else 0} rows)"

        if match:
            successes.append({
                'qid': qid,
                'db_id': db_id,
                'question': gt['question'],
                'pred_sql': pred_sql,
                'gt_sql': gt_sql
            })
        else:
            failures.append({
                'qid': qid,
                'db_id': db_id,
                'question': gt['question'],
                'pred_sql': pred_sql,
                'gt_sql': gt_sql,
                'reason': reason,
                'difficulty': gt.get('difficulty', 'unknown'),
                'pred_error': pred_error,
                'pred_results': pred_results[:3] if pred_results else None,
                'gt_results': gt_results[:3] if gt_results else None
            })

    print(f"\nTotal: {len(ground_truth)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    print(f"Accuracy: {len(successes)/len(ground_truth)*100:.1f}%")

    # Show sample failures
    print(f"\n{'='*80}")
    print(f"SAMPLE FAILURES (first {num_samples})")
    print(f"{'='*80}")

    for i, fail in enumerate(failures[:num_samples], 1):
        print(f"\n[{i}] Question ID: {fail['qid']} | Database: {fail['db_id']} | Difficulty: {fail['difficulty']}")
        print(f"Question: {fail['question'][:150]}")
        print(f"\nReason: {fail['reason']}")

        if 'gt_sql' in fail:
            print(f"\nGround Truth SQL:")
            print(f"  {fail['gt_sql'][:200]}")

        if 'pred_sql' in fail:
            print(f"\nPredicted SQL:")
            print(f"  {fail['pred_sql'][:200]}")

        if fail.get('pred_results') and fail.get('gt_results'):
            print(f"\nPredicted Results (first 3 rows):")
            for row in fail['pred_results'][:3]:
                print(f"  {row}")
            print(f"\nGround Truth Results (first 3 rows):")
            for row in fail['gt_results'][:3]:
                print(f"  {row}")

    # Error pattern analysis
    print(f"\n{'='*80}")
    print(f"ERROR PATTERN ANALYSIS")
    print(f"{'='*80}")

    error_types = {}
    for fail in failures:
        if fail.get('pred_error'):
            # Extract error type
            error = fail['pred_error']
            if 'no such table' in error.lower():
                error_type = 'Table not found'
            elif 'no such column' in error.lower():
                error_type = 'Column not found'
            elif 'syntax error' in error.lower():
                error_type = 'Syntax error'
            elif 'ambiguous column' in error.lower():
                error_type = 'Ambiguous column'
            else:
                error_type = 'Other execution error'

            error_types[error_type] = error_types.get(error_type, 0) + 1
        elif 'Results differ' in fail.get('reason', ''):
            error_types['Wrong results (no error)'] = error_types.get('Wrong results (no error)', 0) + 1
        elif 'No prediction' in fail.get('reason', ''):
            error_types['No prediction generated'] = error_types.get('No prediction generated', 0) + 1

    print(f"\nError Type Distribution:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count} ({count/len(failures)*100:.1f}%)")

    # Sample successes
    if successes:
        print(f"\n{'='*80}")
        print(f"SAMPLE SUCCESSES (first 2)")
        print(f"{'='*80}")

        for i, success in enumerate(successes[:2], 1):
            print(f"\n[{i}] Question ID: {success['qid']} | Database: {success['db_id']}")
            print(f"Question: {success['question'][:150]}")
            print(f"\nGround Truth SQL:")
            print(f"  {success['gt_sql'][:200]}")
            print(f"\nPredicted SQL:")
            print(f"  {success['pred_sql'][:200]}")
            print(f"\n✓ Results match!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    compare_predictions(args.experiment, args.samples)
