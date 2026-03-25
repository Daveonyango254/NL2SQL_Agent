"""
Quick Analysis Script
Analyzes evaluation results without requiring agent traces
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import OUTPUT_DIR


def analyze_results(experiment_name: str = "baseline"):
    """Analyze evaluation results from metadata and results files"""

    print("="*80)
    print("QUICK EVALUATION ANALYSIS")
    print("="*80)

    # Load files
    metadata_file = OUTPUT_DIR / f"{experiment_name}_predictions_metadata.json"
    results_file = OUTPUT_DIR / f"{experiment_name}_results.json"
    ground_truth_file = OUTPUT_DIR / "temp_sampled_dev.json"

    if not metadata_file.exists():
        print(f"[ERROR] Metadata file not found: {metadata_file}")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    with open(results_file, 'r') as f:
        results = json.load(f)

    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    # Create mapping: question_id -> db_id
    qid_to_dbid = {item['question_id']: item['db_id'] for item in ground_truth}

    # Group by database
    db_stats = defaultdict(lambda: {
        'total': 0,
        'slm': 0,
        'llm': 0,
        'retries': [],
        'latencies': [],
        'ex_correct': 0,
        'ves_scores': []
    })

    # Process metadata
    for pred in metadata:
        qid = pred['question_id']
        db_id = qid_to_dbid.get(qid, 'unknown')

        db_stats[db_id]['total'] += 1

        # Model usage
        model_used = pred.get('model_used', 'SLM')
        if model_used == 'LLM':
            db_stats[db_id]['llm'] += 1
        else:
            db_stats[db_id]['slm'] += 1

        # Retries
        retry_count = pred.get('retry_count', 0)
        db_stats[db_id]['retries'].append(retry_count)

        # Latency
        latency = pred.get('latency_ms', 0)
        db_stats[db_id]['latencies'].append(latency)

    # Process EX results
    ex_results = results.get('ex', {}).get('per_question_results', [])
    for i, result in enumerate(ex_results):
        if i < len(ground_truth):
            qid = ground_truth[i]['question_id']
            db_id = qid_to_dbid.get(qid, 'unknown')

            if result.get('res') == 1:
                db_stats[db_id]['ex_correct'] += 1

    # Process VES results
    ves_results = results.get('ves', {}).get('per_question_results', [])
    for i, result in enumerate(ves_results):
        if i < len(ground_truth):
            qid = ground_truth[i]['question_id']
            db_id = qid_to_dbid.get(qid, 'unknown')

            time_ratio = result.get('time_ratio', 0)
            if time_ratio > 0:
                ves_score = np.sqrt(time_ratio) * 100
                db_stats[db_id]['ves_scores'].append(ves_score)

    # Print analysis
    print("\n" + "="*80)
    print("PERFORMANCE BY DATABASE")
    print("="*80)

    # Sort by EX accuracy
    sorted_dbs = sorted(db_stats.items(), key=lambda x: (x[1]['ex_correct'] / x[1]['total']) if x[1]['total'] > 0 else 0, reverse=True)

    for db_id, stats in sorted_dbs:
        ex_acc = (stats['ex_correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_ves = np.mean(stats['ves_scores']) if stats['ves_scores'] else 0
        avg_retries = np.mean(stats['retries']) if stats['retries'] else 0
        avg_latency = np.mean(stats['latencies']) if stats['latencies'] else 0

        print(f"\n{db_id}:")
        print(f"  Queries: {stats['total']}")
        print(f"  EX Accuracy: {ex_acc:.1f}% ({stats['ex_correct']}/{stats['total']})")
        print(f"  VES Score: {avg_ves:.1f}")
        print(f"  Model Usage: {stats['slm']} SLM, {stats['llm']} LLM")
        print(f"  Avg Retries: {avg_retries:.2f}")
        print(f"  Avg Latency: {avg_latency/1000:.1f}s")

    # Overall stats
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    total_queries = sum(s['total'] for s in db_stats.values())
    total_ex_correct = sum(s['ex_correct'] for s in db_stats.values())
    total_slm = sum(s['slm'] for s in db_stats.values())
    total_llm = sum(s['llm'] for s in db_stats.values())
    all_retries = [r for s in db_stats.values() for r in s['retries']]
    all_ves = [v for s in db_stats.values() for v in s['ves_scores']]

    print(f"Total Queries: {total_queries}")
    print(f"Overall EX Accuracy: {(total_ex_correct/total_queries*100):.1f}%")
    print(f"Overall VES: {np.mean(all_ves):.1f}")
    print(f"Model Usage: {total_slm} SLM ({total_slm/total_queries*100:.1f}%), {total_llm} LLM ({total_llm/total_queries*100:.1f}%)")
    print(f"Avg Retries: {np.mean(all_retries):.2f}")

    # Identify problematic databases
    print("\n" + "="*80)
    print("TOP 5 PROBLEMATIC DATABASES (Lowest Accuracy)")
    print("="*80)

    worst_dbs = sorted(sorted_dbs, key=lambda x: (x[1]['ex_correct'] / x[1]['total']) if x[1]['total'] > 0 else 0)[:5]

    for i, (db_id, stats) in enumerate(worst_dbs, 1):
        ex_acc = (stats['ex_correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{i}. {db_id}: {ex_acc:.1f}% ({stats['ex_correct']}/{stats['total']} correct)")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    overall_acc = (total_ex_correct/total_queries*100)

    if overall_acc < 40:
        print("❌ CRITICAL: Overall accuracy is very low (< 40%)")
        print("   → This suggests fundamental issues with the SQL agent")
        print("   → Schema extraction or query generation needs major improvement")
    elif overall_acc < 60:
        print("⚠️  WARNING: Overall accuracy is below target (< 60%)")
        print("   → Agent is functional but needs significant improvements")
    else:
        print("✅ GOOD: Overall accuracy is acceptable (≥ 60%)")

    if total_llm == 0:
        print("\n✅ SLM Only: No fallback to GPT-4o occurred")
        print("   → All queries handled by llama3.1:8b")
    else:
        fallback_rate = (total_llm / total_queries * 100)
        if fallback_rate > 20:
            print(f"\n⚠️  High Fallback Rate: {fallback_rate:.1f}% queries fell back to GPT-4o")
            print("   → SLM is struggling with many queries")
        else:
            print(f"\n✅ Low Fallback Rate: Only {fallback_rate:.1f}% queries fell back to GPT-4o")

    avg_retries = np.mean(all_retries)
    if avg_retries > 1.0:
        print(f"\n⚠️  High Retry Count: {avg_retries:.2f} retries per query on average")
        print("   → Validation is frequently failing")
        print("   → Consider improving SQL generator prompts")
    else:
        print(f"\n✅ Low Retry Count: {avg_retries:.2f} retries per query")

    # Database-specific issues
    databases_with_zero_acc = [db_id for db_id, stats in db_stats.items() if stats['ex_correct'] == 0]
    if databases_with_zero_acc:
        print(f"\n❌ CRITICAL: {len(databases_with_zero_acc)} databases have ZERO accuracy:")
        for db_id in databases_with_zero_acc[:5]:
            print(f"   - {db_id}")
        print("   → These databases need immediate attention")
        print("   → Check schema extraction and example retrieval for these DBs")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick analysis of evaluation results")
    parser.add_argument("--experiment", type=str, default="baseline", help="Experiment name")
    args = parser.parse_args()

    analyze_results(args.experiment)
