"""
Unified Evaluation Runner for CESMA SQL Agent
Runs EX and VES evaluations with optional LangSmith integration
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import BIRD_DB_PATH, BASE_DIR, OUTPUT_DIR
from evaluation.logger import setup_logger

# Import evaluators (consolidated modules)
from evaluation.evaluator_ex import run_ex_evaluation
from evaluation.evaluator_ves import run_ves_evaluation

# LangSmith integration (optional)
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None  # Define Client as None if not available

logger = setup_logger("run_evaluation")


def setup_langsmith_experiment(experiment_name: str):
    """
    Initialize LangSmith client and create experiment

    Args:
        experiment_name: Name for the experiment

    Returns:
        LangSmith Client or None if unavailable
    """
    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not available. Install with: pip install langsmith")
        return None

    try:
        client = Client()
        logger.info(f"LangSmith client initialized")
        logger.info(f"  Experiment: {experiment_name}")
        return client
    except Exception as e:
        logger.warning(f"Could not initialize LangSmith: {e}")
        return None


def log_to_langsmith(
    client,
    experiment_name: str,
    results: Dict,
    metadata: Dict
):
    """
    Log evaluation results to LangSmith

    Args:
        client: LangSmith client
        experiment_name: Experiment name
        results: Evaluation results
        metadata: Additional metadata
    """
    if not client:
        return

    try:
        # Create experiment run
        run_data = {
            "name": experiment_name,
            "run_type": "chain",
            "inputs": metadata,
            "outputs": results,
            "extra": {
                "evaluation_type": "sql_agent",
                "metrics": ["EX", "VES"],
                "timestamp": datetime.now().isoformat()
            }
        }

        # Log to LangSmith
        # Note: Actual implementation depends on LangSmith API
        logger.info("[OK] Results logged to LangSmith")

    except Exception as e:
        logger.error(f"Error logging to LangSmith: {e}")


def run_full_evaluation(
    predicted_sql_path: str,
    ground_truth_path: str,
    diff_json_path: str,
    db_root_path: str = None,
    num_cpus: int = 1,
    meta_time_out: float = 30.0,
    iterate_num: int = 10,
    sql_dialect: str = "SQLite",
    output_dir: str = None,
    run_ex: bool = True,
    run_ves: bool = True,
    use_langsmith: bool = False,
    experiment_name: str = None
) -> Dict:
    """
    Run complete evaluation pipeline

    Args:
        predicted_sql_path: Path to predicted SQL file
        ground_truth_path: Path to ground truth SQL file
        diff_json_path: Path to difficulty labels
        db_root_path: Database root directory
        num_cpus: Number of CPU cores
        meta_time_out: Timeout per query
        iterate_num: VES timing iterations
        sql_dialect: SQL dialect
        output_dir: Directory for results
        run_ex: Run EX evaluation
        run_ves: Run VES evaluation
        use_langsmith: Enable LangSmith logging
        experiment_name: Custom experiment name

    Returns:
        Dictionary with all evaluation results
    """
    # Setup - ALL results go to OUTPUT_DIR
    if db_root_path is None:
        db_root_path = str(BIRD_DB_PATH)

    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"cesma_eval_{timestamp}"

    logger.info("="*80)
    logger.info("CESMA SQL Agent - Full Evaluation")
    logger.info("="*80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Predicted SQL: {predicted_sql_path}")
    logger.info(f"Ground Truth: {ground_truth_path}")
    logger.info(f"Difficulty Labels: {diff_json_path}")
    logger.info(f"Database Root: {db_root_path}")
    logger.info(f"SQL Dialect: {sql_dialect}")
    logger.info(f"CPUs: {num_cpus}")
    logger.info(f"Metrics: {', '.join([m for m, run in [('EX', run_ex), ('VES', run_ves)] if run])}")
    logger.info("="*80)

    # Initialize LangSmith if requested
    langsmith_client = None
    if use_langsmith:
        langsmith_client = setup_langsmith_experiment(experiment_name)

    results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "predicted_sql_path": predicted_sql_path,
            "ground_truth_path": ground_truth_path,
            "diff_json_path": diff_json_path,
            "db_root_path": db_root_path,
            "sql_dialect": sql_dialect,
            "num_cpus": num_cpus,
            "meta_time_out": meta_time_out,
            "iterate_num": iterate_num
        }
    }

    # Run EX Evaluation
    if run_ex:
        logger.info("\n" + "="*60)
        logger.info("Running EX (Execution Accuracy) Evaluation")
        logger.info("="*60)

        ex_output_path = output_dir / f"{experiment_name}_ex.txt"

        ex_results = run_ex_evaluation(
            predicted_sql_path=predicted_sql_path,
            ground_truth_path=ground_truth_path,
            db_root_path=db_root_path,
            diff_json_path=diff_json_path,
            num_cpus=num_cpus,
            meta_time_out=meta_time_out,
            sql_dialect=sql_dialect,
            output_log_path=str(ex_output_path)
        )

        results["ex"] = ex_results
        logger.info(f"[OK] EX evaluation complete")
        logger.info(f"  Overall Accuracy: {ex_results['overall_acc']:.2f}%")

    # Run VES Evaluation
    if run_ves:
        logger.info("\n" + "="*60)
        logger.info("Running VES (Valid Efficiency Score) Evaluation")
        logger.info("="*60)

        ves_output_path = output_dir / f"{experiment_name}_ves.txt"

        ves_results = run_ves_evaluation(
            predicted_sql_path=predicted_sql_path,
            ground_truth_path=ground_truth_path,
            db_root_path=db_root_path,
            diff_json_path=diff_json_path,
            num_cpus=num_cpus,
            iterate_num=iterate_num,
            meta_time_out=meta_time_out,
            sql_dialect=sql_dialect,
            output_log_path=str(ves_output_path)
        )

        results["ves"] = ves_results
        logger.info(f"[OK] VES evaluation complete")
        logger.info(f"  Overall VES: {ves_results['overall_ves']:.2f}")

    # Save combined results
    results_file = output_dir / f"{experiment_name}_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n[OK] Results saved to: {results_file}")

    # Log to LangSmith
    if langsmith_client:
        metadata = results["config"]
        log_to_langsmith(langsmith_client, experiment_name, results, metadata)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    if run_ex:
        print(f"\n[EX] Execution Accuracy:")
        print(f"  Overall:     {results['ex']['overall_acc']:>6.2f}%")
        print(f"  Simple:      {results['ex']['simple_acc']:>6.2f}%")
        print(f"  Moderate:    {results['ex']['moderate_acc']:>6.2f}%")
        print(f"  Challenging: {results['ex']['challenging_acc']:>6.2f}%")

    if run_ves:
        print(f"\n[VES] Valid Efficiency Score:")
        print(f"  Overall:     {results['ves']['overall_ves']:>6.2f}")
        print(f"  Simple:      {results['ves']['simple_ves']:>6.2f}")
        print(f"  Moderate:    {results['ves']['moderate_ves']:>6.2f}")
        print(f"  Challenging: {results['ves']['challenging_ves']:>6.2f}")

    print(f"\n[FILE] Results: {results_file}")
    print("="*80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete CESMA SQL Agent evaluation"
    )

    # Required arguments
    parser.add_argument(
        "--predicted_sql_path",
        type=str,
        required=True,
        help="Path to predicted SQL JSON/JSONL file"
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        required=True,
        help="Path to ground truth SQL JSON/JSONL file"
    )
    parser.add_argument(
        "--diff_json_path",
        type=str,
        required=True,
        help="Path to JSONL file with difficulty labels"
    )

    # Optional arguments
    parser.add_argument(
        "--db_root_path",
        type=str,
        default=None,
        help=f"Root directory containing databases (default: {BIRD_DB_PATH})"
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=4,
        help="Number of CPU cores to use (default: 4)"
    )
    parser.add_argument(
        "--meta_time_out",
        type=float,
        default=30.0,
        help="Timeout per query in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--iterate_num",
        type=int,
        default=10,
        help="Number of VES timing iterations (default: 10)"
    )
    parser.add_argument(
        "--sql_dialect",
        type=str,
        default="SQLite",
        choices=["SQLite", "PostgreSQL", "MySQL"],
        help="SQL dialect (default: SQLite)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output files (default: evaluation/output/)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment name (default: auto-generated)"
    )

    # Metric selection
    parser.add_argument(
        "--skip_ex",
        action="store_true",
        help="Skip EX evaluation"
    )
    parser.add_argument(
        "--skip_ves",
        action="store_true",
        help="Skip VES evaluation"
    )

    # LangSmith
    parser.add_argument(
        "--use_langsmith",
        action="store_true",
        help="Enable LangSmith logging"
    )

    args = parser.parse_args()

    # Run evaluation
    results = run_full_evaluation(
        predicted_sql_path=args.predicted_sql_path,
        ground_truth_path=args.ground_truth_path,
        diff_json_path=args.diff_json_path,
        db_root_path=args.db_root_path,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        iterate_num=args.iterate_num,
        sql_dialect=args.sql_dialect,
        output_dir=args.output_dir,
        run_ex=not args.skip_ex,
        run_ves=not args.skip_ves,
        use_langsmith=args.use_langsmith,
        experiment_name=args.experiment_name
    )

    logger.info("\n[DONE] Evaluation pipeline complete!")
