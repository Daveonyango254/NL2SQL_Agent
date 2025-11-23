"""
End-to-End Evaluation Pipeline for CESMA SQL Agent

This script orchestrates the complete evaluation workflow:
1. Generate predictions from dev.json using SQL_Agent
2. Run EX (Execution Accuracy) evaluation
3. Run VES (Valid Efficiency Score) evaluation
4. Log results to LangSmith (optional)
5. Save all results to output directory

Usage:
    python evaluation/run_full_pipeline.py --limit 10
    python evaluation/run_full_pipeline.py --use_langsmith --experiment_name my_experiment
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import BIRD_DEV_JSON, BIRD_DB_PATH, OUTPUT_DIR
from evaluation.logger import setup_logger
from evaluation.generate_predictions import PredictionGenerator
from evaluation.run_evaluation import run_full_evaluation

# LangSmith integration (optional)
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


def run_pipeline(
    dev_file: str = None,
    db_root_path: str = None,
    limit: int = None,
    skip: int = 0,
    num_cpus: int = 4,
    meta_time_out: float = 30.0,
    iterate_num: int = 10,
    sql_dialect: str = "SQLite",
    run_ex: bool = True,
    run_ves: bool = True,
    use_langsmith: bool = False,
    experiment_name: str = None,
    output_mode: str = "sql_only"
):
    """
    Run complete evaluation pipeline
    All results are saved to OUTPUT_DIR

    Args:
        dev_file: Path to dev.json (default: BIRD_DEV_JSON)
        db_root_path: Database root directory (default: BIRD_DB_PATH)
        limit: Limit number of questions
        skip: Skip first N questions
        num_cpus: Number of CPU cores for evaluation
        meta_time_out: Timeout per query
        iterate_num: VES timing iterations
        sql_dialect: SQL dialect
        run_ex: Run EX evaluation
        run_ves: Run VES evaluation
        use_langsmith: Enable LangSmith tracing
        experiment_name: Custom experiment name
        output_mode: SQL Agent output mode

    Returns:
        Dictionary with all results
    """
    # Setup defaults
    if dev_file is None:
        dev_file = str(BIRD_DEV_JSON)

    if db_root_path is None:
        db_root_path = str(BIRD_DB_PATH)

    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"pipeline_{timestamp}"

    # Setup logger (saves to output directory)
    log_file = OUTPUT_DIR / f"{experiment_name}_pipeline.log"
    logger = setup_logger("pipeline", log_file=str(log_file))

    logger.info("="*80)
    logger.info("CESMA SQL AGENT - FULL EVALUATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Dev file: {dev_file}")
    logger.info(f"Database root: {db_root_path}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"LangSmith: {use_langsmith}")
    logger.info(f"Limit: {limit if limit else 'All questions'}")
    logger.info(f"Metrics: {', '.join([m for m, run in [('EX', run_ex), ('VES', run_ves)] if run])}")
    logger.info("="*80)

    # STEP 1: Generate Predictions
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Generating Predictions")
    logger.info("="*60)

    prediction_file = OUTPUT_DIR / f"{experiment_name}_predictions.json"

    generator = PredictionGenerator(
        output_mode=output_mode,
        use_langsmith=use_langsmith,
        experiment_name=experiment_name
    )

    predictions = generator.generate_predictions(
        dev_file=dev_file,
        output_file=prediction_file.name,
        limit=limit,
        skip=skip
    )

    logger.info(f"[OK] Predictions generated: {prediction_file}")

    # STEP 2: Run Evaluation
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Running Evaluation")
    logger.info("="*60)

    # Create difficulty labels file path (same as ground truth)
    diff_json_path = dev_file

    results = run_full_evaluation(
        predicted_sql_path=str(prediction_file),
        ground_truth_path=dev_file,
        diff_json_path=diff_json_path,
        db_root_path=db_root_path,
        num_cpus=num_cpus,
        meta_time_out=meta_time_out,
        iterate_num=iterate_num,
        sql_dialect=sql_dialect,
        output_dir=str(OUTPUT_DIR),
        run_ex=run_ex,
        run_ves=run_ves,
        use_langsmith=use_langsmith,
        experiment_name=experiment_name
    )

    # STEP 3: Final Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info(f"  - Predictions:  {prediction_file}")
    logger.info(f"  - Results JSON: {OUTPUT_DIR / f'{experiment_name}_results.json'}")

    if run_ex:
        logger.info(f"  - EX Report:    {OUTPUT_DIR / f'{experiment_name}_ex.txt'}")

    if run_ves:
        logger.info(f"  - VES Report:   {OUTPUT_DIR / f'{experiment_name}_ves.txt'}")

    logger.info(f"  - Pipeline Log: {log_file}")
    logger.info("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run complete CESMA SQL Agent evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 questions
  python evaluation/run_full_pipeline.py --limit 10

  # Full evaluation with LangSmith
  python evaluation/run_full_pipeline.py --use_langsmith --experiment_name my_test

  # Run only EX metric with 4 CPUs
  python evaluation/run_full_pipeline.py --skip_ves --num_cpus 4

  # Resume from question 50, run 25 questions
  python evaluation/run_full_pipeline.py --skip 50 --limit 25
        """
    )

    # Data options
    parser.add_argument(
        "--dev_file",
        type=str,
        default=None,
        help=f"Path to dev.json (default: {BIRD_DEV_JSON})"
    )
    parser.add_argument(
        "--db_root_path",
        type=str,
        default=None,
        help=f"Database root directory (default: {BIRD_DB_PATH})"
    )

    # Generation options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (default: all)"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N questions (default: 0)"
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        default="sql_only",
        choices=["sql_only", "sql_with_results", "nlp_explanation"],
        help="SQL Agent output mode (default: sql_only)"
    )

    # Evaluation options
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=4,
        help="Number of CPU cores for evaluation (default: 4)"
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
        help="VES timing iterations (default: 10)"
    )
    parser.add_argument(
        "--sql_dialect",
        type=str,
        default="SQLite",
        choices=["SQLite", "PostgreSQL", "MySQL"],
        help="SQL dialect (default: SQLite)"
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

    # Experiment tracking
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment name (default: auto-generated)"
    )
    parser.add_argument(
        "--use_langsmith",
        action="store_true",
        help="Enable LangSmith tracing"
    )

    args = parser.parse_args()

    # Validate
    if args.use_langsmith and not LANGSMITH_AVAILABLE:
        print("[WARNING] LangSmith requested but not installed")
        print("          Install with: pip install langsmith")
        print("          Continuing without LangSmith...")

    # Run pipeline
    results = run_pipeline(
        dev_file=args.dev_file,
        db_root_path=args.db_root_path,
        limit=args.limit,
        skip=args.skip,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        iterate_num=args.iterate_num,
        sql_dialect=args.sql_dialect,
        run_ex=not args.skip_ex,
        run_ves=not args.skip_ves,
        use_langsmith=args.use_langsmith,
        experiment_name=args.experiment_name,
        output_mode=args.output_mode
    )

    print("\n[DONE] Pipeline complete! Check output directory for results.")


if __name__ == "__main__":
    main()
