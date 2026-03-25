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
from evaluation.database_metrics import (
    load_predictions_with_metadata,
    group_results_by_database,
    calculate_database_metrics,
    generate_database_report_table,
    generate_model_comparison_table
)
import json
from collections import defaultdict
import random

# LangSmith integration (optional)
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


def sample_queries_per_database(dev_file: str, queries_per_db: int, seed: int = 42) -> str:
    """
    Sample N queries from each database in dev_file

    Args:
        dev_file: Path to dev.json file
        queries_per_db: Number of queries to sample per database
        seed: Random seed for reproducibility

    Returns:
        Path to temporary sampled dev file
    """
    random.seed(seed)

    # Load dev data
    with open(dev_file, 'r') as f:
        dev_data = json.load(f)

    # Group by database
    db_groups = defaultdict(list)
    for item in dev_data:
        db_id = item['db_id']
        db_groups[db_id].append(item)

    # Sample from each database
    sampled_data = []
    for db_id, queries in sorted(db_groups.items()):
        if len(queries) <= queries_per_db:
            # Use all queries if fewer than requested
            sampled_data.extend(queries)
        else:
            # Random sample
            sampled = random.sample(queries, queries_per_db)
            sampled_data.extend(sampled)

    # Sort by question_id to maintain order
    sampled_data.sort(key=lambda x: x['question_id'])

    # Save to temporary file
    sampled_file = OUTPUT_DIR / "temp_sampled_dev.json"
    with open(sampled_file, 'w') as f:
        json.dump(sampled_data, f, indent=2)

    return str(sampled_file)


def run_pipeline(
    dev_file: str = None,
    db_root_path: str = None,
    limit: int = None,
    skip: int = 0,
    queries_per_db: int = None,
    num_cpus: int = 4,
    meta_time_out: float = 30.0,
    iterate_num: int = 10,
    sql_dialect: str = "SQLite",
    run_ex: bool = True,
    run_ves: bool = True,
    use_langsmith: bool = False,
    experiment_name: str = None,
    output_mode: str = "sql_only",
    enable_agent_logging: bool = False
):
    """
    Run complete evaluation pipeline
    All results are saved to OUTPUT_DIR

    Args:
        dev_file: Path to dev.json (default: BIRD_DEV_JSON)
        db_root_path: Database root directory (default: BIRD_DB_PATH)
        limit: Limit number of questions
        skip: Skip first N questions
        queries_per_db: Number of queries to sample per database (optional)
                        If provided, samples N queries from each database
                        for balanced evaluation. Overrides limit/skip.
        num_cpus: Number of CPU cores for evaluation
        meta_time_out: Timeout per query
        iterate_num: VES timing iterations
        sql_dialect: SQL dialect
        run_ex: Run EX evaluation
        run_ves: Run VES evaluation
        use_langsmith: Enable LangSmith tracing
        experiment_name: Custom experiment name
        output_mode: SQL Agent output mode
        enable_agent_logging: Enable detailed agent execution logging

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

    # Apply per-database sampling if requested
    if queries_per_db:
        logger.info(f"Applying per-database sampling: {queries_per_db} queries per database...")
        dev_file = sample_queries_per_database(dev_file, queries_per_db)
        logger.info(f"Sampled dev file created: {dev_file}")
        # Don't use limit/skip with queries_per_db (sampling already applied)
        limit = None
        skip = 0

    logger.info("="*80)
    logger.info("CESMA SQL AGENT - FULL EVALUATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Dev file: {dev_file}")
    logger.info(f"Database root: {db_root_path}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"LangSmith: {use_langsmith}")
    logger.info(f"Sampling: {f'{queries_per_db} queries per database' if queries_per_db else 'None'}")
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
        experiment_name=experiment_name,
        enable_agent_logging=enable_agent_logging
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

    # STEP 2.5: Generate Database-Grouped Metrics Report
    logger.info("\n" + "="*60)
    logger.info("STEP 2.5: Generating Database Metrics Report")
    logger.info("="*60)

    try:
        # Load prediction metadata
        metadata_file = OUTPUT_DIR / f"{experiment_name}_predictions_metadata.json"

        # Load EX/VES results and latency data
        ex_results = None
        ves_results = None
        latency_data = None

        if run_ex and 'ex' in results:
            # EX results are stored in per_question_results
            ex_results = results['ex'].get('per_question_results', [])

            # Load latency data
            ex_latency_file = OUTPUT_DIR / f"{experiment_name}_ex_latency.json"
            if ex_latency_file.exists():
                import json
                with open(ex_latency_file, 'r') as f:
                    latency_data = json.load(f)

        if run_ves and 'ves' in results:
            # VES results are stored in per_question_results
            ves_results = results['ves'].get('per_question_results', [])

        # Load predictions with metadata
        with open(metadata_file, 'r') as f:
            import json
            prediction_metadata = json.load(f)

        # Group results by database
        db_groups = group_results_by_database(
            predictions=prediction_metadata,
            ground_truth_file=Path(dev_file),
            ex_results=ex_results,
            ves_results=ves_results,
            latency_data=latency_data
        )

        # Calculate metrics
        db_metrics = calculate_database_metrics(db_groups)

        # Generate and save reports
        db_report_path = OUTPUT_DIR / f"{experiment_name}_database_report"
        db_report = generate_database_report_table(db_metrics, db_report_path)

        # Print to console
        print(db_report)
        logger.info(f"[OK] Database report saved to: {db_report_path}.txt and .csv")

        # Generate model comparison if LLM queries exist
        model_comparison_path = OUTPUT_DIR / f"{experiment_name}_model_comparison.txt"
        model_comparison = generate_model_comparison_table(db_groups, model_comparison_path)
        print(model_comparison)

        if "No LLM queries" not in model_comparison:
            logger.info(f"[OK] Model comparison saved to: {model_comparison_path}")
        else:
            logger.info("[INFO] No LLM queries detected - fallback disabled or not triggered")

    except Exception as e:
        logger.error(f"[ERROR] Failed to generate database metrics: {e}")
        logger.error(f"  This is non-critical - evaluation results are still valid")

    # STEP 3: Final Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll results saved to: {OUTPUT_DIR}")
    logger.info(f"  - Predictions:       {prediction_file}")
    logger.info(f"  - Metadata:          {OUTPUT_DIR / f'{experiment_name}_predictions_metadata.json'}")
    logger.info(f"  - Results JSON:      {OUTPUT_DIR / f'{experiment_name}_results.json'}")

    if run_ex:
        logger.info(f"  - EX Report:         {OUTPUT_DIR / f'{experiment_name}_ex.txt'}")

    if run_ves:
        logger.info(f"  - VES Report:        {OUTPUT_DIR / f'{experiment_name}_ves.txt'}")

    logger.info(f"  - Database Report:   {OUTPUT_DIR / f'{experiment_name}_database_report.txt'}")
    logger.info(f"  - Database CSV:      {OUTPUT_DIR / f'{experiment_name}_database_report.csv'}")
    logger.info(f"  - Model Comparison:  {OUTPUT_DIR / f'{experiment_name}_model_comparison.txt'}")
    logger.info(f"  - Pipeline Log:      {log_file}")
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
        "--queries_per_db",
        type=int,
        default=None,
        help="Sample N queries from each database for balanced evaluation (overrides --limit and --skip)"
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
    parser.add_argument(
        "--enable_agent_logging",
        action="store_true",
        help="Enable detailed agent execution logging for behavior analysis"
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
        queries_per_db=args.queries_per_db,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        iterate_num=args.iterate_num,
        sql_dialect=args.sql_dialect,
        run_ex=not args.skip_ex,
        run_ves=not args.skip_ves,
        use_langsmith=args.use_langsmith,
        experiment_name=args.experiment_name,
        output_mode=args.output_mode,
        enable_agent_logging=args.enable_agent_logging
    )

    print("\n[DONE] Pipeline complete! Check output directory for results.")


if __name__ == "__main__":
    main()
