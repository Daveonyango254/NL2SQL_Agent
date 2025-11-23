"""
Generate SQL predictions for evaluation
Loads questions from dev_testing.json and generates predictions using SQL_Agent
"""
import sys
from pathlib import Path

# Add parent directory to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.logger import setup_logger
from evaluation.config import BIRD_DEV_JSON, OUTPUT_DIR
from SQL_Agent import run_sql_agent, CONFIG
from utils.sql_extractor import extract_sql_for_evaluation
import json
import argparse
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm



# LangSmith integration (optional)
try:
    from langsmith import Client, traceable
    from langsmith.run_helpers import trace
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Create dummy decorator if LangSmith not available

    def traceable(**kwargs):
        def decorator(func):
            return func
        return decorator


logger = setup_logger("generate_predictions")


class PredictionGenerator:
    """Generate SQL predictions using SQL_Agent with optional LangSmith tracing"""

    def __init__(
        self,
        output_mode: str = "sql_only",
        use_langsmith: bool = False,
        experiment_name: str = None
    ):
        """
        Initialize prediction generator

        Args:
            output_mode: Agent output mode (sql_only recommended for eval)
            use_langsmith: Enable LangSmith tracing
            experiment_name: Name for LangSmith experiment
        """
        self.output_mode = output_mode
        self.use_langsmith = use_langsmith
        self.experiment_name = experiment_name or f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.langsmith_client = None

        # Initialize LangSmith if requested
        if self.use_langsmith and LANGSMITH_AVAILABLE:
            try:
                self.langsmith_client = Client()
                logger.info(f"LangSmith client initialized")
                logger.info(f"  Experiment: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"Could not initialize LangSmith: {e}")
                self.use_langsmith = False

    # @traceable(name="generate_single_prediction")  # Commented out to avoid rate limits
    def generate_single_prediction(
        self,
        question: str,
        db_id: str,
        question_id: int,
        evidence: str = ""
    ) -> Dict:
        """
        Generate prediction for a single question with LangSmith tracing

        Args:
            question: Natural language question
            db_id: Database identifier
            question_id: Question ID
            evidence: Optional evidence/hints

        Returns:
            Dict with prediction results
        """
        try:
            # Run SQL agent
            result = run_sql_agent(
                query=question,
                db_id=db_id,
                output_mode=self.output_mode
            )

            # Extract SQL from result (handles different output modes)
            sql = self._extract_sql(result)

            return {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "predicted_sql": sql,
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(
                f"Error generating prediction for question {question_id}: {e}")
            return {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "predicted_sql": "",
                "success": False,
                "error": str(e)
            }

    def _extract_sql(self, agent_result: str) -> str:
        """
        Extract SQL query from agent result using robust extraction utility

        Args:
            agent_result: Raw result from agent

        Returns:
            Extracted SQL query (clean, single-line for evaluation)
        """
        return extract_sql_for_evaluation(agent_result)

    def generate_predictions(
        self,
        dev_file: str,
        output_file: str = None,
        limit: int = None,
        skip: int = 0
    ) -> Dict[int, str]:
        """
        Generate predictions for all questions in dev file
        Saves predictions to output directory

        Args:
            dev_file: Path to dev.json file
            output_file: Path to save predictions (in output directory)
            limit: Optional limit on number of questions
            skip: Number of questions to skip (for resuming)

        Returns:
            Dictionary mapping question_id to predicted SQL
        """
        logger.info("="*80)
        logger.info("SQL Prediction Generation")
        logger.info("="*80)
        logger.info(f"Dev file: {dev_file}")
        logger.info(f"Output mode: {self.output_mode}")
        logger.info(f"LangSmith: {self.use_langsmith}")
        logger.info(f"Model: {CONFIG.get('primary_model_type', 'openai')}")
        logger.info("="*80)

        # Load dev data
        with open(dev_file, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)

        # Apply limit and skip
        if skip > 0:
            dev_data = dev_data[skip:]
            logger.info(f"Skipped first {skip} questions")

        if limit:
            dev_data = dev_data[:limit]
            logger.info(f"Limited to {limit} questions")

        total_questions = len(dev_data)
        logger.info(f"\nProcessing {total_questions} questions...")

        predictions = {}
        errors = []

        # Generate predictions with progress bar
        for item in tqdm(dev_data, desc="Generating predictions"):
            question_id = item['question_id']
            question = item['question']
            db_id = item['db_id']
            evidence = item.get('evidence', '')

            # Generate prediction
            result = self.generate_single_prediction(
                question=question,
                db_id=db_id,
                question_id=question_id,
                evidence=evidence
            )

            if result['success']:
                predictions[question_id] = result['predicted_sql']
            else:
                errors.append(result)
                # Store empty SQL for failed predictions
                predictions[question_id] = ""

        # Save predictions to output directory
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"predictions_{timestamp}.json"
        else:
            output_file = OUTPUT_DIR / Path(output_file).name

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        # Save error log if there were errors
        if errors:
            error_file = output_file.parent / f"{output_file.stem}_errors.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            logger.warning(
                f"\n{len(errors)} errors occurred. See: {error_file}")

        # Print summary
        success_count = len([p for p in predictions.values() if p])
        logger.info("\n" + "="*80)
        logger.info("PREDICTION GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total questions: {total_questions}")
        logger.info(f"Successful:      {success_count}")
        logger.info(f"Failed:          {len(errors)}")
        logger.info(f"\nPredictions saved to: {output_file}")
        logger.info("="*80)

        return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL predictions for evaluation"
    )

    # Input/Output
    parser.add_argument(
        "--dev_file",
        type=str,
        default=str(BIRD_DEV_JSON),
        help=f"Path to dev.json file (default: {BIRD_DEV_JSON})"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output filename (saved to output directory, default: auto-generated)"
    )

    # Generation options
    parser.add_argument(
        "--output_mode",
        type=str,
        default="sql_only",
        choices=["sql_only", "sql_with_results", "nlp_explanation"],
        help="Agent output mode (default: sql_only)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process (default: all)"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N questions (for resuming, default: 0)"
    )

    # LangSmith
    parser.add_argument(
        "--use_langsmith",
        action="store_true",
        help="Enable LangSmith tracing"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="LangSmith experiment name (default: auto-generated)"
    )

    args = parser.parse_args()

    # Check if LangSmith requested but not available
    if args.use_langsmith and not LANGSMITH_AVAILABLE:
        logger.warning(
            "LangSmith requested but not installed. Install with: pip install langsmith")
        logger.warning("Continuing without LangSmith...")

    # Create generator
    generator = PredictionGenerator(
        output_mode=args.output_mode,
        use_langsmith=args.use_langsmith,
        experiment_name=args.experiment_name
    )

    # Generate predictions
    predictions = generator.generate_predictions(
        dev_file=args.dev_file,
        output_file=args.output_file,
        limit=args.limit,
        skip=args.skip
    )

    logger.info("\n[DONE] Prediction generation complete!")
    return predictions


if __name__ == "__main__":
    main()
