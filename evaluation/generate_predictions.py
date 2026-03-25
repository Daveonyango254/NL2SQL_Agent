"""
Generate SQL predictions for evaluation
Loads questions from dev_testing.json and generates predictions using SQL_Agent
"""
import sys
from pathlib import Path

# Add parent directory to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.logger import setup_logger
from evaluation.config import BIRD_DEV_JSON, OUTPUT_DIR, DEFAULT_QUERY_TIMEOUT
from evaluation.latency_tracker import LatencyTracker
from SQL_Agent_2 import execute_query, CONFIG
from api.src.utils.sql_extractor import extract_sql_for_evaluation
from func_timeout import func_timeout, FunctionTimedOut
import json
import argparse
import time
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
        experiment_name: str = None,
        timeout: float = DEFAULT_QUERY_TIMEOUT,
        enable_agent_logging: bool = False
    ):
        """
        Initialize prediction generator

        Args:
            output_mode: Agent output mode (sql_only recommended for eval)
            use_langsmith: Enable LangSmith tracing
            experiment_name: Name for LangSmith experiment
            timeout: Timeout per query in seconds (default: 120s)
            enable_agent_logging: Enable detailed agent execution logging
        """
        self.output_mode = output_mode
        self.use_langsmith = use_langsmith
        self.experiment_name = experiment_name or f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.timeout = timeout
        self.latency_tracker = LatencyTracker()
        self.langsmith_client = None
        self.enable_agent_logging = enable_agent_logging
        self.agent_logger = None

        # Initialize LangSmith if requested
        if self.use_langsmith and LANGSMITH_AVAILABLE:
            try:
                self.langsmith_client = Client()
                logger.info(f"LangSmith client initialized")
                logger.info(f"  Experiment: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"Could not initialize LangSmith: {e}")
                self.use_langsmith = False

        # Initialize agent logger if requested
        if self.enable_agent_logging:
            from evaluation.agent_logger import AgentExecutionLogger
            from evaluation.logged_sql_agent import set_logger

            self.agent_logger = AgentExecutionLogger(experiment_name=self.experiment_name)
            set_logger(self.agent_logger)
            logger.info(f"Agent execution logging enabled")
            logger.info(f"  Trace file: {self.agent_logger.trace_file}")

    # @traceable(name="generate_single_prediction")  # Commented out to avoid rate limits
    def generate_single_prediction(
        self,
        question: str,
        db_id: str,
        question_id: int,
        evidence: str = ""
    ) -> Dict:
        """
        Generate prediction for a single question with timeout and latency tracking

        Args:
            question: Natural language question
            db_id: Database identifier
            question_id: Question ID
            evidence: Optional evidence/hints

        Returns:
            Dict with prediction results
        """
        start_time = time.time()
        is_timeout = False
        success = True
        error_msg = None

        try:
            # Run SQL agent with timeout
            if self.enable_agent_logging and self.agent_logger:
                # Log query start
                self.agent_logger.start_query(question_id, db_id, question)

                try:
                    result = func_timeout(
                        self.timeout,
                        execute_query,
                        args=(question, db_id, self.output_mode)
                    )

                    # Log query end (success)
                    final_sql = self._extract_sql(result)
                    self.agent_logger.end_query(success=True, final_sql=final_sql)

                except Exception as e:
                    # Log query end (failure)
                    self.agent_logger.end_query(success=False, error=str(e))
                    raise
            else:
                # No logging - run directly
                result = func_timeout(
                    self.timeout,
                    execute_query,
                    args=(question, db_id, self.output_mode)
                )

            # Extract SQL from result (handles different output modes)
            sql = self._extract_sql(result)

            # Extract model usage and retry count
            model_info = self._extract_model_info(result)

            latency_ms = (time.time() - start_time) * 1000

            # Record latency
            self.latency_tracker.record(
                query_id=question_id,
                latency_ms=latency_ms,
                is_timeout=False,
                success=True
            )

            return {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "predicted_sql": sql,
                "model_used": model_info["model_used"],
                "retry_count": model_info["retry_count"],
                "success": True,
                "error": None,
                "latency_ms": latency_ms
            }

        except FunctionTimedOut:
            latency_ms = (time.time() - start_time) * 1000
            is_timeout = True
            success = False
            error_msg = f"Query timeout after {self.timeout}s"

            # Record timeout
            self.latency_tracker.record(
                query_id=question_id,
                latency_ms=latency_ms,
                is_timeout=True,
                success=False,
                error_msg=error_msg
            )

            # Log query end if agent logging enabled
            if self.enable_agent_logging and self.agent_logger:
                self.agent_logger.end_query(success=False, error=error_msg)

            logger.error(f"Timeout for question {question_id}: {self.timeout}s exceeded")
            return {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "predicted_sql": "",
                "model_used": "UNKNOWN",
                "retry_count": 0,
                "success": False,
                "error": error_msg,
                "latency_ms": latency_ms,
                "timeout": True
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            success = False
            error_msg = str(e)

            # Record error
            self.latency_tracker.record(
                query_id=question_id,
                latency_ms=latency_ms,
                is_timeout=False,
                success=False,
                error_msg=error_msg
            )

            # Log query end if agent logging enabled
            if self.enable_agent_logging and self.agent_logger:
                self.agent_logger.end_query(success=False, error=error_msg)

            logger.error(
                f"Error generating prediction for question {question_id}: {e}")
            return {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "predicted_sql": "",
                "model_used": "UNKNOWN",
                "retry_count": 0,
                "success": False,
                "error": error_msg,
                "latency_ms": latency_ms
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

    def _extract_model_info(self, agent_result: str) -> Dict:
        """
        Extract model usage and retry count from agent result

        Args:
            agent_result: Raw result from agent (formatted response string)

        Returns:
            Dictionary with model_used and retry_count
        """
        model_used = "SLM"  # Default
        retry_count = 0

        # Parse the result string to find model information
        # The agent includes messages like "SQL generated using Ollama (llama3.1:8b)"
        if "OpenAI (gpt-4o)" in agent_result or "OpenAI (" in agent_result:
            model_used = "LLM"
        elif "Ollama (" in agent_result:
            model_used = "SLM"

        # Try to extract retry/regenerate information from validation messages
        # Look for patterns like "Validation failed" which indicates a retry occurred
        import re
        validation_failures = agent_result.count("Validation failed")
        retry_count = validation_failures  # Number of retries = number of validation failures

        return {
            "model_used": model_used,
            "retry_count": retry_count
        }

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
        logger.info(f"Timeout: {self.timeout}s per query")
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
        prediction_metadata = []  # Store full results with metadata
        errors = []

        # Generate predictions with progress bar
        # Use file=sys.stderr to avoid conflicts with logging, disable=False to force display
        for item in tqdm(dev_data, desc="Generating predictions", file=sys.stderr, ncols=100,
                        ascii=True, dynamic_ncols=False):
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

            # Store full result with metadata
            prediction_metadata.append(result)

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

        # Save full prediction metadata (includes model_used, retry_count, etc.)
        metadata_file = output_file.parent / f"{output_file.stem}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_metadata, f, indent=2, ensure_ascii=False)

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
        logger.info(f"Metadata saved to: {metadata_file}")

        # Save and display latency statistics
        latency_file = output_file.parent / f"{output_file.stem}_latency.json"
        self.latency_tracker.save_to_file(latency_file)
        logger.info(f"Latency stats saved to: {latency_file}")

        # Print latency summary
        self.latency_tracker.print_summary()

        # Save agent execution logs if enabled
        if self.enable_agent_logging and self.agent_logger:
            self.agent_logger.save_summary()
            self.agent_logger.print_summary()

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
