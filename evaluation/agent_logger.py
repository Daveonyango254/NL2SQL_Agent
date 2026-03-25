"""
Agent Execution Logger for Behavior Analysis
Logs detailed agent state after each node execution for debugging and analysis
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from .config import OUTPUT_DIR


class AgentExecutionLogger:
    """
    Logs agent execution traces for behavior analysis

    Captures:
    - Node transitions (schema → decomposer → generator → validator → formatter)
    - State snapshots after each node
    - Validation failures and regeneration attempts
    - Model selection decisions (SLM vs LLM)
    - Error messages and stack traces
    """

    def __init__(self, experiment_name: str = None, output_dir: Path = None):
        """
        Initialize agent logger

        Args:
            experiment_name: Name for this logging session
            output_dir: Directory to save logs (default: OUTPUT_DIR)
        """
        self.experiment_name = experiment_name or f"agent_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log file paths
        self.trace_file = self.output_dir / f"{self.experiment_name}_agent_trace.jsonl"
        self.summary_file = self.output_dir / f"{self.experiment_name}_agent_summary.json"

        # In-memory storage
        self.current_query_trace: List[Dict] = []
        self.all_traces: List[Dict] = []
        self.current_query_id: Optional[int] = None
        self.current_db_id: Optional[str] = None
        self.query_start_time: float = 0

        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_regenerations': 0,
            'node_errors': {},
            'avg_nodes_per_query': 0,
            'by_database': {}
        }

    def start_query(self, query_id: int, db_id: str, question: str):
        """
        Start logging a new query execution

        Args:
            query_id: Question ID
            db_id: Database ID
            question: Natural language question
        """
        # Save previous query trace if exists
        if self.current_query_trace:
            self._save_query_trace()

        # Reset for new query
        self.current_query_id = query_id
        self.current_db_id = db_id
        self.query_start_time = time.time()
        self.current_query_trace = []

        # Log initial state
        self.log_event(
            node_name="START",
            event_type="query_start",
            data={
                'question_id': query_id,
                'db_id': db_id,
                'question': question,
                'timestamp': datetime.now().isoformat()
            }
        )

    def log_node_execution(
        self,
        node_name: str,
        state: Dict[str, Any],
        success: bool = True,
        error: str = None
    ):
        """
        Log state after node execution

        Args:
            node_name: Name of the node (schema_extraction, query_decomposer, etc.)
            state: Current agent state dictionary
            success: Whether node executed successfully
            error: Error message if failed
        """
        # Extract relevant fields from state
        node_data = {
            'node_name': node_name,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat(),

            # State snapshot (selective fields)
            'regenerate_count': state.get('regenerate_count', 0),
            'error_count': state.get('error_count', 0),
            'validation_status': state.get('validation_status', None),
            'confidence_score': state.get('confidence_score', 0.0),

            # Node-specific data
            'extracted_schema': state.get('extracted_schema') if node_name == 'schema_extraction' else None,
            'execution_plan': state.get('execution_plan') if node_name == 'query_decomposer' else None,
            'evidence_mapping': state.get('evidence_mapping') if node_name == 'query_decomposer' else None,
            'generated_sql': state.get('generated_sql') if node_name == 'sql_generator' else None,
            'validation_errors': state.get('validation_errors') if node_name == 'executor_validator' else None,
            'formatted_response': state.get('formatted_response') if node_name in ['formatter', 'error_handler'] else None,
        }

        # Track model usage for generator node
        if node_name == 'sql_generator':
            node_data['model_type'] = self._infer_model_type(state)

        self.log_event(
            node_name=node_name,
            event_type="node_execution",
            data=node_data
        )

        # Update stats
        if not success:
            self.stats['node_errors'][node_name] = self.stats['node_errors'].get(node_name, 0) + 1

    def log_validation_failure(
        self,
        regenerate_count: int,
        validation_errors: List[str],
        will_retry: bool,
        fallback_triggered: bool = False
    ):
        """
        Log validation failure and regeneration decision

        Args:
            regenerate_count: Current regeneration attempt number
            validation_errors: List of validation error messages
            will_retry: Whether regeneration will be attempted
            fallback_triggered: Whether fallback to LLM was triggered
        """
        self.log_event(
            node_name="executor_validator",
            event_type="validation_failure",
            data={
                'regenerate_count': regenerate_count,
                'validation_errors': validation_errors,
                'will_retry': will_retry,
                'fallback_triggered': fallback_triggered,
                'timestamp': datetime.now().isoformat()
            }
        )

        self.stats['total_regenerations'] += 1

    def log_event(self, node_name: str, event_type: str, data: Dict):
        """
        Generic event logger

        Args:
            node_name: Name of the node generating the event
            event_type: Type of event (node_execution, validation_failure, etc.)
            data: Event data dictionary
        """
        event = {
            'query_id': self.current_query_id,
            'db_id': self.current_db_id,
            'node_name': node_name,
            'event_type': event_type,
            'elapsed_ms': (time.time() - self.query_start_time) * 1000 if self.query_start_time else 0,
            **data
        }

        self.current_query_trace.append(event)

    def end_query(self, success: bool, final_sql: str = None, error: str = None):
        """
        End logging for current query

        Args:
            success: Whether query succeeded
            final_sql: Final generated SQL (if successful)
            error: Error message (if failed)
        """
        query_duration_ms = (time.time() - self.query_start_time) * 1000

        self.log_event(
            node_name="END",
            event_type="query_end",
            data={
                'success': success,
                'final_sql': final_sql,
                'error': error,
                'total_duration_ms': query_duration_ms,
                'total_nodes_executed': len(self.current_query_trace),
                'timestamp': datetime.now().isoformat()
            }
        )

        # Update stats
        self.stats['total_queries'] += 1
        if success:
            self.stats['successful_queries'] += 1
        else:
            self.stats['failed_queries'] += 1

        # Track by database
        if self.current_db_id:
            if self.current_db_id not in self.stats['by_database']:
                self.stats['by_database'][self.current_db_id] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'regenerations': 0
                }

            db_stats = self.stats['by_database'][self.current_db_id]
            db_stats['total'] += 1
            if success:
                db_stats['successful'] += 1
            else:
                db_stats['failed'] += 1

            # Count regenerations for this query
            regenerations = sum(1 for event in self.current_query_trace if event.get('event_type') == 'validation_failure')
            db_stats['regenerations'] += regenerations

        # Save trace
        self._save_query_trace()

    def _save_query_trace(self):
        """Save current query trace to JSONL file"""
        if not self.current_query_trace:
            return

        trace_entry = {
            'query_id': self.current_query_id,
            'db_id': self.current_db_id,
            'events': self.current_query_trace,
            'total_events': len(self.current_query_trace),
            'timestamp': datetime.now().isoformat()
        }

        # Append to JSONL file
        with open(self.trace_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace_entry, ensure_ascii=False) + '\n')

        # Store in memory for summary
        self.all_traces.append(trace_entry)

        # Reset
        self.current_query_trace = []

    def _infer_model_type(self, state: Dict) -> str:
        """
        Infer model type (SLM/LLM) from state

        Args:
            state: Agent state dictionary

        Returns:
            "SLM", "LLM", or "UNKNOWN"
        """
        # Check if fallback was triggered
        regenerate_count = state.get('regenerate_count', 0)

        # This is a heuristic - actual implementation depends on config
        # We'll check for model selection logic in state
        if 'model_type' in state:
            return state['model_type']

        # Fallback heuristic: check regenerate count vs fallback threshold
        # (This would need access to config, so we'll just return UNKNOWN for now)
        return "UNKNOWN"

    def save_summary(self):
        """Save execution summary to JSON file"""
        # Calculate averages
        if self.stats['total_queries'] > 0:
            total_nodes = sum(len(trace['events']) for trace in self.all_traces)
            self.stats['avg_nodes_per_query'] = total_nodes / self.stats['total_queries']

        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'trace_file': str(self.trace_file),
            'total_traces': len(self.all_traces)
        }

        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print execution summary to console"""
        print("\n" + "="*80)
        print("AGENT EXECUTION LOG SUMMARY")
        print("="*80)
        print(f"Experiment: {self.experiment_name}")
        print(f"Total Queries: {self.stats['total_queries']}")
        print(f"  Successful: {self.stats['successful_queries']}")
        print(f"  Failed: {self.stats['failed_queries']}")
        print(f"Total Regenerations: {self.stats['total_regenerations']}")
        print(f"Avg Nodes per Query: {self.stats['avg_nodes_per_query']:.1f}")

        if self.stats['node_errors']:
            print("\nNode Errors:")
            for node, count in sorted(self.stats['node_errors'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {node}: {count}")

        print(f"\nTrace saved to: {self.trace_file}")
        print(f"Summary saved to: {self.summary_file}")
        print("="*80 + "\n")


# Global logger instance (optional - for convenience)
_global_logger: Optional[AgentExecutionLogger] = None


def get_logger(experiment_name: str = None) -> AgentExecutionLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None or experiment_name:
        _global_logger = AgentExecutionLogger(experiment_name=experiment_name)
    return _global_logger


def reset_logger():
    """Reset global logger instance"""
    global _global_logger
    _global_logger = None
