"""
Agent Behavior Analyzer
Analyzes agent execution traces to identify patterns, failure modes, and optimization opportunities
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import numpy as np

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import OUTPUT_DIR
from evaluation.logger import setup_logger

logger = setup_logger("analyze_agent_behavior")


class AgentBehaviorAnalyzer:
    """
    Analyzes agent execution traces to identify:
    - Common failure patterns by node
    - Databases with high error rates
    - Validation failure causes
    - Query complexity correlation with errors
    - Model fallback effectiveness
    """

    def __init__(self, trace_file: Path):
        """
        Initialize analyzer

        Args:
            trace_file: Path to agent trace JSONL file
        """
        self.trace_file = trace_file
        self.traces = self._load_traces()

        # Analysis results
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_regenerations': 0,
            'by_database': {},
            'by_node': {},
            'validation_failure_patterns': [],
            'error_messages': Counter()
        }

    def _load_traces(self) -> List[Dict]:
        """Load all traces from JSONL file"""
        traces = []
        with open(self.trace_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        logger.info(f"Loaded {len(traces)} query traces from {self.trace_file}")
        return traces

    def analyze(self) -> Dict:
        """
        Run complete analysis

        Returns:
            Dictionary with analysis results
        """
        logger.info("="*80)
        logger.info("AGENT BEHAVIOR ANALYSIS")
        logger.info("="*80)

        # Basic statistics
        self._analyze_basic_stats()

        # Per-database analysis
        self._analyze_by_database()

        # Per-node analysis
        self._analyze_by_node()

        # Validation failure patterns
        self._analyze_validation_failures()

        # Error message analysis
        self._analyze_error_messages()

        # Query complexity analysis
        self._analyze_query_complexity()

        return self.stats

    def _analyze_basic_stats(self):
        """Calculate basic success/failure statistics"""
        for trace in self.traces:
            self.stats['total_queries'] += 1

            # Check final event
            events = trace.get('events', [])
            if not events:
                continue

            final_event = events[-1]
            if final_event.get('event_type') == 'query_end':
                if final_event.get('success'):
                    self.stats['successful_queries'] += 1
                else:
                    self.stats['failed_queries'] += 1

            # Count regenerations
            regenerations = sum(1 for e in events if e.get('event_type') == 'validation_failure')
            self.stats['total_regenerations'] += regenerations

        success_rate = (self.stats['successful_queries'] / self.stats['total_queries'] * 100) if self.stats['total_queries'] > 0 else 0

        logger.info(f"\nBasic Statistics:")
        logger.info(f"  Total Queries: {self.stats['total_queries']}")
        logger.info(f"  Successful: {self.stats['successful_queries']} ({success_rate:.1f}%)")
        logger.info(f"  Failed: {self.stats['failed_queries']}")
        logger.info(f"  Total Regenerations: {self.stats['total_regenerations']}")
        logger.info(f"  Avg Regenerations per Query: {self.stats['total_regenerations'] / self.stats['total_queries']:.2f}")

    def _analyze_by_database(self):
        """Analyze performance grouped by database"""
        db_stats = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'regenerations': 0,
            'avg_nodes': [],
            'avg_latency_ms': []
        })

        for trace in self.traces:
            db_id = trace.get('db_id', 'unknown')
            events = trace.get('events', [])

            db_stats[db_id]['total'] += 1

            # Check success
            final_event = events[-1] if events else {}
            if final_event.get('event_type') == 'query_end':
                if final_event.get('success'):
                    db_stats[db_id]['successful'] += 1
                else:
                    db_stats[db_id]['failed'] += 1

                # Latency
                if 'total_duration_ms' in final_event:
                    db_stats[db_id]['avg_latency_ms'].append(final_event['total_duration_ms'])

            # Regenerations
            regenerations = sum(1 for e in events if e.get('event_type') == 'validation_failure')
            db_stats[db_id]['regenerations'] += regenerations

            # Node count
            db_stats[db_id]['avg_nodes'].append(len(events))

        # Calculate averages
        for db_id, stats in db_stats.items():
            stats['success_rate'] = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
            stats['avg_regenerations'] = stats['regenerations'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_nodes_per_query'] = np.mean(stats['avg_nodes']) if stats['avg_nodes'] else 0
            stats['avg_latency'] = np.mean(stats['avg_latency_ms']) if stats['avg_latency_ms'] else 0

        self.stats['by_database'] = dict(db_stats)

        # Print top failures
        logger.info(f"\nPer-Database Analysis:")
        sorted_dbs = sorted(db_stats.items(), key=lambda x: x[1]['failed'], reverse=True)[:10]

        for db_id, stats in sorted_dbs:
            logger.info(f"  {db_id}:")
            logger.info(f"    Success Rate: {stats['success_rate']:.1f}% ({stats['successful']}/{stats['total']})")
            logger.info(f"    Avg Regenerations: {stats['avg_regenerations']:.2f}")
            logger.info(f"    Avg Latency: {stats['avg_latency']:.0f}ms")

    def _analyze_by_node(self):
        """Analyze performance by node (agent component)"""
        node_stats = defaultdict(lambda: {
            'executions': 0,
            'failures': 0,
            'errors': []
        })

        for trace in self.traces:
            events = trace.get('events', [])

            for event in events:
                if event.get('event_type') == 'node_execution':
                    node_name = event.get('node_name', 'unknown')
                    node_stats[node_name]['executions'] += 1

                    if not event.get('success', True):
                        node_stats[node_name]['failures'] += 1
                        if event.get('error'):
                            node_stats[node_name]['errors'].append(event['error'])

        self.stats['by_node'] = dict(node_stats)

        logger.info(f"\nPer-Node Analysis:")
        for node_name, stats in sorted(node_stats.items()):
            failure_rate = (stats['failures'] / stats['executions'] * 100) if stats['executions'] > 0 else 0
            logger.info(f"  {node_name}:")
            logger.info(f"    Executions: {stats['executions']}")
            logger.info(f"    Failures: {stats['failures']} ({failure_rate:.1f}%)")

            if stats['errors']:
                # Show most common errors
                error_counter = Counter(stats['errors'])
                top_errors = error_counter.most_common(3)
                logger.info(f"    Top Errors:")
                for error, count in top_errors:
                    logger.info(f"      - {error[:100]}... ({count}x)")

    def _analyze_validation_failures(self):
        """Analyze validation failure patterns"""
        validation_failures = []

        for trace in self.traces:
            events = trace.get('events', [])
            db_id = trace.get('db_id', 'unknown')
            query_id = trace.get('query_id', -1)

            for event in events:
                if event.get('event_type') == 'validation_failure':
                    validation_failures.append({
                        'query_id': query_id,
                        'db_id': db_id,
                        'regenerate_count': event.get('regenerate_count', 0),
                        'errors': event.get('validation_errors', []),
                        'will_retry': event.get('will_retry', False),
                        'fallback_triggered': event.get('fallback_triggered', False)
                    })

        self.stats['validation_failure_patterns'] = validation_failures

        logger.info(f"\nValidation Failure Analysis:")
        logger.info(f"  Total Validation Failures: {len(validation_failures)}")

        if validation_failures:
            # Analyze by regenerate count
            regenerate_counts = Counter([f['regenerate_count'] for f in validation_failures])
            logger.info(f"  Failures by Regeneration Attempt:")
            for count, freq in sorted(regenerate_counts.items()):
                logger.info(f"    Attempt {count}: {freq}")

            # Analyze fallback triggers
            fallback_count = sum(1 for f in validation_failures if f['fallback_triggered'])
            logger.info(f"  Fallback Triggered: {fallback_count} times")

            # Most common validation errors
            all_errors = []
            for failure in validation_failures:
                all_errors.extend(failure['errors'])

            if all_errors:
                error_counter = Counter(all_errors)
                logger.info(f"  Most Common Validation Errors:")
                for error, count in error_counter.most_common(5):
                    logger.info(f"    - {error[:100]}... ({count}x)")

    def _analyze_error_messages(self):
        """Analyze error messages to find common failure causes"""
        error_messages = []

        for trace in self.traces:
            events = trace.get('events', [])

            for event in events:
                if event.get('error'):
                    error_messages.append(event['error'])

        self.stats['error_messages'] = Counter(error_messages)

        logger.info(f"\nError Message Analysis:")
        logger.info(f"  Total Errors: {len(error_messages)}")

        if error_messages:
            top_errors = self.stats['error_messages'].most_common(10)
            logger.info(f"  Top 10 Error Messages:")
            for error, count in top_errors:
                logger.info(f"    ({count}x) {error[:150]}")

    def _analyze_query_complexity(self):
        """Analyze correlation between query complexity (node count) and failures"""
        successful_complexities = []
        failed_complexities = []

        for trace in self.traces:
            events = trace.get('events', [])
            node_count = len([e for e in events if e.get('event_type') == 'node_execution'])

            final_event = events[-1] if events else {}
            if final_event.get('event_type') == 'query_end':
                if final_event.get('success'):
                    successful_complexities.append(node_count)
                else:
                    failed_complexities.append(node_count)

        logger.info(f"\nQuery Complexity Analysis:")
        if successful_complexities:
            logger.info(f"  Successful Queries:")
            logger.info(f"    Avg Nodes: {np.mean(successful_complexities):.1f}")
            logger.info(f"    Median Nodes: {np.median(successful_complexities):.0f}")

        if failed_complexities:
            logger.info(f"  Failed Queries:")
            logger.info(f"    Avg Nodes: {np.mean(failed_complexities):.1f}")
            logger.info(f"    Median Nodes: {np.median(failed_complexities):.0f}")

    def generate_report(self, output_file: Path = None):
        """
        Generate detailed analysis report

        Args:
            output_file: Path to save report (default: auto-generated)
        """
        if output_file is None:
            output_file = self.trace_file.parent / f"{self.trace_file.stem}_analysis.json"

        report = {
            'trace_file': str(self.trace_file),
            'statistics': self.stats,
            'analysis_summary': {
                'total_queries': self.stats['total_queries'],
                'success_rate': (self.stats['successful_queries'] / self.stats['total_queries'] * 100) if self.stats['total_queries'] > 0 else 0,
                'avg_regenerations': self.stats['total_regenerations'] / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0,
                'problematic_databases': self._get_problematic_databases(),
                'problematic_nodes': self._get_problematic_nodes(),
                'common_validation_errors': self._get_common_validation_errors(),
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"\nAnalysis report saved to: {output_file}")

        return report

    def _get_problematic_databases(self, top_n: int = 5) -> List[Dict]:
        """Get databases with highest failure rates"""
        db_stats = self.stats.get('by_database', {})
        problematic = []

        for db_id, stats in db_stats.items():
            if stats['total'] >= 2:  # At least 2 queries
                problematic.append({
                    'db_id': db_id,
                    'success_rate': stats['success_rate'],
                    'total_queries': stats['total'],
                    'failed_queries': stats['failed']
                })

        # Sort by failure rate (ascending success_rate)
        problematic.sort(key=lambda x: x['success_rate'])
        return problematic[:top_n]

    def _get_problematic_nodes(self) -> List[Dict]:
        """Get nodes with highest failure rates"""
        node_stats = self.stats.get('by_node', {})
        problematic = []

        for node_name, stats in node_stats.items():
            if stats['executions'] > 0:
                failure_rate = (stats['failures'] / stats['executions'] * 100)
                if failure_rate > 0:
                    problematic.append({
                        'node_name': node_name,
                        'failure_rate': failure_rate,
                        'failures': stats['failures'],
                        'executions': stats['executions']
                    })

        # Sort by failure rate
        problematic.sort(key=lambda x: x['failure_rate'], reverse=True)
        return problematic

    def _get_common_validation_errors(self, top_n: int = 5) -> List[Dict]:
        """Get most common validation errors"""
        validation_failures = self.stats.get('validation_failure_patterns', [])

        all_errors = []
        for failure in validation_failures:
            all_errors.extend(failure.get('errors', []))

        if not all_errors:
            return []

        error_counter = Counter(all_errors)
        return [{'error': error, 'count': count} for error, count in error_counter.most_common(top_n)]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze agent execution traces to identify failure patterns"
    )

    parser.add_argument(
        "trace_file",
        type=str,
        help="Path to agent trace JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for analysis report (default: auto-generated)"
    )

    args = parser.parse_args()

    trace_file = Path(args.trace_file)
    if not trace_file.exists():
        logger.error(f"Trace file not found: {trace_file}")
        return

    # Run analysis
    analyzer = AgentBehaviorAnalyzer(trace_file)
    analyzer.analyze()

    # Generate report
    output_file = Path(args.output) if args.output else None
    analyzer.generate_report(output_file)

    logger.info("\n[DONE] Analysis complete!")


if __name__ == "__main__":
    main()
