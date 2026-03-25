"""
Recommendation Generator
Analyzes evaluation results and agent behavior to generate actionable improvement recommendations
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import OUTPUT_DIR
from evaluation.logger import setup_logger

logger = setup_logger("generate_recommendations")


class RecommendationGenerator:
    """
    Generates actionable recommendations based on:
    - Database metrics (EX, VES, latency by database)
    - Agent behavior analysis (failure patterns, node errors)
    - Model comparison (SLM vs LLM performance)
    """

    def __init__(
        self,
        database_report_file: Path,
        analysis_report_file: Path,
        model_comparison_file: Path = None
    ):
        """
        Initialize recommendation generator

        Args:
            database_report_file: Path to database metrics JSON/CSV
            analysis_report_file: Path to agent behavior analysis JSON
            model_comparison_file: Optional path to model comparison report
        """
        self.database_report_file = database_report_file
        self.analysis_report_file = analysis_report_file
        self.model_comparison_file = model_comparison_file

        # Load data
        self.database_metrics = self._load_database_metrics()
        self.analysis_results = self._load_analysis_results()
        self.model_comparison = self._load_model_comparison() if model_comparison_file else None

        # Recommendations
        self.recommendations = []

    def _load_database_metrics(self) -> Dict:
        """Load database metrics from CSV/JSON"""
        # Try JSON first
        json_path = self.database_report_file.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Fall back to CSV parsing (not implemented here - would need pandas)
        logger.warning(f"Database metrics JSON not found: {json_path}")
        return {}

    def _load_analysis_results(self) -> Dict:
        """Load agent behavior analysis results"""
        if not self.analysis_report_file.exists():
            logger.warning(f"Analysis file not found: {self.analysis_report_file}")
            return {}

        with open(self.analysis_report_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_model_comparison(self) -> Dict:
        """Load model comparison data"""
        if not self.model_comparison_file.exists():
            logger.warning(f"Model comparison file not found: {self.model_comparison_file}")
            return {}

        with open(self.model_comparison_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate(self) -> List[Dict]:
        """
        Generate all recommendations

        Returns:
            List of recommendation dictionaries
        """
        logger.info("="*80)
        logger.info("GENERATING IMPROVEMENT RECOMMENDATIONS")
        logger.info("="*80)

        # Category 1: Database-specific improvements
        self._generate_database_recommendations()

        # Category 2: Model selection improvements
        self._generate_model_recommendations()

        # Category 3: Agent workflow improvements
        self._generate_workflow_recommendations()

        # Category 4: Validation improvements
        self._generate_validation_recommendations()

        # Category 5: Performance optimizations
        self._generate_performance_recommendations()

        # Sort by priority
        self.recommendations.sort(key=lambda x: x['priority'], reverse=True)

        return self.recommendations

    def _add_recommendation(
        self,
        category: str,
        title: str,
        description: str,
        priority: int,
        evidence: Dict = None,
        action_items: List[str] = None
    ):
        """Add a recommendation to the list"""
        self.recommendations.append({
            'category': category,
            'title': title,
            'description': description,
            'priority': priority,  # 1-5, 5 = highest
            'evidence': evidence or {},
            'action_items': action_items or []
        })

    def _generate_database_recommendations(self):
        """Generate recommendations based on per-database performance"""
        logger.info("\n[1/5] Analyzing database-specific performance...")

        analysis_summary = self.analysis_results.get('analysis_summary', {})
        problematic_dbs = analysis_summary.get('problematic_databases', [])

        if problematic_dbs:
            # Identify databases with low success rates
            for db_info in problematic_dbs[:3]:  # Top 3 most problematic
                db_id = db_info['db_id']
                success_rate = db_info['success_rate']

                if success_rate < 50:
                    self._add_recommendation(
                        category="Database-Specific",
                        title=f"Critical: Low success rate on database '{db_id}'",
                        description=f"Database '{db_id}' has only {success_rate:.1f}% success rate. "
                                    f"This suggests the schema extraction or query decomposition may not be working well for this specific database structure.",
                        priority=5,
                        evidence={
                            'db_id': db_id,
                            'success_rate': success_rate,
                            'total_queries': db_info['total_queries'],
                            'failed_queries': db_info['failed_queries']
                        },
                        action_items=[
                            f"Manually inspect the schema for database '{db_id}'",
                            f"Check if schema embeddings are capturing the full schema",
                            f"Review example queries for this database in training data",
                            f"Consider adding database-specific prompt templates"
                        ]
                    )
                elif success_rate < 70:
                    self._add_recommendation(
                        category="Database-Specific",
                        title=f"Moderate: Below-average success on database '{db_id}'",
                        description=f"Database '{db_id}' has {success_rate:.1f}% success rate, below the overall average. "
                                    f"Schema complexity or domain-specific terminology may be causing issues.",
                        priority=3,
                        evidence={
                            'db_id': db_id,
                            'success_rate': success_rate
                        },
                        action_items=[
                            f"Review schema extraction results for '{db_id}'",
                            f"Check if examples are being retrieved correctly",
                            f"Analyze common failure patterns for this database"
                        ]
                    )

    def _generate_model_recommendations(self):
        """Generate recommendations based on SLM vs LLM performance"""
        logger.info("\n[2/5] Analyzing model selection strategy...")

        if not self.model_comparison:
            logger.info("  No model comparison data available - skipping")
            return

        # Compare SLM vs LLM metrics (if available)
        # This would require parsing the model comparison table
        # For now, we'll make recommendations based on fallback usage

        validation_failures = self.analysis_results.get('statistics', {}).get('validation_failure_patterns', [])
        fallback_triggers = sum(1 for f in validation_failures if f.get('fallback_triggered', False))

        total_queries = self.analysis_results.get('statistics', {}).get('total_queries', 1)
        fallback_rate = (fallback_triggers / total_queries * 100) if total_queries > 0 else 0

        if fallback_rate > 20:
            self._add_recommendation(
                category="Model Selection",
                title="High LLM fallback rate indicates SLM struggles",
                description=f"Fallback to LLM (gpt-4o) was triggered in {fallback_rate:.1f}% of queries. "
                            f"This suggests the SLM (llama3.1:8b) is frequently failing validation. "
                            f"Consider improving prompt templates or fine-tuning the SLM further.",
                priority=4,
                evidence={
                    'fallback_rate': fallback_rate,
                    'fallback_count': fallback_triggers,
                    'total_queries': total_queries
                },
                action_items=[
                    "Review validation errors that triggered fallback",
                    "Improve SLM prompt templates for SQL generation",
                    "Consider fine-tuning SLM on failed examples",
                    "Adjust fallback threshold if SLM is close to passing"
                ]
            )
        elif fallback_rate < 5:
            self._add_recommendation(
                category="Model Selection",
                title="SLM performing well - consider disabling fallback",
                description=f"Fallback to LLM only occurred in {fallback_rate:.1f}% of queries. "
                            f"The SLM is handling most queries successfully. You may be able to disable fallback "
                            f"to reduce costs and latency.",
                priority=2,
                evidence={
                    'fallback_rate': fallback_rate
                },
                action_items=[
                    "Test evaluation with fallback disabled (enable_fallback: false)",
                    "Compare EX/VES metrics with and without fallback",
                    "Monitor edge cases that might still benefit from fallback"
                ]
            )

    def _generate_workflow_recommendations(self):
        """Generate recommendations based on agent workflow execution"""
        logger.info("\n[3/5] Analyzing agent workflow...")

        problematic_nodes = self.analysis_results.get('analysis_summary', {}).get('problematic_nodes', [])

        for node_info in problematic_nodes:
            node_name = node_info['node_name']
            failure_rate = node_info['failure_rate']

            if failure_rate > 10:
                self._add_recommendation(
                    category="Agent Workflow",
                    title=f"Node '{node_name}' has high failure rate",
                    description=f"The '{node_name}' node is failing {failure_rate:.1f}% of the time. "
                                f"This indicates a systematic issue in this component.",
                    priority=4,
                    evidence={
                        'node_name': node_name,
                        'failure_rate': failure_rate,
                        'failures': node_info['failures'],
                        'executions': node_info['executions']
                    },
                    action_items=[
                        f"Review implementation of '{node_name}' node",
                        f"Check error logs for common failure patterns",
                        f"Consider adding retry logic or error recovery",
                        f"Validate input/output contracts for this node"
                    ]
                )

    def _generate_validation_recommendations(self):
        """Generate recommendations based on validation failures"""
        logger.info("\n[4/5] Analyzing validation patterns...")

        common_errors = self.analysis_results.get('analysis_summary', {}).get('common_validation_errors', [])

        if common_errors:
            top_error = common_errors[0]
            error_msg = top_error['error']
            count = top_error['count']

            self._add_recommendation(
                category="Validation",
                title=f"Most common validation error: '{error_msg[:50]}...'",
                description=f"This validation error occurred {count} times, indicating a systematic issue. "
                            f"Addressing this single error could significantly improve success rate.",
                priority=5,
                evidence={
                    'error_message': error_msg,
                    'occurrence_count': count
                },
                action_items=[
                    "Investigate root cause of this validation error",
                    "Update prompt templates to avoid this error pattern",
                    "Consider adding pre-validation checks before execution",
                    "Update validation logic if error is incorrectly flagged"
                ]
            )

        # Check regeneration effectiveness
        total_regenerations = self.analysis_results.get('statistics', {}).get('total_regenerations', 0)
        total_queries = self.analysis_results.get('statistics', {}).get('total_queries', 1)
        avg_regenerations = total_regenerations / total_queries if total_queries > 0 else 0

        if avg_regenerations > 1.5:
            self._add_recommendation(
                category="Validation",
                title="High regeneration count suggests validation is too strict",
                description=f"Average of {avg_regenerations:.2f} regenerations per query. "
                            f"This suggests either validation is too strict or the SQL generator needs improvement.",
                priority=3,
                evidence={
                    'avg_regenerations': avg_regenerations,
                    'total_regenerations': total_regenerations
                },
                action_items=[
                    "Review validation criteria - are they too strict?",
                    "Check if validation errors are actionable for regeneration",
                    "Improve SQL generator prompt to reduce validation failures",
                    "Consider relaxing non-critical validation checks"
                ]
            )

    def _generate_performance_recommendations(self):
        """Generate recommendations for performance optimization"""
        logger.info("\n[5/5] Analyzing performance metrics...")

        # Check latency from database metrics (if available)
        # This would require more detailed parsing of database_metrics

        # Check query complexity correlation
        success_rate = self.analysis_results.get('analysis_summary', {}).get('success_rate', 100)

        if success_rate < 60:
            self._add_recommendation(
                category="Performance",
                title="Overall accuracy below target - comprehensive review needed",
                description=f"Overall success rate is {success_rate:.1f}%, well below the 70-80% target. "
                            f"This suggests fundamental issues with the agent pipeline.",
                priority=5,
                evidence={
                    'success_rate': success_rate
                },
                action_items=[
                    "Review the entire agent workflow for systematic issues",
                    "Check if schema extraction is providing sufficient context",
                    "Validate that query decomposition is helping or hurting",
                    "Consider A/B testing with query decomposer disabled",
                    "Review prompt templates for all agent components"
                ]
            )

    def save_recommendations(self, output_file: Path = None):
        """
        Save recommendations to file

        Args:
            output_file: Path to save recommendations (default: auto-generated)
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"recommendations_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(self.recommendations),
            'recommendations': self.recommendations,
            'sources': {
                'database_metrics': str(self.database_report_file),
                'analysis_results': str(self.analysis_report_file),
                'model_comparison': str(self.model_comparison_file) if self.model_comparison_file else None
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"\nRecommendations saved to: {output_file}")

        # Also save markdown version
        md_file = output_file.with_suffix('.md')
        self._save_markdown_report(md_file)

        return output_file

    def _save_markdown_report(self, output_file: Path):
        """Save recommendations as markdown report"""
        lines = [
            "# SQL Agent Improvement Recommendations",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            f"## Summary",
            f"- Total Recommendations: {len(self.recommendations)}",
            ""
        ]

        # Group by category
        categories = {}
        for rec in self.recommendations:
            cat = rec['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rec)

        # Write by category
        for category, recs in sorted(categories.items()):
            lines.append(f"## {category}")
            lines.append("")

            for i, rec in enumerate(recs, 1):
                priority_stars = "⭐" * rec['priority']
                lines.append(f"### {i}. {rec['title']} {priority_stars}")
                lines.append("")
                lines.append(rec['description'])
                lines.append("")

                if rec.get('evidence'):
                    lines.append("**Evidence:**")
                    for key, value in rec['evidence'].items():
                        lines.append(f"- {key}: {value}")
                    lines.append("")

                if rec.get('action_items'):
                    lines.append("**Action Items:**")
                    for action in rec['action_items']:
                        lines.append(f"- [ ] {action}")
                    lines.append("")

                lines.append("---")
                lines.append("")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Markdown report saved to: {output_file}")

    def print_summary(self):
        """Print recommendations summary to console"""
        print("\n" + "="*80)
        print("IMPROVEMENT RECOMMENDATIONS SUMMARY")
        print("="*80)
        print(f"Total Recommendations: {len(self.recommendations)}\n")

        # Group by priority
        by_priority = {}
        for rec in self.recommendations:
            priority = rec['priority']
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(rec)

        for priority in sorted(by_priority.keys(), reverse=True):
            recs = by_priority[priority]
            priority_label = {5: "CRITICAL", 4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "INFO"}[priority]
            print(f"\n{priority_label} Priority ({len(recs)} items):")
            for rec in recs:
                print(f"  • [{rec['category']}] {rec['title']}")

        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate improvement recommendations from evaluation results"
    )

    parser.add_argument(
        "--database_metrics",
        type=str,
        required=True,
        help="Path to database metrics file (JSON or CSV)"
    )
    parser.add_argument(
        "--analysis_results",
        type=str,
        required=True,
        help="Path to agent behavior analysis JSON file"
    )
    parser.add_argument(
        "--model_comparison",
        type=str,
        default=None,
        help="Path to model comparison file (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for recommendations (default: auto-generated)"
    )

    args = parser.parse_args()

    # Create generator
    generator = RecommendationGenerator(
        database_report_file=Path(args.database_metrics),
        analysis_report_file=Path(args.analysis_results),
        model_comparison_file=Path(args.model_comparison) if args.model_comparison else None
    )

    # Generate recommendations
    generator.generate()

    # Print summary
    generator.print_summary()

    # Save to file
    output_file = Path(args.output) if args.output else None
    generator.save_recommendations(output_file)

    logger.info("\n[DONE] Recommendations generated!")


if __name__ == "__main__":
    main()
