"""
Latency Tracker for Evaluation
Tracks query latencies and calculates percentiles (p50, p95, p99)
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime


class LatencyTracker:
    """Track query latencies and calculate statistics"""

    def __init__(self):
        """Initialize latency tracker"""
        self.latencies = []  # List of (query_id, latency_ms, is_timeout)
        self.query_details = {}  # Dict mapping query_id to details

    def record(
        self,
        query_id: int,
        latency_ms: float,
        is_timeout: bool = False,
        success: bool = True,
        error_msg: Optional[str] = None
    ):
        """
        Record a query latency

        Args:
            query_id: Query identifier
            latency_ms: Latency in milliseconds
            is_timeout: Whether query timed out
            success: Whether query executed successfully
            error_msg: Error message if failed
        """
        self.latencies.append((query_id, latency_ms, is_timeout))
        self.query_details[query_id] = {
            "latency_ms": latency_ms,
            "is_timeout": is_timeout,
            "success": success,
            "error_msg": error_msg
        }

    def calculate_percentiles(self) -> Dict[str, float]:
        """
        Calculate latency percentiles

        Returns:
            Dictionary with p50, p95, p99, mean, min, max
        """
        if not self.latencies:
            return {
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0
            }

        # Extract latencies (excluding timeouts for percentile calculation)
        non_timeout_latencies = [
            lat for _, lat, is_timeout in self.latencies if not is_timeout
        ]

        # If all timed out, use all latencies
        if not non_timeout_latencies:
            non_timeout_latencies = [lat for _, lat, _ in self.latencies]

        return {
            "p50_ms": float(np.percentile(non_timeout_latencies, 50)),
            "p95_ms": float(np.percentile(non_timeout_latencies, 95)),
            "p99_ms": float(np.percentile(non_timeout_latencies, 99)),
            "mean_ms": float(np.mean(non_timeout_latencies)),
            "min_ms": float(np.min(non_timeout_latencies)),
            "max_ms": float(np.max(non_timeout_latencies))
        }

    def get_timeout_stats(self) -> Dict[str, any]:
        """
        Get timeout statistics

        Returns:
            Dictionary with timeout count, percentage, and timeout IDs
        """
        total_queries = len(self.latencies)
        timeout_queries = [qid for qid, _, is_timeout in self.latencies if is_timeout]

        return {
            "total_queries": total_queries,
            "timeout_count": len(timeout_queries),
            "timeout_percentage": (len(timeout_queries) / total_queries * 100) if total_queries > 0 else 0.0,
            "timeout_query_ids": timeout_queries
        }

    def get_summary(self) -> Dict:
        """
        Get complete latency summary

        Returns:
            Dictionary with all latency statistics
        """
        percentiles = self.calculate_percentiles()
        timeout_stats = self.get_timeout_stats()

        return {
            **percentiles,
            **timeout_stats,
            "timestamp": datetime.now().isoformat()
        }

    def save_to_file(self, output_path: Path):
        """
        Save latency details to JSON file

        Args:
            output_path: Path to save latency details
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "per_query_latencies": [
                {
                    "query_id": qid,
                    "latency_ms": lat,
                    "is_timeout": is_timeout
                }
                for qid, lat, is_timeout in self.latencies
            ],
            "query_details": self.query_details
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print latency summary to console"""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("LATENCY STATISTICS")
        print("="*80)
        print(f"p50 (median):  {summary['p50_ms']:>10,.1f} ms")
        print(f"p95:           {summary['p95_ms']:>10,.1f} ms")
        print(f"p99:           {summary['p99_ms']:>10,.1f} ms")
        print(f"Mean:          {summary['mean_ms']:>10,.1f} ms")
        print(f"Min:           {summary['min_ms']:>10,.1f} ms")
        print(f"Max:           {summary['max_ms']:>10,.1f} ms")
        print(f"\nTimeouts:      {summary['timeout_count']:>5} / {summary['total_queries']} ({summary['timeout_percentage']:.2f}%)")
        print("="*80)

    def get_slowest_queries(self, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get the N slowest queries (excluding timeouts)

        Args:
            n: Number of slowest queries to return

        Returns:
            List of (query_id, latency_ms) tuples
        """
        non_timeout_queries = [
            (qid, lat) for qid, lat, is_timeout in self.latencies if not is_timeout
        ]

        # Sort by latency descending
        sorted_queries = sorted(non_timeout_queries, key=lambda x: x[1], reverse=True)

        return sorted_queries[:n]

    def get_fastest_queries(self, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get the N fastest queries

        Args:
            n: Number of fastest queries to return

        Returns:
            List of (query_id, latency_ms) tuples
        """
        non_timeout_queries = [
            (qid, lat) for qid, lat, is_timeout in self.latencies if not is_timeout
        ]

        # Sort by latency ascending
        sorted_queries = sorted(non_timeout_queries, key=lambda x: x[1])

        return sorted_queries[:n]


def combine_latency_trackers(trackers: List[LatencyTracker]) -> LatencyTracker:
    """
    Combine multiple latency trackers into one

    Args:
        trackers: List of LatencyTracker instances

    Returns:
        Combined LatencyTracker
    """
    combined = LatencyTracker()

    for tracker in trackers:
        combined.latencies.extend(tracker.latencies)
        combined.query_details.update(tracker.query_details)

    return combined
