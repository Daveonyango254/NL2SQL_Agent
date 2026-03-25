"""
Configuration for evaluation scripts
"""

from pathlib import Path

# Base directory (parent of evaluation folder)
BASE_DIR = Path(__file__).parent.parent

# BIRD dataset paths
BIRD_DB_PATH = BASE_DIR / "data" / "bird" / "dev_databases"
BIRD_DEV_JSON = BASE_DIR / "data" / "bird" / \
    "sample_dev(testing).json"   # or "sample_dev(testing).json" for smaller set & dev,json for full set
# Uncoment for errors evaluation
# BIRD_DEV_JSON = BASE_DIR / "evaluation" / "output" / "test1" / \
# "pipeline_20251121_125835_predictions_errors.json"

# Output directory - ALL results saved here
OUTPUT_DIR = BASE_DIR / "evaluation" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Spider dataset paths (if needed)
SPIDER_DB_PATH = BASE_DIR / "data" / "spider" / "database"
SPIDER_DEV_JSON = BASE_DIR / "data" / "spider" / "dev.json"

# Evaluation Settings
DEFAULT_QUERY_TIMEOUT = 120.0  # Default timeout per query in seconds (2 minutes)
ENABLE_LATENCY_TRACKING = True  # Track latency statistics (p95, p99)
SAVE_LATENCY_DETAILS = True  # Save detailed per-query latency to file
