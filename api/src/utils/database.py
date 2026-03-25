"""
Database utility functions
Handles database discovery, path resolution, and example loading
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# Base directory configuration
def get_base_dir() -> Path:
    """Get base directory (project root)"""
    # Go up from api/src/utils to project root
    return Path(__file__).parent.parent.parent.parent.resolve()


# Path configuration
BIRD_DB_PATH = get_base_dir() / "data" / "bird" / "dev_databases"
BIRD_EXAMPLES_PATH = get_base_dir() / "data" / "bird" / "dev_2_examples.json"


def discover_databases(base_path: Path = BIRD_DB_PATH) -> Dict[str, Path]:
    """Discover all available databases in BIRD dataset structure"""
    databases = {}
    if base_path.exists():
        for db_dir in base_path.iterdir():
            if db_dir.is_dir():
                sqlite_file = db_dir / f"{db_dir.name}.sqlite"
                if sqlite_file.exists():
                    databases[db_dir.name] = sqlite_file
    return databases


def get_database_path(db_id: str, base_path: Path = BIRD_DB_PATH) -> Optional[Path]:
    """Get database path for given db_id"""
    db_path = base_path / db_id / f"{db_id}.sqlite"
    return db_path if db_path.exists() else None


def get_database_csv_paths(db_id: str, base_path: Path = BIRD_DB_PATH) -> List[Path]:
    """Get all CSV file paths in the database_description directory"""
    csv_dir = base_path / db_id / "database_description"
    if csv_dir.exists():
        return list(csv_dir.glob("*.csv"))
    return []


def load_database_examples(db_id: str, examples_path: Path = BIRD_EXAMPLES_PATH,
                           config: Dict = None) -> List[Dict]:
    """Load example queries for few-shot learning"""
    try:
        if examples_path.exists():
            with open(examples_path, 'r', encoding='utf-8') as f:
                all_examples = json.load(f)

            db_examples = [
                ex for ex in all_examples if ex.get('db_id') == db_id]

            # Sort by difficulty
            difficulty_order = {"simple": 0, "moderate": 1, "challenging": 2}
            db_examples.sort(key=lambda x: difficulty_order.get(
                x.get('difficulty', 'moderate'), 1))

            return db_examples
    except Exception as e:
        if config and config.get('features', {}).get('enable_debug_output'):
            print(f"Warning: Could not load examples: {e}")

    return []
