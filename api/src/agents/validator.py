"""
Validator Agent
Executes SQL queries and validates results with multi-layer validation
"""

import sqlite3
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"      # Query is definitely wrong, regenerate
    WARNING = "warning"  # Query might be wrong, lower confidence
    INFO = "info"        # Informational, no action needed
    PASS = "pass"        # Validation passed


@dataclass
class ValidationResult:
    """Result of a validation check"""
    level: ValidationLevel
    check_name: str
    message: str
    details: Optional[Dict] = None


class SQLValidator:
    """
    Multi-layered SQL validation:
    1. Structural - SQL syntax and structure
    2. Schema - Tables and columns exist
    3. Semantic - Query addresses the question
    4. Result - Output sanity checks
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.schema_cache = {}

    def validate_all(
        self,
        sql: str,
        question: str,
        schema_context: Dict = None,
        results: Any = None
    ) -> Tuple[bool, List[ValidationResult], float]:
        """
        Run all validations and return overall result

        Args:
            sql: SQL query to validate
            question: Original natural language question
            schema_context: Schema information from RAG
            results: Query execution results (if available)

        Returns:
            Tuple of (is_valid, validation_results, confidence_adjustment)
        """
        all_results = []
        confidence_adjustment = 0.0

        # 1. Structural validation
        structural = self.validate_structure(sql)
        all_results.extend(structural)

        # 2. Schema validation (if db_path available)
        if self.db_path:
            schema = self.validate_schema(sql)
            all_results.extend(schema)

        # 3. Semantic validation
        semantic = self.validate_semantic(sql, question, schema_context)
        all_results.extend(semantic)

        # 4. Result validation (if results available)
        if results is not None:
            result_checks = self.validate_results(sql, question, results)
            all_results.extend(result_checks)

        # Calculate overall result
        has_errors = any(r.level == ValidationLevel.ERROR for r in all_results)
        warning_count = sum(1 for r in all_results if r.level == ValidationLevel.WARNING)

        # Adjust confidence based on warnings
        confidence_adjustment = -0.1 * warning_count

        is_valid = not has_errors
        return is_valid, all_results, confidence_adjustment

    def validate_structure(self, sql: str) -> List[ValidationResult]:
        """Validate SQL structure and syntax - basic checks"""
        results = []
        sql_upper = sql.upper().strip()

        # Check: Must start with valid SQL keyword
        valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
        if not any(sql_upper.startswith(kw) for kw in valid_starts):
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                check_name="valid_start",
                message=f"Query must start with valid SQL keyword, got: {sql[:20]}..."
            ))

        # Check: Balanced parentheses
        if sql.count('(') != sql.count(')'):
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                check_name="balanced_parens",
                message="Unbalanced parentheses in query"
            ))

        # Check: Has FROM clause for SELECT
        if sql_upper.startswith('SELECT') and ' FROM ' not in sql_upper:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                check_name="missing_from",
                message="SELECT query missing FROM clause"
            ))

        return results

    def validate_schema(self, sql: str) -> List[ValidationResult]:
        """Validate tables and columns exist in database"""
        results = []

        if not self.db_path:
            return results

        try:
            # Get actual schema from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0].lower() for row in cursor.fetchall()}

            # Extract table names from SQL (simple regex-based extraction)
            import re
            sql_upper = sql.upper()

            # Extract FROM and JOIN table references
            from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'

            tables_in_sql = set()
            tables_in_sql.update(re.findall(from_pattern, sql, re.IGNORECASE))
            tables_in_sql.update(re.findall(join_pattern, sql, re.IGNORECASE))

            # Check if tables exist
            for table in tables_in_sql:
                if table.lower() not in valid_tables:
                    results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        check_name="table_exists",
                        message=f"Table '{table}' does not exist in database",
                        details={"table": table, "valid_tables": list(valid_tables)}
                    ))

            conn.close()

        except Exception as e:
            # Don't fail validation on schema check errors, just log
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                check_name="schema_check_error",
                message=f"Could not validate schema: {str(e)}"
            ))

        return results

    def validate_semantic(self, sql: str, question: str, schema_context: Dict = None) -> List[ValidationResult]:
        """Validate query semantically matches the question"""
        results = []
        question_lower = question.lower()
        sql_upper = sql.upper()

        # Check 1: Count questions should return single aggregate value
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            if 'COUNT' not in sql_upper and 'SUM' not in sql_upper:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="count_query_missing_aggregate",
                    message="Question asks for count but SQL missing COUNT/SUM aggregate"
                ))

        # Check 2: Average/mean questions should use AVG
        if any(word in question_lower for word in ['average', 'mean', 'avg']):
            if 'AVG' not in sql_upper:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="average_query_missing_avg",
                    message="Question asks for average but SQL missing AVG function"
                ))

        # Check 3: List/show questions shouldn't have COUNT (unless asking for count)
        if any(word in question_lower for word in ['list', 'show', 'display', 'what are']):
            if 'count' not in question_lower and 'COUNT(*)' in sql_upper:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="list_query_has_count",
                    message="Question asks to list items but SQL uses COUNT(*)"
                ))

        # Check 4: Top N questions should have LIMIT
        import re
        top_match = re.search(r'top\s+(\d+)', question_lower)
        if top_match:
            limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
            if not limit_match:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="top_n_missing_limit",
                    message=f"Question asks for top {top_match.group(1)} but SQL missing LIMIT clause"
                ))

        return results

    def validate_results(self, sql: str, question: str, results: Any) -> List[ValidationResult]:
        """Validate query results make sense"""
        validations = []

        # Check: Empty results
        if isinstance(results, list) and len(results) == 0:
            validations.append(ValidationResult(
                level=ValidationLevel.WARNING,
                check_name="empty_results",
                message="Query returned 0 rows - might be incorrect"
            ))

        return validations


def execute_and_validate(
    sql_query: str,
    user_query: str,
    db_path: str,
    schema_context: Dict,
    config: Dict
) -> Tuple[bool, Any, List[ValidationResult], float]:
    """
    Execute SQL and validate results with enhanced multi-layer validation

    Args:
        sql_query: SQL query to execute
        user_query: Original user question
        db_path: Path to database
        schema_context: Schema context
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, results, validation_results, confidence_adjustment)
    """
    validator = SQLValidator(db_path=db_path)

    # Step 1: Pre-execution validation (structural + semantic)
    is_valid_pre, pre_results, confidence_adj = validator.validate_all(
        sql=sql_query,
        question=user_query,
        schema_context=schema_context,
        results=None
    )

    # Check for critical errors before execution
    pre_errors = [r for r in pre_results if r.level == ValidationLevel.ERROR]
    if pre_errors:
        return False, None, pre_results, confidence_adj

    # Step 2: Execute the query
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(sql_query)
        results = cursor.fetchall()
        conn.close()

        # Step 3: Post-execution validation (result checks)
        is_valid_post, post_results, post_confidence_adj = validator.validate_all(
            sql=sql_query,
            question=user_query,
            schema_context=schema_context,
            results=results
        )

        # Combine confidence adjustments
        total_confidence_adj = confidence_adj + post_confidence_adj

        # Check for post-execution errors
        post_errors = [r for r in post_results if r.level == ValidationLevel.ERROR]
        post_warnings = [r for r in post_results if r.level == ValidationLevel.WARNING]

        if post_errors:
            return False, results, post_results, total_confidence_adj
        elif len(post_warnings) >= 3:
            # Too many warnings - likely incorrect
            return False, results, post_results, total_confidence_adj
        else:
            return True, results, post_results, total_confidence_adj

    except Exception as e:
        error_result = ValidationResult(
            level=ValidationLevel.ERROR,
            check_name="execution_error",
            message=f"Execution error: {str(e)}"
        )
        return False, None, [error_result], -0.5


def get_validation_summary(results: List[ValidationResult]) -> str:
    """Get human-readable validation summary"""
    if not results:
        return "No validation checks"

    error_count = sum(1 for r in results if r.level == ValidationLevel.ERROR)
    warning_count = sum(1 for r in results if r.level == ValidationLevel.WARNING)

    if error_count > 0:
        return f"{error_count} errors"
    elif warning_count > 0:
        return f"{warning_count} warnings"
    else:
        return "Passed"
