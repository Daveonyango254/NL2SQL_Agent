"""
Enhanced SQL Validator
Multi-layered validation to catch incorrect queries before they pass through
"""

import re
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
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
        """Validate SQL structure and syntax"""
        results = []
        sql_upper = sql.upper().strip()

        # Check 1: Must start with valid SQL keyword
        valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
        if not any(sql_upper.startswith(kw) for kw in valid_starts):
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                check_name="valid_sql_start",
                message="SQL does not start with a valid keyword (SELECT, WITH, etc.)"
            ))
            return results  # No point continuing if not valid SQL

        # Check 2: SELECT must have FROM (unless it's a simple expression)
        if sql_upper.startswith('SELECT') and 'FROM' not in sql_upper:
            # Allow simple expressions like SELECT 1+1 or SELECT COUNT(*)
            if not re.search(r'SELECT\s+[\d\'\"\(\)]', sql_upper):
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="select_without_from",
                    message="SELECT query without FROM clause - might be incomplete"
                ))

        # Check 3: Check for balanced parentheses
        open_count = sql.count('(')
        close_count = sql.count(')')
        if open_count != close_count:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                check_name="unbalanced_parentheses",
                message=f"Unbalanced parentheses: {open_count} open, {close_count} close"
            ))

        # Check 4: Check for unclosed quotes
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                check_name="unclosed_quotes",
                message="Unclosed single quotes in SQL"
            ))

        # Check 5: JOIN without ON (common mistake)
        if re.search(r'\bJOIN\b', sql_upper) and 'ON' not in sql_upper:
            # Check if it's CROSS JOIN or NATURAL JOIN (which don't need ON)
            if not re.search(r'(CROSS|NATURAL)\s+JOIN', sql_upper):
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="join_without_on",
                    message="JOIN without ON clause - might cause cartesian product"
                ))

        # Check 6: GROUP BY without aggregate function
        if 'GROUP BY' in sql_upper:
            aggregates = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP_CONCAT']
            has_aggregate = any(agg in sql_upper for agg in aggregates)
            if not has_aggregate:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="group_by_without_aggregate",
                    message="GROUP BY without aggregate function - results may be unexpected"
                ))

        # Check 7: LIMIT without ORDER BY (non-deterministic)
        if 'LIMIT' in sql_upper and 'ORDER BY' not in sql_upper:
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                check_name="limit_without_order",
                message="LIMIT without ORDER BY - results may be non-deterministic"
            ))

        if not results:
            results.append(ValidationResult(
                level=ValidationLevel.PASS,
                check_name="structure",
                message="SQL structure validation passed"
            ))

        return results

    def validate_schema(self, sql: str) -> List[ValidationResult]:
        """Validate that tables and columns exist in database"""
        results = []

        if not self.db_path:
            return results

        try:
            # Get schema if not cached
            if not self.schema_cache:
                self._load_schema()

            # Extract table names from SQL
            tables_in_sql = self._extract_tables(sql)

            # Check each table exists
            for table in tables_in_sql:
                if table.lower() not in [t.lower() for t in self.schema_cache.keys()]:
                    results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        check_name="table_exists",
                        message=f"Table '{table}' does not exist in database",
                        details={"table": table, "available_tables": list(self.schema_cache.keys())}
                    ))

            # Extract and check columns (basic check)
            columns_in_sql = self._extract_columns(sql)
            all_columns = set()
            for cols in self.schema_cache.values():
                all_columns.update(c.lower() for c in cols)

            for col in columns_in_sql:
                # Skip if it's a table alias or function
                if col.lower() not in all_columns and not self._is_sql_keyword(col):
                    # This is a soft warning - column might be aliased
                    results.append(ValidationResult(
                        level=ValidationLevel.INFO,
                        check_name="column_exists",
                        message=f"Column '{col}' not found in schema (might be aliased)"
                    ))

        except Exception as e:
            results.append(ValidationResult(
                level=ValidationLevel.INFO,
                check_name="schema_check_error",
                message=f"Could not validate schema: {str(e)}"
            ))

        if not results:
            results.append(ValidationResult(
                level=ValidationLevel.PASS,
                check_name="schema",
                message="Schema validation passed"
            ))

        return results

    def validate_semantic(
        self,
        sql: str,
        question: str,
        schema_context: Dict = None
    ) -> List[ValidationResult]:
        """Validate that SQL semantically matches the question"""
        results = []
        question_lower = question.lower()
        sql_upper = sql.upper()

        # Check 1: Question asks "how many" -> should have COUNT
        count_phrases = ['how many', 'count of', 'number of', 'total number']
        if any(phrase in question_lower for phrase in count_phrases):
            if 'COUNT' not in sql_upper:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="count_expected",
                    message="Question asks 'how many' but SQL doesn't use COUNT()"
                ))

        # Check 2: Question asks for "average" -> should have AVG
        if 'average' in question_lower or 'avg' in question_lower:
            if 'AVG' not in sql_upper:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="avg_expected",
                    message="Question asks for 'average' but SQL doesn't use AVG()"
                ))

        # Check 3: Question asks for "sum" or "total" -> should have SUM
        if 'sum' in question_lower or 'total' in question_lower:
            if 'SUM' not in sql_upper and 'COUNT' not in sql_upper:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="sum_expected",
                    message="Question asks for 'sum/total' but SQL doesn't use SUM()"
                ))

        # Check 4: Question asks for "top N" or "first N" -> should have LIMIT or BETWEEN
        top_match = re.search(r'(top|first|bottom|last)\s+(\d+)', question_lower)
        if top_match:
            expected_limit = top_match.group(2)
            has_limit = 'LIMIT' in sql_upper
            has_between = 'BETWEEN' in sql_upper

            if not has_limit and not has_between:
                # This is a semantic warning - regeneration may not help with weak models
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="limit_expected",
                    message=f"Question asks for '{top_match.group(1)} {expected_limit}' but SQL doesn't use LIMIT or BETWEEN"
                ))
            elif has_limit:
                # Check if LIMIT value matches (only if using LIMIT, not BETWEEN)
                limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
                if limit_match and limit_match.group(1) != expected_limit:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        check_name="limit_mismatch",
                        message=f"Question asks for {top_match.group(1)} {expected_limit} but LIMIT is {limit_match.group(1)}"
                    ))

        # Check 5: Question asks for "list" or "all" or "find" -> should not have restrictive LIMIT
        list_phrases = ['list all', 'show all', 'find all', 'what are all', 'find the']
        if any(phrase in question_lower for phrase in list_phrases):
            # Check for artificial LIMIT that wasn't asked for
            limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
            if limit_match:
                # Only warn for small limits that seem artificial
                limit_val = int(limit_match.group(1))
                if limit_val <= 10 and not top_match:  # No "top N" in question
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        check_name="limit_unexpected",
                        message=f"Question asks to 'find/list' but SQL has LIMIT {limit_val} which may truncate results"
                    ))

        # Check 6: Question asks for "percentage" or "percent" -> should have calculation
        if 'percentage' in question_lower or 'percent' in question_lower:
            if '*' not in sql and '/' not in sql and '100' not in sql:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="percentage_expected",
                    message="Question asks for percentage but SQL doesn't appear to calculate it"
                ))

        # Check 7: Question mentions specific values -> should be in WHERE
        # Extract quoted values from question
        quoted_values = re.findall(r'["\']([^"\']+)["\']', question)
        for value in quoted_values:
            if value.lower() not in sql.lower():
                results.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    check_name="value_not_in_sql",
                    message=f"Question mentions '{value}' but not found in SQL"
                ))

        # Check 8: Question asks for "no X" or "not X" -> should have negation in WHERE
        negation_patterns = [
            r'\bno\s+(\w+)',           # "no bromine"
            r'\bnot\s+(\w+)',          # "not carcinogenic"
            r'\bwithout\s+(\w+)',      # "without sodium"
            r'\bexclude\s+(\w+)',      # "exclude molecules"
            r'\bisn\'t\s+(\w+)',       # "isn't carcinogenic"
            r'\bare\s+not\s+(\w+)',    # "are not single-bonded"
        ]
        for pattern in negation_patterns:
            neg_match = re.search(pattern, question_lower)
            if neg_match:
                # Check if SQL has negation operator
                has_negation = any(op in sql_upper for op in ['<>', '!=', 'NOT ', 'NOT(', 'IS NOT'])
                if not has_negation:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        check_name="negation_expected",
                        message=f"Question asks for 'no/not {neg_match.group(1)}' but SQL lacks negation (<>, !=, NOT)"
                    ))
                break  # Only report once

        if not results:
            results.append(ValidationResult(
                level=ValidationLevel.PASS,
                check_name="semantic",
                message="Semantic validation passed"
            ))

        return results

    def validate_results(
        self,
        sql: str,
        question: str,
        results: Any
    ) -> List[ValidationResult]:
        """Validate query results make sense"""
        validation_results = []
        question_lower = question.lower()
        sql_upper = sql.upper()

        # Handle different result types
        if results is None:
            validation_results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                check_name="null_results",
                message="Query returned NULL results"
            ))
            return validation_results

        # Convert to list if needed
        if isinstance(results, str):
            return validation_results  # Can't validate string results

        result_list = list(results) if not isinstance(results, list) else results

        # Check 1: Empty results
        if len(result_list) == 0:
            # Empty might be valid for some questions
            if any(word in question_lower for word in ['any', 'exist', 'is there']):
                validation_results.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    check_name="empty_results",
                    message="Query returned 0 rows (might be valid for existence check)"
                ))
            else:
                validation_results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="empty_results",
                    message="Query returned 0 rows - might indicate incorrect query"
                ))

        # Check 2: COUNT query should return single numeric value
        if 'COUNT' in sql_upper and 'GROUP BY' not in sql_upper:
            if len(result_list) != 1:
                validation_results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    check_name="count_multiple_rows",
                    message=f"COUNT query returned {len(result_list)} rows instead of 1"
                ))
            elif result_list and result_list[0]:
                value = result_list[0][0] if isinstance(result_list[0], (list, tuple)) else result_list[0]
                if value is not None and not isinstance(value, (int, float)):
                    validation_results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        check_name="count_non_numeric",
                        message=f"COUNT query returned non-numeric value: {value}"
                    ))

        # Check 3: "How many" question should return reasonable count
        if 'how many' in question_lower and result_list:
            try:
                value = result_list[0][0] if isinstance(result_list[0], (list, tuple)) else result_list[0]
                if isinstance(value, (int, float)):
                    if value < 0:
                        validation_results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            check_name="negative_count",
                            message=f"Count query returned negative value: {value}"
                        ))
                    elif value > 1000000:
                        validation_results.append(ValidationResult(
                            level=ValidationLevel.INFO,
                            check_name="large_count",
                            message=f"Count returned very large value: {value}"
                        ))
            except (IndexError, TypeError):
                pass

        # Check 4: LIMIT N should return at most N rows
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if len(result_list) > limit_value:
                validation_results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    check_name="limit_exceeded",
                    message=f"Query has LIMIT {limit_value} but returned {len(result_list)} rows"
                ))

        if not validation_results:
            validation_results.append(ValidationResult(
                level=ValidationLevel.PASS,
                check_name="results",
                message="Result validation passed"
            ))

        return validation_results

    def _load_schema(self):
        """Load database schema into cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                self.schema_cache[table] = columns

            conn.close()
        except Exception:
            pass

    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL"""
        tables = []
        sql_upper = sql.upper()

        # Match FROM and JOIN clauses
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)',
            r'UPDATE\s+(\w+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql_upper)
            tables.extend(matches)

        return list(set(tables))

    def _extract_columns(self, sql: str) -> List[str]:
        """Extract column names from SQL (basic extraction)"""
        columns = []

        # This is a simplified extraction - won't catch everything
        # Match word.word patterns (table.column)
        table_col = re.findall(r'(\w+)\.(\w+)', sql)
        columns.extend([col for _, col in table_col])

        return list(set(columns))

    def _is_sql_keyword(self, word: str) -> bool:
        """Check if word is a SQL keyword"""
        keywords = {
            'select', 'from', 'where', 'join', 'on', 'and', 'or', 'not',
            'in', 'like', 'between', 'is', 'null', 'as', 'order', 'by',
            'group', 'having', 'limit', 'offset', 'union', 'all', 'distinct',
            'count', 'sum', 'avg', 'max', 'min', 'case', 'when', 'then',
            'else', 'end', 'asc', 'desc', 'inner', 'left', 'right', 'outer',
            'cross', 'natural', 'using', 'exists', 'any', 'some'
        }
        return word.lower() in keywords


def validate_sql_query(
    sql: str,
    question: str,
    db_path: str = None,
    schema_context: Dict = None,
    results: Any = None
) -> Tuple[bool, List[ValidationResult], float]:
    """
    Convenience function to validate SQL query

    Returns:
        Tuple of (is_valid, validation_results, confidence_adjustment)
    """
    validator = SQLValidator(db_path)
    return validator.validate_all(sql, question, schema_context, results)


def get_validation_summary(results: List[ValidationResult]) -> str:
    """Get a human-readable summary of validation results"""
    errors = [r for r in results if r.level == ValidationLevel.ERROR]
    warnings = [r for r in results if r.level == ValidationLevel.WARNING]

    if errors:
        return f"FAILED: {len(errors)} errors, {len(warnings)} warnings"
    elif warnings:
        return f"PASSED with {len(warnings)} warnings"
    else:
        return "PASSED all validations"
