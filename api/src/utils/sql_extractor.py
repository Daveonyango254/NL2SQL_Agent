"""
SQL Extraction Utility
Robust extraction of SQL queries from verbose LLM outputs
Handles various formats including markdown, explanations, and reasoning tags
"""

import re
from typing import Optional


def extract_sql(text: str) -> str:
    """
    Extract clean SQL query from potentially verbose LLM output.
    Handles multiple formats:
    - Markdown code blocks (```sql ... ```)
    - <think> tags from reasoning models
    - Explanatory text before/after SQL
    - Multi-line SQL statements

    Args:
        text: Raw LLM output that may contain SQL

    Returns:
        Cleaned SQL query string
    """
    if not text:
        return ""

    # Step 1: Remove <think> tags and their contents (reasoning models)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^<think>|</think>$', '', text).strip()

    # Step 2: Try to extract from markdown code blocks first
    sql = _extract_from_markdown(text)
    if sql:
        return clean_sql(sql)

    # Step 3: Try to find SQL statement in text
    sql = _extract_sql_statement(text)
    if sql:
        return clean_sql(sql)

    # Step 4: If no SQL found, return cleaned original text
    return clean_sql(text)


def _extract_from_markdown(text: str) -> Optional[str]:
    """Extract SQL from markdown code blocks"""
    # Try ```sql ... ``` format
    match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ``` format
    match = re.search(r'```\s*(SELECT|WITH|INSERT|UPDATE|DELETE).*?```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).replace('```', '').strip()

    return None


def _extract_sql_statement(text: str) -> Optional[str]:
    """Extract SQL statement from text using keyword detection"""
    lines = text.split('\n')
    sql_lines = []
    in_sql = False
    paren_depth = 0

    # SQL keywords that can start a statement
    start_keywords = r'^(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b'
    # SQL keywords that continue a statement
    continue_keywords = r'^(FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|FULL|CROSS|NATURAL|ON|AND|OR|GROUP|ORDER|HAVING|LIMIT|UNION|INTERSECT|EXCEPT|AS|SET|VALUES|INTO)\b'

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_sql and sql_lines:
                # Empty line might indicate end of SQL
                continue
            continue

        # Check if line starts with SQL keyword
        if re.match(start_keywords, stripped, re.IGNORECASE):
            in_sql = True
            sql_lines = [line]  # Reset and start fresh
            paren_depth = stripped.count('(') - stripped.count(')')
        elif in_sql:
            # Check if this looks like SQL continuation
            if (re.match(continue_keywords, stripped, re.IGNORECASE) or
                stripped.startswith('(') or
                stripped.startswith(')') or
                stripped.endswith(',') or
                stripped.endswith('(') or
                paren_depth > 0 or
                _looks_like_sql(stripped)):

                sql_lines.append(line)
                paren_depth += stripped.count('(') - stripped.count(')')

                # Check for statement end
                if stripped.endswith(';') and paren_depth <= 0:
                    break
            else:
                # Doesn't look like SQL continuation
                if sql_lines and _is_valid_sql('\n'.join(sql_lines)):
                    break
                # Otherwise, might be explanation - skip

    if sql_lines:
        sql = '\n'.join(sql_lines)
        if _is_valid_sql(sql):
            return sql

    # Fallback: Try regex to find SELECT statement
    match = re.search(
        r'\b(SELECT\s+.*?(?:FROM\s+\w+.*?)?(?:;|$))',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1)

    return None


def _looks_like_sql(line: str) -> bool:
    """Check if a line looks like SQL (heuristic)"""
    sql_patterns = [
        r'\b(SELECT|FROM|WHERE|JOIN|AND|OR|ON|GROUP BY|ORDER BY|LIMIT|COUNT|SUM|AVG|MAX|MIN|DISTINCT|AS|IN|NOT|NULL|LIKE|BETWEEN)\b',
        r'\w+\s*=\s*[\'\"]',  # column = 'value'
        r'\w+\s*[<>=!]+\s*\d+',  # column > 10
        r'\w+\.\w+',  # table.column
    ]
    for pattern in sql_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def _is_valid_sql(sql: str) -> bool:
    """Basic validation that string looks like SQL"""
    sql_upper = sql.upper().strip()
    return any(sql_upper.startswith(kw) for kw in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER'])


def clean_sql(sql: str) -> str:
    """
    Clean and normalize SQL query

    Args:
        sql: Raw SQL string

    Returns:
        Cleaned SQL string
    """
    if not sql:
        return ""

    # Remove markdown formatting
    sql = re.sub(r'^```sql\s*', '', sql, flags=re.MULTILINE | re.IGNORECASE)
    sql = re.sub(r'^```\s*', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'```\s*$', '', sql, flags=re.MULTILINE)

    # Remove <think> tags
    sql = re.sub(r'<think>.*?</think>', '', sql, flags=re.DOTALL)
    sql = re.sub(r'</?think>', '', sql)

    # Remove common prefixes from verbose outputs
    prefixes_to_remove = [
        r'^(Here\'s?\s+(the\s+)?SQL\s*(query)?:?\s*)',
        r'^(The\s+SQL\s*(query)?\s*(is|would be):?\s*)',
        r'^(SQL\s*(Query)?:?\s*)',
        r'^(Query:?\s*)',
        r'^(Answer:?\s*)',
        r'^(Result:?\s*)',
    ]
    for prefix in prefixes_to_remove:
        sql = re.sub(prefix, '', sql, flags=re.IGNORECASE | re.MULTILINE)

    # Remove trailing explanations (after semicolon)
    if ';' in sql:
        # Find the last semicolon that's part of the SQL
        parts = sql.split(';')
        sql_parts = []
        for i, part in enumerate(parts[:-1]):
            sql_parts.append(part)
            # Check if next part looks like explanation
            if i + 1 < len(parts):
                next_part = parts[i + 1].strip()
                if next_part and not _looks_like_sql(next_part.split('\n')[0]):
                    break
        sql = ';'.join(sql_parts) + ';'

    # Clean whitespace
    sql = sql.strip()

    # Ensure semicolon at end
    if sql and not sql.endswith(';'):
        sql += ';'

    # Final validation - if doesn't start with SQL keyword, try to find it
    if sql and not _is_valid_sql(sql):
        match = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*', sql, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(0)
            if not sql.endswith(';'):
                sql += ';'

    return sql


def extract_sql_for_evaluation(agent_output: str) -> str:
    """
    Extract SQL specifically for evaluation purposes.
    More aggressive cleaning for BIRD benchmark evaluation.

    Args:
        agent_output: Full agent output (may include formatting)

    Returns:
        Clean SQL for evaluation
    """
    sql = extract_sql(agent_output)

    # Additional cleaning for evaluation
    # Remove any remaining formatting characters
    sql = sql.replace('**', '')
    sql = re.sub(r'\n+', ' ', sql)  # Single line
    sql = re.sub(r'\s+', ' ', sql)  # Normalize spaces
    sql = sql.strip()

    # Ensure ends with semicolon
    if sql and not sql.endswith(';'):
        sql += ';'

    return sql
