"""
SQL Generator Prompt Template
Generates precise SQL queries from decomposed plans and evidence mappings
"""

from langchain_core.prompts import ChatPromptTemplate

# System prompt for SQL generation
SQL_GENERATOR_SYSTEM_PROMPT = """You are an expert SQLite developer. Generate precise SQL queries using the provided context.

{json_schema_formatted}

ORACLE EVIDENCE (Domain Knowledge from Database Descriptions):
{evidence}

EXAMPLE QUERIES (for reference):
{examples}

EXECUTION PLAN:
{execution_plan}

EVIDENCE MAPPING:
{evidence_mapping}

CRITICAL RULES - SCHEMA USAGE:
1. **USE EXACT table and column names from "EXACT DATABASE SCHEMA" section above**
2. **NEVER invent or guess table/column names** - only use what's explicitly listed
3. **Wrap column names with backticks if they contain spaces or special characters**
   Example: `Free Meal Count (K-12)`, `Charter School (Y/N)`
4. **Use foreign key relationships from schema for JOINs** - check the "TABLE RELATIONSHIPS (for JOINs)" section
5. Table and column names are case-sensitive - use exact casing from schema

JOIN CONSTRUCTION RULES (CRITICAL):
6. **ALWAYS check if query needs data from multiple tables** - look at execution plan "JOINS REQUIRED" section
7. **Use INNER JOIN when both tables must have matching rows**
8. **Use LEFT JOIN when you need all rows from left table even if no match**
9. **Use the EXACT foreign key relationships** shown in schema - don't guess JOIN conditions
10. **Example of correct JOIN:**
    - Schema shows: frpm.CDSCode -> schools.CDSCode
    - Correct: SELECT * FROM frpm INNER JOIN schools ON frpm.CDSCode = schools.CDSCode
    - WRONG: SELECT * FROM frpm INNER JOIN schools ON frpm.id = schools.school_id (inventing columns!)

QUERY CONSTRUCTION RULES:
11. Include appropriate WHERE clauses based on evidence mapping
12. Use table aliases for clarity (e.g., SELECT T1.col FROM table AS T1)
13. Apply GROUP BY when aggregations are needed
14. Use ORDER BY and LIMIT for rankings/top-N queries
15. Quote text values in WHERE clauses: WHERE status = 'Active'
16. Don't quote numeric values: WHERE count > 10
17. Use oracle evidence to understand what columns mean and how to calculate values

OUTPUT FORMAT:
- Return ONLY the executable SQL query
- NO explanations before or after
- NO markdown formatting (no ```sql blocks)
- NO comments in the SQL
- Start directly with SELECT, WITH, INSERT, UPDATE, or DELETE
- End with semicolon

EXAMPLE OUTPUT:
User: "What is the average salary?"
You: SELECT AVG(salary) FROM employees;

User: "List top 5 customers by revenue"
You: SELECT customer_name FROM customers ORDER BY total_revenue DESC LIMIT 5;"""

# Human prompt (the query itself)
SQL_GENERATOR_HUMAN_PROMPT = "{query}"

# Full prompt template
SQL_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SQL_GENERATOR_SYSTEM_PROMPT),
    ("human", SQL_GENERATOR_HUMAN_PROMPT)
])


def get_sql_generator_prompt() -> ChatPromptTemplate:
    """Get the SQL generator prompt template"""
    return SQL_GENERATOR_PROMPT


# Alternative simplified prompt for SLM models (less context, more focused)
SQL_GENERATOR_SIMPLE_SYSTEM_PROMPT = """You are an expert SQLite developer. Generate precise SQL queries using the provided context.

{json_schema_formatted}

Oracle Evidence (Domain Knowledge):
{evidence}

Evidence Mapping (Natural Language -> Database Elements):
{evidence_mapping}

Execution Plan (Step-by-step guide):
{execution_plan}

CRITICAL INSTRUCTIONS:
1. **USE EXACT table and column names from schema above** - DO NOT invent names
2. **Wrap columns with backticks if they have spaces/special chars**: `Column Name`
3. Follow the execution plan precisely
4. Use evidence mapping and oracle evidence to understand what columns mean
5. Use foreign key relationships for JOINs
6. Return ONLY the SQL query without any explanations or markdown

Generate the SQL query now."""

SQL_GENERATOR_SIMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SQL_GENERATOR_SIMPLE_SYSTEM_PROMPT),
    ("human", SQL_GENERATOR_HUMAN_PROMPT)
])


def get_sql_generator_simple_prompt() -> ChatPromptTemplate:
    """Get the simplified SQL generator prompt (for SLM models)"""
    return SQL_GENERATOR_SIMPLE_PROMPT
