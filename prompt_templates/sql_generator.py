"""
SQL Generator Prompt Template
Generates precise SQL queries from decomposed plans and evidence mappings
"""

from langchain_core.prompts import ChatPromptTemplate

# System prompt for SQL generation
SQL_GENERATOR_SYSTEM_PROMPT = """You are an expert SQLite developer. Generate precise SQL queries using the provided context.

DATABASE SCHEMA:
- Direct Schema: {direct_schema}
- RAG Context (Column Descriptions): {rag_context}
- Evidence/Mappings: {evidence}

EXAMPLE QUERIES (for reference):
{examples}

EXECUTION PLAN:
{execution_plan}

EVIDENCE MAPPING:
{evidence_mapping}

RULES:
1. Use proper JOIN syntax when combining tables
2. Include appropriate WHERE clauses based on evidence mapping
3. Use table aliases for clarity (e.g., SELECT t1.col FROM table AS t1)
4. Apply GROUP BY when aggregations are needed
5. Use ORDER BY and LIMIT for rankings/top-N queries
6. Apply the evidence mappings to correctly translate natural language to database columns
7. Learn from the example queries provided
8. Use exact column names from the schema (case-sensitive)
9. Quote text values in WHERE clauses: WHERE status = 'Active'
10. Don't quote numeric values: WHERE count > 10

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
SQL_GENERATOR_SIMPLE_SYSTEM_PROMPT = """You are an expert SQLite developer. Generate precise SQL queries using:

RAG Context (Database Schema Information):
{rag_context}

Evidence Mapping (Natural Language -> Database Elements):
{evidence_mapping}

Execution Plan (Step-by-step guide):
{execution_plan}

Instructions:
1. Follow the execution plan precisely
2. Use evidence mapping to correctly translate terms to database columns/tables
3. Write syntactically correct SQLite queries
4. Use proper JOIN syntax when combining tables
5. Include appropriate WHERE, GROUP BY, ORDER BY clauses as needed
6. Use aliases for clarity
7. Return ONLY the SQL query without any explanations or markdown

Generate the SQL query now."""

SQL_GENERATOR_SIMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SQL_GENERATOR_SIMPLE_SYSTEM_PROMPT),
    ("human", SQL_GENERATOR_HUMAN_PROMPT)
])


def get_sql_generator_simple_prompt() -> ChatPromptTemplate:
    """Get the simplified SQL generator prompt (for SLM models)"""
    return SQL_GENERATOR_SIMPLE_PROMPT
