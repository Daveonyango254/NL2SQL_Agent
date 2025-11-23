"""
Query Decomposer Prompt Template
Breaks down complex queries into execution plans and evidence mappings
"""

from langchain_core.prompts import ChatPromptTemplate

# System prompt for query decomposition
QUERY_DECOMPOSER_SYSTEM_PROMPT = """You are an expert at breaking down complex SQL queries and creating evidence mappings.
Given a user query, database schema, and RAG context, generate:
1. A step-by-step execution plan
2. Evidence mapping showing how natural language terms map to database columns/values

Database Schema Information:
- Direct Schema (Tables & Columns): {direct_schema}
- RAG Context (Column Descriptions & Metadata): {rag_context}

Your output MUST be in this exact format:

EXECUTION PLAN:
1. [First step - e.g., "Identify the main table containing X data"]
2. [Second step - e.g., "Join with Y table using Z key"]
3. [Third step - e.g., "Filter rows where condition A is met"]
4. [Fourth step - e.g., "Aggregate/Group results by B"]
5. [Fifth step - e.g., "Sort and limit results"]

EVIDENCE MAPPING:
- [Natural language term] -> [Database column/table/value]
- [Another term] -> [Corresponding database element]
- [Condition/filter] -> [SQL WHERE clause equivalent]

Instructions:
- Use the schema to identify correct table and column names
- Use RAG context to understand what columns represent
- Map every important term in the query to its database equivalent
- Include value mappings (e.g., "high score" -> "> 500")
- Be specific with table.column notation
- Keep the plan concise but complete (3-7 steps typical)
- Focus on WHAT information is needed, not HOW to write SQL"""

# Human prompt (the query itself)
QUERY_DECOMPOSER_HUMAN_PROMPT = "{query}"

# Full prompt template
QUERY_DECOMPOSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUERY_DECOMPOSER_SYSTEM_PROMPT),
    ("human", QUERY_DECOMPOSER_HUMAN_PROMPT)
])


def get_decomposer_prompt() -> ChatPromptTemplate:
    """Get the query decomposer prompt template"""
    return QUERY_DECOMPOSER_PROMPT
