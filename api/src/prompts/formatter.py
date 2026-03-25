"""
Formatter Prompt Template
Converts SQL results into natural language explanations
"""

from langchain_core.prompts import ChatPromptTemplate

# System prompt for result formatting
FORMATTER_SYSTEM_PROMPT = """Convert SQL results into a natural language explanation.
Be clear, concise, and highlight key insights.

Original Question: {question}
SQL Query: {sql}
Results: {results}
Schema Context: {schema_context}

Guidelines:
1. Directly answer the user's original question
2. Summarize the key findings from the results
3. Use natural, conversational language
4. Include specific numbers/values from the results
5. If the results are empty, explain what that means
6. Keep the explanation concise but informative
7. Highlight any notable patterns or insights"""

# Human prompt
FORMATTER_HUMAN_PROMPT = "Provide a comprehensive explanation of the results"

# Full prompt template
FORMATTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FORMATTER_SYSTEM_PROMPT),
    ("human", FORMATTER_HUMAN_PROMPT)
])


def get_formatter_prompt() -> ChatPromptTemplate:
    """Get the formatter prompt template"""
    return FORMATTER_PROMPT


# Response templates for different output modes
SQL_ONLY_TEMPLATE = """```sql
{sql_query}
```"""

SQL_WITH_RESULTS_TEMPLATE = """**SQL Query:**
```sql
{sql_query}
```

**Results:**
{results}"""

NLP_EXPLANATION_TEMPLATE = """**Your Question:** {user_query}

**SQL Query Generated:**
```sql
{sql_query}
```

**Explanation:**
{explanation}

**Confidence Score:** {confidence_score:.2f}"""

ERROR_TEMPLATE = """**Error occurred while processing your query:**

Original Query: {user_query}

Attempted SQL:
```sql
{sql_query}
```

Error Count: {error_count}

Please try rephrasing your question or contact support for assistance."""


def format_sql_only(sql_query: str) -> str:
    """Format output for sql_only mode"""
    return SQL_ONLY_TEMPLATE.format(sql_query=sql_query)


def format_sql_with_results(sql_query: str, results: str) -> str:
    """Format output for sql_with_results mode"""
    return SQL_WITH_RESULTS_TEMPLATE.format(sql_query=sql_query, results=results)


def format_nlp_explanation(user_query: str, sql_query: str, explanation: str, confidence_score: float) -> str:
    """Format output for nlp_explanation mode"""
    return NLP_EXPLANATION_TEMPLATE.format(
        user_query=user_query,
        sql_query=sql_query,
        explanation=explanation,
        confidence_score=confidence_score
    )


def format_error(user_query: str, sql_query: str, error_count: int) -> str:
    """Format error output"""
    return ERROR_TEMPLATE.format(
        user_query=user_query,
        sql_query=sql_query if sql_query else "Not generated",
        error_count=error_count
    )
