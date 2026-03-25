"""
SQL Generator Agent
Generates SQL queries using LLM with execution plans and evidence mapping
"""

from typing import Dict, List


class SQLGenerator:
    """Generate SQL using configured LLM"""

    def __init__(self, llm, sql_prompt, sql_extractor_func):
        """
        Initialize SQL Generator

        Args:
            llm: Language model instance
            sql_prompt: Prompt template for SQL generation
            sql_extractor_func: Function to extract SQL from LLM response
        """
        self.llm = llm
        self.sql_prompt = sql_prompt
        self.extract_sql = sql_extractor_func

    def format_examples(self, examples: List[Dict]) -> str:
        """Format examples for the prompt"""
        if not examples:
            return "No similar examples available."

        formatted = []
        for i, ex in enumerate(examples[:3], 1):
            formatted.append(f"""
Example {i}:
Question: {ex.get('question', 'N/A')}
Evidence: {ex.get('evidence', 'N/A')}
SQL: {ex.get('SQL', 'N/A')}
""")
        return "\n".join(formatted)

    def generate(self, query: str, schema: Dict, execution_plan: List[str],
                 evidence_mapping: List[str]) -> str:
        """
        Generate SQL query

        Args:
            query: User's natural language query
            schema: Schema context from SchemaExtractor
            execution_plan: Execution plan from QueryDecomposer
            evidence_mapping: Evidence mapping from QueryDecomposer

        Returns:
            Generated SQL query string
        """
        chain = self.sql_prompt | self.llm

        examples_str = self.format_examples(schema.get("examples", []))
        plan_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(
            execution_plan)]) if execution_plan else "Direct SQL generation"
        evidence_str = "\n".join(
            evidence_mapping) if evidence_mapping else "No evidence mapping"

        # Format evidence for display
        evidence_items = schema.get("evidence", [])
        if evidence_items:
            evidence_formatted = "\n".join([f"- {item.get('content', item)}" for item in evidence_items])
        else:
            evidence_formatted = "No oracle evidence available."

        response = chain.invoke({
            "query": query,
            "json_schema_formatted": schema.get("json_schema_formatted", "Schema not available"),
            "direct_schema": str(schema.get("direct_schema", {})),
            "rag_context": str(schema.get("rag_context", {})),
            "evidence": evidence_formatted,
            "examples": examples_str,
            "execution_plan": plan_str,
            "evidence_mapping": evidence_str
        })

        content = response.content if hasattr(
            response, 'content') else str(response)
        # Use robust SQL extraction
        return self.extract_sql(content)
