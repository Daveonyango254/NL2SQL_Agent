"""
Query Decomposer Agent
Decomposes complex queries and generates evidence mapping
NO CONDITIONAL SKIPPING - Always executes decomposition
"""

from typing import Dict, Any


class QueryDecomposer:
    """Decompose complex queries and generate evidence mapping"""

    def __init__(self, llm, decompose_prompt):
        """
        Initialize Query Decomposer

        Args:
            llm: Language model instance
            decompose_prompt: Prompt template for decomposition
        """
        self.llm = llm
        self.decompose_prompt = decompose_prompt

    def decompose(self, query: str, schema: Dict) -> Dict[str, Any]:
        """
        Decompose query and generate evidence mapping
        ALWAYS executes - no conditional skipping

        Args:
            query: User's natural language query
            schema: Schema context from SchemaExtractor

        Returns:
            Dictionary with execution_plan and evidence_mapping
        """
        chain = self.decompose_prompt | self.llm

        # Extract foreign key relationships from JSON schema
        foreign_keys = []
        json_schema = schema.get("json_schema", {})
        if json_schema and "foreign_key_relationships" in json_schema:
            for fk in json_schema["foreign_key_relationships"]:
                fk_str = f"{fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
                foreign_keys.append(fk_str)

        foreign_keys_str = "\n".join(foreign_keys) if foreign_keys else "No foreign key relationships available"

        response = chain.invoke({
            "query": query,
            "direct_schema": str(schema.get("direct_schema", {})),
            "rag_context": str(schema.get("rag_context", {})),
            "foreign_keys": foreign_keys_str
        })

        content = response.content if hasattr(
            response, 'content') else str(response)
        return self._parse_decomposition_response(content)

    def _parse_decomposition_response(self, content: str) -> Dict[str, Any]:
        """Parse the decomposition response into structured format"""
        result = {"execution_plan": [], "evidence_mapping": []}

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if 'EXECUTION PLAN' in line.upper():
                current_section = 'execution_plan'
                continue
            elif 'EVIDENCE MAPPING' in line.upper() or 'EVIDENCE' in line.upper():
                current_section = 'evidence_mapping'
                continue

            if current_section == 'execution_plan':
                cleaned = line.lstrip('0123456789.-) ')
                if cleaned:
                    result["execution_plan"].append(cleaned)
            elif current_section == 'evidence_mapping':
                if '→' in line or '->' in line or ':' in line:
                    result["evidence_mapping"].append(line.lstrip('- '))

        return result
