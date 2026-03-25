"""
Unit tests for Query Decomposer
"""

import pytest
from api.src.agents.query_decomposer import QueryDecomposer


class TestQueryDecomposer:
    """Test QueryDecomposer agent"""

    def test_parse_decomposition_response(self):
        """Test parsing of decomposition response"""
        # Mock LLM and prompt
        class MockLLM:
            def invoke(self, inputs):
                class Response:
                    content = """
EXECUTION PLAN:
1. Find the schools table
2. Filter by charter status
3. Return results

EVIDENCE MAPPING:
- charter schools -> charter_flag = 1
- schools -> schools table
"""
                return Response()

        class MockPrompt:
            def __or__(self, other):
                return self

            def invoke(self, inputs):
                return MockLLM().invoke(inputs)

        decomposer = QueryDecomposer(llm=MockLLM(), decompose_prompt=MockPrompt())
        result = decomposer._parse_decomposition_response(MockLLM().invoke({}).content)

        assert len(result["execution_plan"]) > 0
        assert len(result["evidence_mapping"]) > 0

    def test_always_executes(self):
        """Test that decomposer always executes (no conditional skipping)"""
        # This test ensures the decompose method doesn't have skip logic
        class MockLLM:
            def invoke(self, inputs):
                class Response:
                    content = "EXECUTION PLAN:\n1. Test\nEVIDENCE MAPPING:\n- test -> test"
                return Response()

        class MockPrompt:
            def __or__(self, other):
                return self

            def invoke(self, inputs):
                return MockLLM().invoke(inputs)

        decomposer = QueryDecomposer(llm=MockLLM(), decompose_prompt=MockPrompt())

        # Should always return a result, never skip
        result = decomposer.decompose("simple query", {})
        assert "execution_plan" in result
        assert "evidence_mapping" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
