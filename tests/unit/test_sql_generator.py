"""
Unit tests for SQL Generator
"""

import pytest
from api.src.agents.sql_generator import SQLGenerator


class TestSQLGenerator:
    """Test SQLGenerator agent"""

    def test_format_examples(self):
        """Test example formatting"""
        def mock_extract_sql(text):
            return "SELECT * FROM test"

        class MockLLM:
            pass

        class MockPrompt:
            pass

        generator = SQLGenerator(
            llm=MockLLM(),
            sql_prompt=MockPrompt(),
            sql_extractor_func=mock_extract_sql
        )

        examples = [
            {"question": "Test", "evidence": "test evidence", "SQL": "SELECT * FROM test"}
        ]

        formatted = generator.format_examples(examples)
        assert "Test" in formatted
        assert "test evidence" in formatted

    def test_empty_examples(self):
        """Test handling of empty examples"""
        def mock_extract_sql(text):
            return "SELECT * FROM test"

        class MockLLM:
            pass

        class MockPrompt:
            pass

        generator = SQLGenerator(
            llm=MockLLM(),
            sql_prompt=MockPrompt(),
            sql_extractor_func=mock_extract_sql
        )

        formatted = generator.format_examples([])
        assert "No similar examples" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
