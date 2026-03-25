"""
Unit tests for Schema Extractor
"""

import pytest
from pathlib import Path
from api.src.agents.schema_extractor import SchemaExtractor


class TestSchemaExtractor:
    """Test SchemaExtractor agent"""

    def test_initialization(self):
        """Test basic initialization"""
        extractor = SchemaExtractor(
            db_id="test_db",
            db_path=None,
            csv_paths=[],
            examples=[],
            config={}
        )
        assert extractor.db_id == "test_db"
        assert extractor.db is None

    def test_get_similar_examples(self):
        """Test example similarity matching"""
        examples = [
            {"question": "Show all schools", "SQL": "SELECT * FROM schools"},
            {"question": "List students", "SQL": "SELECT * FROM students"}
        ]
        extractor = SchemaExtractor(
            db_id="test_db",
            examples=examples,
            config={}
        )

        similar = extractor.get_similar_examples("Show schools", limit=1)
        assert len(similar) > 0

    def test_query_complexity_analysis(self):
        """Test complexity analysis"""
        extractor = SchemaExtractor(db_id="test_db", config={})

        # Simple query
        complexity = extractor._analyze_query_complexity(
            "List all schools",
            {"examples": []}
        )
        assert complexity in ["simple", "moderate", "complex"]

    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        extractor = SchemaExtractor(db_id="test_db", config={})

        confidence = extractor._calculate_relevance_confidence(
            "test query",
            {"examples": [], "evidence": [], "rag_context": None}
        )
        assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
