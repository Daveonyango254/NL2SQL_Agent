"""
Unit tests for Validator
"""

import pytest
from api.src.agents.validator import SQLValidator, ValidationLevel, ValidationResult


class TestSQLValidator:
    """Test SQLValidator agent"""

    def test_validate_structure_valid(self):
        """Test structural validation with valid SQL"""
        validator = SQLValidator()
        sql = "SELECT * FROM schools WHERE id = 1"

        results = validator.validate_structure(sql)

        # Should not have errors for valid SQL
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0

    def test_validate_structure_invalid_start(self):
        """Test structural validation with invalid SQL start"""
        validator = SQLValidator()
        sql = "INVALID SQL QUERY"

        results = validator.validate_structure(sql)

        # Should have error for invalid start
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0

    def test_validate_structure_unbalanced_parens(self):
        """Test structural validation with unbalanced parentheses"""
        validator = SQLValidator()
        sql = "SELECT * FROM schools WHERE (id = 1"

        results = validator.validate_structure(sql)

        # Should have error for unbalanced parentheses
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0

    def test_validate_results_empty(self):
        """Test result validation with empty results"""
        validator = SQLValidator()
        sql = "SELECT * FROM schools"
        results = []

        validations = validator.validate_results(sql, "test query", results)

        # Should have warning for empty results
        warnings = [r for r in validations if r.level == ValidationLevel.WARNING]
        assert len(warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
