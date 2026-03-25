# NL2SQL Agent Tests

## Test Structure

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── test_schema_extractor.py
│   ├── test_query_decomposer.py
│   ├── test_sql_generator.py
│   └── test_validator.py
└── integration/               # Integration tests for workflow
    └── test_workflow.py
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run unit tests only
```bash
pytest tests/unit/ -v
```

### Run integration tests only
```bash
pytest tests/integration/ -v
```

### Run specific test file
```bash
pytest tests/unit/test_schema_extractor.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=api --cov-report=html
```

## Test Categories

### Unit Tests
- **test_schema_extractor.py**: Tests for schema extraction, RAG, and complexity analysis
- **test_query_decomposer.py**: Tests for query decomposition (ensures no conditional skipping)
- **test_sql_generator.py**: Tests for SQL generation and example formatting
- **test_validator.py**: Tests for SQL validation at multiple levels

### Integration Tests
- **test_workflow.py**: Tests for end-to-end workflow and graph flow correctness

## Writing New Tests

### Unit Test Template
```python
import pytest
from api.src.agents.your_agent import YourAgent

class TestYourAgent:
    def test_feature(self):
        agent = YourAgent(...)
        result = agent.method()
        assert result == expected
```

### Integration Test Template
```python
import pytest
from SQL_Agent_New import execute_query

class TestIntegration:
    def test_end_to_end(self):
        result = execute_query("test query", "test_db", "sql_only")
        assert "SELECT" in result
```

## Coverage Goals

- Unit test coverage: >80%
- Integration test coverage: >60%
- Critical paths: 100%
