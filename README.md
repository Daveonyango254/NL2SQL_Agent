# NL2SQL_AGENT

A modular, configurable SQL generation system using LangGraph for natural language to SQL translation. Supports both cloud-based LLMs (OpenAI) and local Small Language Models (SLMs) via Ollama.

## Features

- **Configurable Model Backend**: Switch between OpenAI GPT-4o and local Ollama models via `config.yaml`
- **Automatic Fallback**: Falls back to OpenAI when Ollama fails or after configured retry attempts
- **RAG-Enhanced Schema Understanding**: Uses persistent embeddings for better schema comprehension
- **Query Decomposition**: Breaks complex queries into execution plans and evidence mappings
- **Multi-Layer SQL Validation**: Structural, schema, semantic, and result validation
- **Robust SQL Extraction**: Handles verbose LLM outputs with intelligent SQL extraction
- **BIRD Benchmark Evaluation**: Full evaluation pipeline with EX and VES metrics

## Project Structure

```
NL2SQL_Agent/
├── SQL_Agent.py              # Main agent (configurable OpenAI/Ollama)
├── config.yaml               # Central configuration file
├── app.py                    # Streamlit web interface
├── embed_training_data.py    # Generate persistent embeddings
│
├── prompt_templates/         # Centralized prompt management
│   ├── __init__.py
│   ├── query_decomposer.py   # Query decomposition prompts
│   ├── sql_generator.py      # SQL generation prompts
│   └── formatter.py          # Result formatting prompts
│
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── sql_extractor.py      # Robust SQL extraction
│   └── sql_validator.py      # Multi-level SQL validation
│
├── evaluation/               # BIRD benchmark evaluation
│   ├── config.py             # Evaluation configuration
│   ├── generate_predictions.py
│   ├── run_evaluation.py
│   ├── run_full_pipeline.py
│   ├── evaluator_ex.py       # Execution accuracy evaluator
│   ├── evaluator_ves.py      # Valid efficiency score evaluator
│   └── logger.py             # Logging utilities
│
├── data/
│   └── bird/
│       ├── dev_databases/    # BIRD benchmark databases
│       ├── dev.json          # BIRD dev questions
│       └── dev_2_examples.json # Few-shot examples
│
└── embeddings/               # Persistent vector embeddings (per database)
```

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (for GPT models)
- Ollama (optional, for local SLM models)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/NL2SQL_Agent.git
   cd NL2SQL_Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Configure model backend** (edit `config.yaml`)
   ```yaml
   # Use OpenAI
   primary_model_type: "openai"

   # Or use Ollama (local)
   primary_model_type: "ollama"
   ```

5. **Generate embeddings** (required for RAG)
   ```bash
   python embed_training_data.py
   # Or for specific database:
   python embed_training_data.py --db california_schools
   ```

## Configuration

### config.yaml

```yaml
# Model Selection
primary_model_type: "ollama"  # "openai" or "ollama"

# OpenAI Configuration
openai:
  sql_generator_model: "gpt-4o"
  query_decomposer_model: "gpt-4o"
  fallback_model: "gpt-4o"
  temperature: 0

# Ollama Configuration
ollama:
  base_url: "http://localhost:11434"
  sql_generator_model: "llama3.1:8b"
  query_decomposer_model: "mistral:7b"
  formatter_model: "llama3.1:8b"
  fallback_to_openai: true
  temperature: 0

# Retry Configuration
retry:
  max_retries: 3
  fallback_after_retry: 2
```

## Usage

### Command Line

```bash
# Basic usage
python SQL_Agent.py <db_id> "<question>" <output_mode>

# Examples
python SQL_Agent.py california_schools "List all charter schools" sql_only
python SQL_Agent.py toxicology "How many molecules are carcinogenic?" sql_with_results
```

### Python API

```python
from SQL_Agent import run_sql_agent

result = run_sql_agent(
    query="What are the top 5 schools by enrollment?",
    db_id="california_schools",
    output_mode="sql_only"  # or "sql_with_results", "nlp_explanation"
)
print(result)
```

### Web Interface

```bash
streamlit run app.py
```

## Evaluation

Run the full evaluation pipeline on BIRD benchmark:

```bash
# Evaluate on all test questions
python evaluation/run_full_pipeline.py

# Evaluate with limit
python evaluation/run_full_pipeline.py --limit 100

# Run with custom experiment name
python evaluation/run_full_pipeline.py --experiment_name my_test
```

### Metrics

- **EX (Execution Accuracy)**: Percentage of queries returning correct results
- **R-VES (Valid Efficiency Score)**: Measures both correctness and execution efficiency

## Architecture

### Agent Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│  Schema_Extraction  │  ← Load DB schema + RAG context
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Query_Decomposer   │  ← Break query into execution plan
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   SQL_Generator     │  ← Generate SQL using plan + evidence
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Executor_Validator  │  ← Execute SQL and validate
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
   Success    Error
     │           │
     ▼           ▼
┌─────────┐ ┌───────────┐
│Formatter│ │Regenerate │ ← Retry with fallback model
└────┬────┘ └─────┬─────┘
     │            │
     ▼            │
    END ◄─────────┘
```

## Ollama Setup

### Local Installation

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.1:8b
ollama pull mistral:7b

# Start server
ollama serve
```

### Google Colab (Free GPU)

Use `ollama_server_colab.ipynb` to run Ollama with a free GPU and expose via ngrok tunnel for faster inference in the event of very limited hardware capability.

## Troubleshooting

### Ollama Connection Error

```
[FATAL ERROR] Cannot connect to Ollama server
```

**Solutions:**
1. Start Ollama: `ollama serve`
2. Check URL in `config.yaml`
3. Use OpenAI: set `primary_model_type: "openai"`

### Missing Embeddings

```
No precomputed embeddings found for <db_id>
```

**Solution:**
```bash
python embed_training_data.py --db <db_id>
```

## License

MIT License

## Acknowledgments

- BIRD Benchmark for SQL evaluation dataset
- LangGraph for agent orchestration
- Ollama for local LLM inference
