# CESMA SQL Agent - Evaluation System

Comprehensive evaluation pipeline for the CESMA SQL Agent, supporting EX (Execution Accuracy) and VES (Valid Efficiency Score) metrics with optional LangSmith integration.

## 📁 Directory Structure

```
evaluation/
├── output/                      # ALL results saved here
│   ├── predictions_*.json       # Generated SQL predictions
│   ├── *_results.json          # Combined evaluation results
│   ├── *_ex.txt                # EX metric reports
│   ├── *_ves.txt               # VES metric reports
│   └── *_pipeline.log          # Pipeline execution logs
├── config.py                    # Configuration (paths, output dir)
├── logger.py                    # Logging utilities
├── evaluation_utils.py          # Common utility functions
├── evaluator_ex.py              # EX metric (consolidated)
├── evaluator_ves.py             # VES metric (consolidated)
├── generate_predictions.py      # Prediction generation
├── run_evaluation.py            # Evaluation runner
├── run_full_pipeline.py         # End-to-end pipeline
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Full Pipeline (Recommended)

Run the complete evaluation workflow: generate predictions → run metrics → save results.

```bash
# Test with 10 questions
python evaluation/run_full_pipeline.py --limit 10

# Full evaluation
python evaluation/run_full_pipeline.py

# With LangSmith tracing
python evaluation/run_full_pipeline.py --use_langsmith --experiment_name my_experiment

# Custom configuration
python evaluation/run_full_pipeline.py --limit 50 --num_cpus 8 --skip_ves
```

### 2. Generate Predictions Only

Generate SQL predictions from dev.json without running evaluation.

```bash
# Generate predictions
python evaluation/generate_predictions.py

# With options
python evaluation/generate_predictions.py --limit 20 --use_langsmith

# Resume from question 50
python evaluation/generate_predictions.py --skip 50 --limit 25
```

### 3. Evaluate Existing Predictions

Run evaluation on pre-generated predictions.

```bash
python evaluation/run_evaluation.py \
  --predicted_sql_path evaluation/output/predictions_20250113_123456.json \
  --ground_truth_path data/bird/dev.json \
  --diff_json_path data/bird/dev.json \
  --num_cpus 4
```

## 📊 Evaluation Metrics

### EX (Execution Accuracy)
Measures whether predicted SQL returns the same results as ground truth.
- **Score**: Percentage of queries with matching results
- **Breakdown**: By difficulty (simple, moderate, challenging)

### VES (Valid Efficiency Score)
Measures both correctness and query efficiency.
- **Calculation**: Based on execution time ratio
- **Rewards**:
  - 1.25: Predicted query is 2x+ faster
  - 1.00: Predicted query is 1-2x faster
  - 0.75: Predicted query is 0.5-1x speed
  - 0.50: Predicted query is 0.25-0.5x speed
  - 0.25: Predicted query is slower than 0.25x
  - 0.00: Incorrect results or error

## 🔧 Configuration

### Database Configuration
Edit `evaluation/config.py`:
```python
BIRD_DB_PATH = BASE_DIR / "data" / "bird" / "dev_databases"
BIRD_DEV_JSON = BASE_DIR / "data" / "bird" / "dev.json"
OUTPUT_DIR = BASE_DIR / "evaluation" / "output"  # All results here
```

### SQL Agent Configuration
Edit `config.yaml` in project root:
```yaml
primary_model_type: "openai"  # or "ollama"
openai:
  sql_generator_model: "gpt-4o"
  fallback_model: "gpt-4o"
retry:
  max_retries: 3
```

## 📈 LangSmith Integration

### Setup
```bash
# Install LangSmith
pip install langsmith

# Set API key
export LANGSMITH_API_KEY="your-api-key"
```

### Usage
```bash
# Enable tracing for full pipeline
python evaluation/run_full_pipeline.py \
  --use_langsmith \
  --experiment_name my_experiment_v1

# Enable for predictions only
python evaluation/generate_predictions.py \
  --use_langsmith \
  --experiment_name pred_test_v1
```

### Benefits
- Track all LLM calls and token usage
- Monitor prediction generation progress
- Compare different model configurations
- Debug failed predictions
- Analyze performance over time

## 📝 Output Files

All files are saved to `evaluation/output/`:

### Predictions File
```json
{
  "0": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm...",
  "1": "SELECT ...",
  ...
}
```

### Results JSON
```json
{
  "experiment_name": "pipeline_20250113_123456",
  "timestamp": "2025-01-13T12:34:56",
  "config": {...},
  "ex": {
    "overall_acc": 75.5,
    "simple_acc": 85.2,
    ...
  },
  "ves": {
    "overall_ves": 68.3,
    ...
  }
}
```

### Metric Reports (TXT)
```
================================================================================
EX EVALUATION RESULTS
================================================================================
Difficulty      Count      EX
--------------------------------------------------------------------------------
Simple          50          85.20
Moderate        30          70.00
Challenging     20          55.00
--------------------------------------------------------------------------------
Overall         100         75.50
================================================================================
```

## 🎯 Common Workflows

### Test on Small Dataset
```bash
python evaluation/run_full_pipeline.py --limit 5
```

### Run Only EX Metric
```bash
python evaluation/run_full_pipeline.py --skip_ves --num_cpus 4
```

### Evaluate Specific Database
Edit `dev.json` to filter by `db_id`, or modify the script to filter questions.

### Resume Interrupted Run
```bash
# Skip first 50 questions, run next 25
python evaluation/run_full_pipeline.py --skip 50 --limit 25
```

### Compare Model Configurations
```bash
# Test with OpenAI
python evaluation/run_full_pipeline.py \
  --experiment_name openai_gpt4 \
  --limit 20

# Test with Ollama (edit config.yaml first)
python evaluation/run_full_pipeline.py \
  --experiment_name ollama_local \
  --limit 20
```

## 🐛 Troubleshooting

### "No database connection" Error
- Check `db_path` in `SQL_Agent.py`
- Verify database files exist in `data/bird/dev_databases/`
- Ensure `db_id` matches directory name

### Import Errors
```bash
# Ensure you're in project root
cd "C:\Users\david\OneDrive - Minnesota State\Desktop\School\FALL 2025\APP\Projects\CESMA AGENT"

# Run from root directory
python evaluation/run_full_pipeline.py
```

### Missing Dependencies
```bash
pip install tqdm func_timeout langsmith  # Optional: langsmith
```

### Evaluation Timeout
Increase timeout for complex queries:
```bash
python evaluation/run_full_pipeline.py --meta_time_out 60.0
```

## 📚 API Reference

### PredictionGenerator
```python
from evaluation.generate_predictions import PredictionGenerator

generator = PredictionGenerator(
    output_mode="sql_only",
    use_langsmith=True,
    experiment_name="my_test"
)

predictions = generator.generate_predictions(
    dev_file="data/bird/dev.json",
    limit=10
)
```

### run_full_evaluation
```python
from evaluation.run_evaluation import run_full_evaluation

results = run_full_evaluation(
    predicted_sql_path="evaluation/output/predictions.json",
    ground_truth_path="data/bird/dev.json",
    diff_json_path="data/bird/dev.json",
    num_cpus=4,
    run_ex=True,
    run_ves=True
)
```

## 🔄 Development

### Adding New Metrics
1. Create new evaluator in `evaluation/evaluator_*.py`
2. Add to `run_evaluation.py`
3. Update `run_full_pipeline.py` to include metric
4. Document in this README

### Modifying Output Format
- Edit `print_data()` in `evaluation_utils.py`
- Results are automatically saved to output directory

## 📄 License

Part of the CESMA SQL Agent project.
