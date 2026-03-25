# SQL Agent Evaluation Guide

## Overview

This guide explains how to run a comprehensive evaluation of the NL2SQL Agent with database-grouped metrics, agent behavior logging, and automated recommendations.

## Quick Start

### Run Evaluation with 10 Queries per Database

```bash
python evaluation/run_full_pipeline.py \
    --queries_per_db 10 \
    --experiment_name baseline_eval_10per_db \
    --enable_agent_logging
```

This will:
1. Sample 10 queries from each database for balanced evaluation
2. Generate predictions with the SQL Agent
3. Run EX and VES evaluation metrics
4. Generate database-grouped performance reports
5. Log detailed agent execution traces for behavior analysis

## Output Files

All results are saved to `evaluation/output/`:

### Core Evaluation Files
- `{experiment}_predictions.json` - Predicted SQL queries
- `{experiment}_predictions_metadata.json` - Full prediction metadata (model used, retries, latency)
- `{experiment}_results.json` - Evaluation results (EX, VES scores)

### Database Metrics Reports
- `{experiment}_database_report.txt` - Formatted table of metrics by database
- `{experiment}_database_report.csv` - CSV version for analysis
- `{experiment}_model_comparison.txt` - SLM vs LLM comparison (if fallback enabled)

### Agent Behavior Logs
- `{experiment}_agent_trace.jsonl` - Detailed execution traces (JSONL format)
- `{experiment}_agent_summary.json` - Summary statistics

### Analysis & Recommendations
- `{experiment}_agent_trace_analysis.json` - Behavior analysis results
- `recommendations_{timestamp}.json` - Improvement recommendations
- `recommendations_{timestamp}.md` - Markdown report with action items

## Step-by-Step Workflow

### Step 1: Run Evaluation

```bash
# Full evaluation with agent logging
python evaluation/run_full_pipeline.py \
    --queries_per_db 10 \
    --experiment_name my_evaluation \
    --enable_agent_logging
```

**Options:**
- `--queries_per_db N` - Sample N queries from each database (balanced evaluation)
- `--limit N` - Limit total number of questions
- `--skip N` - Skip first N questions (for resuming)
- `--enable_agent_logging` - Enable detailed agent execution logging
- `--use_langsmith` - Enable LangSmith tracing (optional)
- `--skip_ex` - Skip EX evaluation
- `--skip_ves` - Skip VES evaluation

### Step 2: Analyze Agent Behavior

```bash
python evaluation/analyze_agent_behavior.py \
    evaluation/output/my_evaluation_agent_trace.jsonl \
    --output evaluation/output/my_evaluation_analysis.json
```

This analyzes:
- Success/failure rates by database
- Node-level error patterns
- Validation failure causes
- Query complexity correlations

### Step 3: Generate Recommendations

```bash
python evaluation/generate_recommendations.py \
    --database_metrics evaluation/output/my_evaluation_database_report.csv \
    --analysis_results evaluation/output/my_evaluation_analysis.json \
    --output evaluation/output/recommendations.json
```

This generates:
- Prioritized improvement recommendations
- Database-specific action items
- Model selection suggestions
- Workflow optimization ideas

## Database Metrics Report

The database report includes:

| Database | Queries | EX (%) | VES | SLM | LLM | Avg Retries | Mean (ms) | p50 (ms) | p95 (ms) | Timeouts |
|----------|---------|--------|-----|-----|-----|-------------|-----------|----------|----------|----------|
| california_schools | 11 | 72.7 | 68.5 | 11 | 0 | 1.2 | 9,100 | 8,450 | 12,300 | 0 |
| ...      | ...     | ...    | ... | ... | ... | ...         | ...       | ...      | ...      | ...      |

**Columns:**
- **Queries** - Number of queries evaluated
- **EX (%)** - Execution Accuracy percentage
- **VES** - Valid Efficiency Score (higher = better)
- **SLM** - Queries handled by SLM (llama3.1:8b)
- **LLM** - Queries that fell back to LLM (gpt-4o)
- **Avg Retries** - Average regeneration attempts
- **Mean/p50/p95 (ms)** - Latency statistics
- **Timeouts** - Queries that exceeded timeout

## Model Comparison Report

If fallback is enabled, you'll get a comparison table:

```
MODEL COMPARISON REPORT
========================================================================
Metric               | SLM (llama3.1:8b) | LLM (gpt-4o)      | Difference
------------------------------------------------------------------------
Total Queries        |                85 |                15 | -
EX Accuracy (%)      |              70.6 |              93.3 |        +22.7%
VES Score            |              68.2 |              87.4 |        +19.2
Mean Latency (ms)    |             8,234 |            12,567 |     +4,333 ms
p95 Latency (ms)     |            15,123 |            18,943 |     +3,820 ms
Avg Retries          |              1.24 |              0.47 |         -0.77
========================================================================
```

## Configuration

### Enable/Disable Fallback

Edit `api/config.yaml`:

```yaml
retry:
  enable_fallback: false  # Disable fallback to GPT-4o (SLM only)
```

### Adjust Timeout

Edit `api/config.yaml`:

```yaml
evaluation:
  default_query_timeout: 120.0  # 2 minutes per query
```

## Troubleshooting

### Progress Bar Not Updating
- This is normal if you see model loading messages
- The bar updates after each query completes
- Check the output files to verify progress

### Import Errors
- Ensure you're running from the project root directory
- Activate your virtual environment: `.venv\Scripts\activate`

### High Latency
- Check if Ollama server is running: `ollama list`
- For Colab/ngrok setup, verify the URL in `config.yaml`

## Next Steps After Evaluation

1. **Review Database Report**
   - Identify databases with low success rates
   - Check if specific databases need schema improvements

2. **Analyze Agent Traces**
   - Look for systematic failure patterns
   - Identify which agent nodes are problematic

3. **Read Recommendations**
   - Start with CRITICAL priority items
   - Implement quick wins from LOW priority items

4. **Iterate**
   - Make improvements based on recommendations
   - Re-run evaluation to measure impact
   - Compare results with previous runs

## Example: Complete Workflow

```bash
# 1. Run evaluation
python evaluation/run_full_pipeline.py \
    --queries_per_db 10 \
    --experiment_name baseline_v1 \
    --enable_agent_logging

# 2. Analyze behavior
python evaluation/analyze_agent_behavior.py \
    evaluation/output/baseline_v1_agent_trace.jsonl

# 3. Generate recommendations
python evaluation/generate_recommendations.py \
    --database_metrics evaluation/output/baseline_v1_database_report.csv \
    --analysis_results evaluation/output/baseline_v1_agent_trace_analysis.json

# 4. Review markdown report
cat evaluation/output/recommendations_*.md

# 5. Make improvements...

# 6. Re-evaluate
python evaluation/run_full_pipeline.py \
    --queries_per_db 10 \
    --experiment_name baseline_v2 \
    --enable_agent_logging

# 7. Compare results
diff evaluation/output/baseline_v1_database_report.txt \
     evaluation/output/baseline_v2_database_report.txt
```

## Notes

- **Agent logging** adds ~5-10% overhead but provides valuable debugging info
- **10 queries per database** is a good balance for quick evaluation
- **Full evaluation** (all questions) takes several hours - use `--limit` for testing
- **Latency percentiles** are recalculated per database (not averaged)
