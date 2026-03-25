# Immediate Recommendations for SQL Agent Improvement

**Based on Baseline Evaluation Results (110 queries, 35.45% EX accuracy)**

---

## 🔴 CRITICAL ISSUES (Fix First)

### 1. Schema Extraction Providing Wrong Table/Column Names

**Problem:**
- 33.3% of failures due to "Column not found" errors
- 2.8% due to "Table not found" errors
- Agent is hallucinating standard naming conventions instead of using actual schema

**Examples:**
```sql
# Wrong (predicted):
SELECT Math_Avg FROM sat_scores ...

# Correct (ground truth):
SELECT AvgScrMath FROM satscores ...
```

**Root Cause:**
- Schema extractor may not be providing complete schema information
- Or SQL generator is ignoring the provided schema
- RAG embeddings might not be retrieving relevant schema (Can we create a json of entire BIRD Database Schema / Descriptions and just use the json with context aware RAG?)

**Action Items:**
1. ✅ **Verify schema extraction output**
   ```bash
   # Add debug logging to schema_extractor.py
   # Check what schema is being passed to SQL generator
   ```

2. ✅ **Review SQL generator prompt**
   - Ensure prompt emphasizes using EXACT table/column names from schema
   - Add explicit instruction: "DO NOT invent table or column names"
   - Add few-shot examples showing proper schema usage

3. ✅ **Improve schema representation**
   - Include full CREATE TABLE statements in prompt
   - Add sample data to show column formats
   - Highlight foreign key relationships

**Priority:** 🔴 CRITICAL
**Estimated Impact:** +20-30% accuracy improvement

---

### 2. Validation Not Catching Errors (Zero Retries)

**Problem:**
- Average retry count: 0.00
- 55.6% of failures produce syntactically valid SQL that returns wrong results
- Validator is passing queries on first attempt despite being incorrect

**Root Cause:**
- Validator only checks syntax, not semantic correctness
- No result sanity checks (e.g., checking row counts)
- Validation might be disabled or too lenient

**Action Items:**
1. ✅ **Check validator configuration**
   ```yaml
   # In api/config.yaml
   agents:
     validator:
       enabled: true
       validation_levels:
         structural: true
         schema: true      # ← Is this working?
         semantic: true    # ← Is this working?
         result: true      # ← Is this working?
   ```

2. ✅ **Add execution-based validation**
   - Execute predicted SQL
   - Check if result is empty when question implies data should exist
   - Validate data types match expected output

3. ✅ **Add schema validation**
   - Before execution, verify all table/column names exist in schema
   - This would catch 36% of current errors (table/column not found)

**Priority:** 🔴 CRITICAL
**Estimated Impact:** +15-20% accuracy (by triggering retries)

---

## ⚠️ HIGH PRIORITY ISSUES

### 3. JOIN Logic Errors (Wrong Results Despite Valid SQL)

**Problem:**
- 55.6% of failures execute successfully but return wrong data
- Missing JOINs or wrong JOIN conditions

**Example:**
```sql
# Question: "List zip codes of charter schools in Fresno County Office of Education"

# Predicted (returns 136 rows - wrong):
SELECT Zip FROM schools WHERE DOC = '00' AND SOC IN ('65', '66')

# Ground Truth (returns 5 rows - correct):
SELECT T2.Zip FROM frpm AS T1
INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode
WHERE T1.`District Name` = 'Fresno County Office of Education'
  AND T1.`Charter School (Y/N)` = 1
```

**Root Cause:**
- Query decomposer not identifying need for JOINs
- SQL generator not understanding relationships between tables

**Action Items:**
1. ✅ **Enhance schema with relationship information**
   - Explicitly list foreign keys in schema prompt
   - Add example queries showing common JOINs for each database

2. ✅ **Improve query decomposer prompts**
   - Add explicit step: "Identify which tables need to be joined"
   - Show example decompositions that include JOIN planning

3. ✅ **Add validation for expected result characteristics**
   - If question mentions specific entity, check result contains it
   - Validate cardinality (e.g., if "list all X" but result empty → retry)

**Priority:** ⚠️ HIGH
**Estimated Impact:** +10-15% accuracy

---

### 4. Syntax Errors from Special Characters in Column Names

**Problem:**
- Column names with spaces require backticks: \`FRPM Count (K-12)\`
- Agent producing: `Number of Test Takers` (unquoted, causes syntax error)

**Action Items:**
1. ✅ **Update SQL generator prompt**
   - Add rule: "Always wrap column names with spaces/special chars in backticks"
   - Add examples showing proper quoting

2. ✅ **Pre-process schema**
   - Highlight which columns need quoting
   - Example: `County Name [QUOTED]` or `` `County Name` ``

**Priority:** ⚠️ HIGH
**Estimated Impact:** +3-5% accuracy

---

## 💡 MEDIUM PRIORITY IMPROVEMENTS

### 5. Query Decomposer May Not Be Helping

**Observation:**
- Query decomposer is enabled and always runs
- But errors suggest it's not effectively breaking down queries

**Hypothesis:**
- Decomposer output might not be improving SQL generation
- Could be adding latency without value

**Action Items:**
1. ✅ **Run A/B test**
   ```yaml
   # Test 1: Decomposer enabled (current)
   agents:
     query_decomposer:
       enabled: true

   # Test 2: Decomposer disabled
   agents:
     query_decomposer:
       enabled: false
   ```

2. ✅ **Compare results**
   - Run 50 queries with decomposer ON
   - Run same 50 queries with decomposer OFF
   - Compare EX accuracy and latency

**Priority:** 💡 MEDIUM
**Estimated Impact:** Unknown (could be +5% or -5%)

---

### 6. Model Selection (SLM vs LLM)

**Current State:**
- Fallback disabled (`enable_fallback: false`)
- 100% queries handled by SLM (llama3.1:8b)
- 35.45% accuracy

**Recommendation:**
1. ✅ **Test with fallback enabled**
   ```yaml
   retry:
     enable_fallback: true
     fallback_after_retry: 2  # Try SLM twice, then use LLM
   ```

2. ✅ **Compare costs vs accuracy**
   - If LLM gives +30% accuracy, calculate cost trade-off
   - Hybrid approach: Use SLM first, fallback selectively

**Priority:** 💡 MEDIUM
**Estimated Impact:** +20-40% accuracy (but higher cost)

---

## 📊 QUICK WINS (Easy to implement)

### 7. Fix Obvious Schema Mismatches

**Databases with 0% accuracy need immediate attention:**
- `california_schools` - Table name issues (`satscores` vs `sat_scores`)
- All other databases also 0% (per database report bug)

**Action:**
1. ✅ **Manually inspect top 3 failing databases**
   - Check if embeddings are working
   - Verify schema is being loaded correctly
   - Test with a single query manually

2. ✅ **Add schema validation test**
   - Script to verify all tables/columns in schema match database
   - Alert if embeddings are stale

**Priority:** ✅ QUICK WIN
**Estimated Impact:** +5-10% accuracy

---

### 8. Improve Prompt Engineering

**Current prompts may lack specificity**

**Action Items:**
1. ✅ **Review all prompts** in `prompt_templates/`
   - Add constraint: "Use EXACT table and column names from schema"
   - Add negative examples (what NOT to do)
   - Add chain-of-thought reasoning

2. ✅ **Test with refined prompts**
   - Start with SQL generator prompt
   - Test on 10 failed queries
   - Measure improvement

**Priority:** ✅ QUICK WIN
**Estimated Impact:** +5-15% accuracy

---

## 🎯 RECOMMENDED ACTION PLAN

### Week 1: Critical Fixes
1. **Day 1-2**: Fix schema extraction and validation
   - Ensure full schema is provided to SQL generator
   - Add table/column name validation before execution
   - Enable retry on validation failure

2. **Day 3-4**: Improve SQL generator prompt
   - Emphasize exact schema usage
   - Add JOIN planning examples
   - Add column quoting rules

3. **Day 5**: Test and measure
   - Re-run evaluation with fixes
   - Target: 50-60% accuracy

### Week 2: Optimization
1. **Day 1-2**: A/B test query decomposer
2. **Day 3-4**: Test LLM fallback strategy
3. **Day 5**: Final evaluation and comparison

### Expected Outcome
- **Baseline:** 35.45% EX accuracy
- **After Week 1:** 50-60% EX accuracy
- **After Week 2:** 60-75% EX accuracy

---

## 🔍 Monitoring Checklist

After each change, verify:
- [ ] EX accuracy improved
- [ ] VES score maintained or improved
- [ ] Latency acceptable (< 30s per query)
- [ ] Retry count in reasonable range (0.5-1.5 avg)
- [ ] No regressions on previously passing queries

---

## 📝 Notes

- **SLM (llama3.1:8b) is capable** but needs better guidance
- **Schema accuracy is the #1 issue** - fix this first
- **Validation is not working** - fix this second
- **Quick wins available** through prompt engineering

**Focus on these two areas and accuracy should improve significantly.**
