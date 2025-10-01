# RAG Implementation Practice

Hands-on learning for production-grade RAG systems with emphasis on evaluation, optimization, and statistical validation.

## 📚 Overview

Progressive weekly modules covering:
- **Week 0**: Vector databases (LanceDB)
- **Week 1**: Synthetic test generation → Retrieval benchmarking → Statistical validation

**Core Focus:** Prove your RAG improvements are real, not just noise.

---

## 🚀 Quick Start

```bash
# Setup
git clone <repo-url> && cd rag-course-practice
uv venv && uv pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI API key

# Run Week 1 workflow
uv run week-1/01_syn_ques_eg.py      # Generate test questions
uv run week-1/02_bench_retrieve.py   # Benchmark configurations
uv run week-1/03_visualise_results.py # Statistical analysis
```

---

## 📁 Project Structure

```
week-0/
  └── lancedb_example.py          # Vector DB basics

week-1/
  ├── 01_syn_ques_eg.py           # Generate evaluation datasets
  ├── 02_bench_retrieve.py        # Benchmark 8 configurations
  ├── 03_visualise_results.py     # Bootstrapping & significance tests
  ├── 02_learnings.md             # Retrieval metrics guide
  ├── 03_learnings.md             # Statistical validation guide ⭐
  └── semaphore.md                # AsyncIO patterns
```

---

## 🎯 What You'll Learn

### Week 1: Statistical Validation of RAG Systems

**The Problem:** You test two models and see:
- Model A: recall@10 = 0.847
- Model B: recall@10 = 0.851

Is Model B actually better, or just lucky?

**The Solution:** Bootstrapping + Statistical Testing

```python
# After 1,000 bootstrap simulations:
Model A: 0.847 ± 0.027 (CI: [0.820, 0.874])
Model B: 0.851 ± 0.026 (CI: [0.825, 0.877])
p-value: 0.234 (NOT significant)

Decision: ❌ Don't deploy Model B (improvement is noise)
```

**Files:**
- `03_visualise_results.py` - Implementation
- `03_learnings.md` - Complete guide (when to use, sample sizes, decision framework)

---

## 📊 Evaluation Workflow

```bash
# 1. Generate test dataset (100 questions)
uv run week-1/01_syn_ques_eg.py

# 2. Test configurations (8 experiments: 2 rerankers × 2 search modes × 2 embeddings)
uv run week-1/02_bench_retrieve.py

# 3. Statistical validation
uv run week-1/03_visualise_results.py
# → Bootstrapping (1,000 simulations)
# → Confidence intervals
# → p-values
# → Plots with uncertainty bands

# 4. Make decision
# If p < 0.05 AND worth the cost → Deploy ✅
# If p > 0.05 → Don't deploy (noise) ❌
```

---

## 📖 Key Metrics

**Recall@k**: Did we find the correct answer in top k results?
```python
recall@10 = 0.85  # 85% of queries had correct answer in top 10
```

**MRR (Mean Reciprocal Rank)**: How high did we rank the correct answer?
```python
mrr@10 = 0.75  # Average position = 1.33 (1/0.75)
```

---

## 🎓 Key Takeaways

1. **Don't trust single numbers** - Use confidence intervals
   ```python
   ❌ "recall@10 = 0.85"
   ✅ "recall@10 = 0.85 ± 0.03 (CI: [0.82, 0.88])"
   ```

2. **Statistical significance ≠ practical significance**
   - Can be significant (p < 0.05) but not worth deploying (tiny improvement, high cost)

3. **Sample size matters**
   - < 30 questions: Too small
   - 50-100: Reasonable
   - 100+: Good
   - 10,000+: Skip bootstrapping

4. **Context over p-values**
   - Free upgrade with p=0.08? → Deploy anyway
   - 6.5x cost with p=0.23? → Don't deploy

---

## 🛠️ Development

```bash
# Code quality
uv run black <file.py>
uv run ruff <file.py>
uv run mypy <file.py>

# Environment variables (.env)
OPENAI_API_KEY=sk-...
BRAINTRUST_API_KEY=...  # Optional
COHERE_API_KEY=...      # Optional
```

---

## 📚 Essential Reading

- **[03_learnings.md](week-1/03_learnings.md)** - Complete statistical validation guide
  - When to use bootstrapping
  - How to interpret p-values
  - Sample size guidelines
  - 5 real-world decision examples

- **[semaphore.md](week-1/semaphore.md)** - AsyncIO concurrency patterns

---

## 🔧 Troubleshooting

**Rate limits**: Reduce `Semaphore(10)` → `Semaphore(5)`
**Memory**: Process in smaller batches
**Dependencies**: `rm -rf .venv && uv venv && uv pip install -r requirements.txt`

---

## 🤝 Contributing

1. Create file in `week-X/`
2. Add docstring header (see existing files)
3. Update README
4. Create `XX_learnings.md` with insights

---

**"Don't trust a single number. Get the distribution. Test significance. Make data-driven decisions."**
