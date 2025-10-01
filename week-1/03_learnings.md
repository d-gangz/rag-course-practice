<!--
Document Type: Learning Guide
Purpose: Teach engineers without data science backgrounds how to validate RAG improvements using bootstrapping and statistical significance testing
Context: Created after analyzing week-1/03_visualise_results.py to extract practical implementation knowledge
Key Topics: Bootstrapping, confidence intervals, statistical significance, A/B testing for RAG systems
Target Use: Step-by-step reference guide for implementing experiment validation in production RAG systems
-->

# Statistical Validation for RAG Systems: A Practical Guide

**Target Audience:** Engineers with no data science background who need to prove their RAG improvements are real, not just lucky.

## The Problem You're Solving

You made a change to your RAG system (new embedding model, better reranking, different chunking strategy). You test it and see:
- **Before:** recall@10 = 0.82
- **After:** recall@10 = 0.85

**Question:** Is this 0.03 improvement real, or just random luck?

**Why this matters:**
- You don't want to deploy changes based on noise
- You don't want to waste money on expensive models that don't actually help
- You need to justify decisions to your team/manager

## The Solution: Bootstrapping + Statistical Testing

This guide shows you how to:
1. **Bootstrap your results** - Create 1,000 simulated experiments to understand uncertainty
2. **Visualize confidence intervals** - See if improvements are meaningful
3. **Run statistical tests** - Get a yes/no answer: "Is this change worth it?"

---

## Part 1: Understanding Bootstrapping (The Core Concept)

### What Is Bootstrapping?

**Simple analogy:** You flip a coin 100 times and get 55 heads. Is the coin biased, or did you just get lucky?

Bootstrap answer:
- Pretend your 100 flips are the "population"
- Randomly pick 100 flips WITH REPLACEMENT (some flips counted 2-3 times, others skipped)
- Count heads again → maybe 52 this time
- Repeat 1,000 times
- Now you have 1,000 different "head counts" ranging from 48-62
- This distribution tells you: "The true bias is probably between 50-60 heads"

### How It Applies to RAG

You have 100 test questions. Each question gets a recall score.

**Without bootstrapping:**
```
recall@10 = 0.85  (average of 100 questions)
```
You have ONE number. Is 0.85 good? Could it have been 0.80 or 0.90?

**With bootstrapping:**
```python
# Step 1: Resample your 100 questions (with replacement) 1,000 times
# Step 2: Calculate recall@10 for each resample
# Step 3: Get 1,000 different recall@10 values

Sample 1: recall@10 = 0.84
Sample 2: recall@10 = 0.86
Sample 3: recall@10 = 0.83
...
Sample 1000: recall@10 = 0.87

# Now you know: "recall@10 is probably between 0.82-0.88 (95% confidence)"
```

### Key Insight

Bootstrapping transforms:
- **"My metric is 0.85"** (useless - no context)
- Into: **"My metric is 0.85, probably between 0.82-0.88"** (actionable!)

---

## Part 2: Step-by-Step Implementation

### Prerequisites

1. **Logged experiment results** in a system like Braintrust (or any logging tool)
2. **Per-question results** - You need individual question scores, not just averages
3. **Two experiments to compare** - e.g., "text-embedding-3-small" vs "text-embedding-3-large"

### Step 1: Define Your Experiment Configurations

```python
@dataclass
class ExperimentConfig:
    experiment_id: str    # UUID from your logging system
    project_name: str     # Project name in Braintrust
    label: str           # Human-readable name for plots

configs = [
    ExperimentConfig(
        experiment_id="uuid-1-small",
        project_name="Text-2-SQL",
        label="text-embedding-3-small"
    ),
    ExperimentConfig(
        experiment_id="uuid-2-large",
        project_name="Text-2-SQL",
        label="text-embedding-3-large"
    ),
]
```

**Critical requirement:** The two experiments must differ in **only ONE variable** (e.g., just the embedding model). If you change multiple things, you won't know what caused the improvement.

### Step 2: Define Your Metrics

```python
def calculate_recall(predictions: list[str], gt: list[str]):
    """How many correct items did we retrieve?"""
    return len([label for label in gt if label in predictions]) / len(gt)

def calculate_mrr(predictions: list[str], gt: list[str]):
    """How high did we rank the first correct item?"""
    mrr = 0
    for label in gt:
        if label in predictions:
            mrr = max(mrr, 1 / (predictions.index(label) + 1))
    return mrr

metrics = [("recall", calculate_recall), ("mrr", calculate_mrr)]
k = [1, 3, 5, 10, 15, 20]  # Test at different k values
```

**Common metrics:**
- **Recall@k:** "Did we find the correct answer in the top k results?"
- **MRR (Mean Reciprocal Rank):** "How high did we rank the correct answer?"
- **Precision@k:** "What % of top k results are correct?"

### Step 3: Run Bootstrapping

```python
def bootstrap_samples(config: ExperimentConfig, n_samples: int = 1000):
    means_list = []
    stds_list = []

    # Get raw per-question results
    items = config.get_results()  # [{preds: [...], labels: [...]}, ...]

    # Create 1,000 resampled experiments
    for _ in range(n_samples):
        # Resample WITH REPLACEMENT (key to bootstrapping!)
        sample = np.random.choice(items, size=len(items), replace=True)

        # Calculate metrics for each question in this resample
        sample_scores = []
        for row in sample:
            row_scores = {}
            for metric_name, metric_fn in metrics:
                for subset_k in k:
                    metric_key = f"{metric_name}@{subset_k}"
                    row_scores[metric_key] = metric_fn(
                        row["preds"][:subset_k],
                        row["labels"]
                    )
            sample_scores.append(row_scores)

        # Average across all questions in this resample
        sample_df = pd.DataFrame(sample_scores)
        means_list.append(sample_df.mean().to_dict())
        stds_list.append(sample_df.std().to_dict())

    # Return 1,000 rows of statistics
    return pd.DataFrame(means_list), pd.DataFrame(stds_list)
```

**What you get back:**
- `means_df`: 1,000 rows × metrics (each row = one simulated experiment)
- `stds_df`: 1,000 rows × metrics (standard deviations for each simulation)

---

#### **IMPORTANT: Why We Fetch Raw Predictions from Braintrust**

**Your question:** "Wait, doesn't Braintrust already store the calculated metrics (recall@10, mrr@5, etc.)? Why recalculate them?"

**Short answer:** For **flexibility** - you can test different k values without re-running experiments.

**The Two-Phase Process:**

**Phase 1: Running Experiments (`02_bench_retrieve.py`)**
```python
# This runs ONCE when you test your RAG system
def evaluate_braintrust(input, output, **kwargs):
    predictions = [item["id"] for item in output]  # Top 40 chunk IDs
    labels = [kwargs["metadata"]["chunk_id"]]      # Correct answer

    # Calculate metrics at k = [1, 3, 5, 10, 15, 20, ...]
    for metric, score_fn in metrics:
        for subset_k in k:
            scores.append(
                Score(name=f"{metric}@{subset_k}",
                      score=score_fn(predictions[:subset_k], labels))
            )
    return scores

# Braintrust stores for EACH question:
# {
#   "output": [{"id": "chunk_123"}, {"id": "chunk_456"}, ...],  ← Raw predictions
#   "scores": {"recall@1": 0.0, "recall@10": 1.0, ...},        ← Pre-calculated
#   "metadata": {"chunk_id": "chunk_456"}
# }
```

**Phase 2: Analyzing Results (`03_visualise_results.py`)**
```python
# This can run MULTIPLE times with different k values
def get_results(self):
    return [
        {
            "preds": [item["id"] for item in row["output"]],  # ← Fetch RAW predictions
            "labels": [row["metadata"]["chunk_id"]],
        }
        for row in braintrust.init(...)
    ]

# Then recalculate with YOUR chosen k values:
k = [1, 5, 10, 20]  # Could be different from Phase 1!

for subset_k in k:
    row_scores[metric_key] = metric_fn(
        row["preds"][:subset_k],  # ← Slice at ANY k value
        row["labels"]
    )
```

**Why This Design?**

**Scenario 1: You want to test NEW k values**
```python
# Originally tested in Phase 1:
k = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]

# Later, you want to analyze:
k = [7, 13, 21]  # Different k values!

# With raw predictions: ✅ Works! Just recalculate
recall@7 = calculate_recall(preds[:7], labels)

# With only pre-calculated scores: ❌ FAIL! You'd need to re-run experiments
```

**Scenario 2: You want to add NEW metrics**
```python
# Originally calculated: recall@k, mrr@k

# Later, you want to add:
def calculate_precision(predictions, labels):
    # New metric you didn't test in Phase 1!
    ...

# With raw predictions: ✅ Works! Just add the new metric function
# With only pre-calculated scores: ❌ FAIL! You'd need to re-run experiments
```

**Visual Example:**

```
BRAINTRUST STORAGE:
┌─────────────────────────────────────────┐
│ Question 1:                              │
│   Raw Predictions: [chunk_A, chunk_B,   │
│                     chunk_C, ..., chunk_Z] (40 chunks)│
│   ↓                                      │
│   Pre-calculated:                        │
│     recall@1 = 0.0                       │
│     recall@3 = 1.0                       │
│     recall@5 = 1.0                       │
│     ...                                  │
└─────────────────────────────────────────┘

WHAT GET_RESULTS() FETCHES:
┌─────────────────────────────────────────┐
│ {"preds": [chunk_A, chunk_B, ..., chunk_Z],│
│  "labels": [chunk_B]}                    │
└─────────────────────────────────────────┘
        ↓
    WHY? Flexibility!
        ↓
Can calculate ANY k value:
  recall@7  = calculate_recall(preds[:7], labels)
  recall@13 = calculate_recall(preds[:13], labels)
  recall@99 = calculate_recall(preds[:99], labels)
```

**Trade-off:**
- ✅ **Flexibility:** Test new k values and metrics without re-running experiments
- ✅ **Debugging:** Inspect raw predictions to understand failures
- ❌ **Redundancy:** Metrics are calculated twice (once in Phase 1, again in Phase 2)
- ❌ **Slightly slower:** Recalculation takes time during bootstrap

**Bottom line:** The code fetches raw predictions to decouple experiment execution from analysis, giving you maximum flexibility for future investigation.

### Step 4: Calculate Confidence Intervals

**What we have so far:**
- From Step 3, we have `means_df` with 1,000 rows (1,000 bootstrap samples)
- Each row contains metrics like `recall@10`, `recall@5`, etc.
- Example: `means_df['recall@10']` = [0.85, 0.83, 0.87, 0.84, ..., 0.86] (1,000 values)

**What we need for visualization:**
- A single "best estimate" for each metric
- Upper and lower bounds (confidence intervals) for the shaded area in plots

```python
def calculate_bootstrap_stats(means_df, stds_df, label):
    results = {}
    ci_lower = {}
    ci_upper = {}

    for col in means_df.columns:  # For each metric (recall@1, recall@3, etc.)
        # STEP 1: Calculate the "best estimate" across all 1,000 bootstrap samples
        results[col] = means_df[col].mean()
        # Example: recall@10 = mean([0.85, 0.83, 0.87, ...]) = 0.850

        # STEP 2: Calculate 95% confidence interval using percentiles
        ci_lower[col] = np.percentile(means_df[col], 2.5)   # 2.5th percentile
        ci_upper[col] = np.percentile(means_df[col], 97.5)  # 97.5th percentile
        # Example: recall@10 lower = 0.820, upper = 0.880

    return BootstrapResults(
        label=label,          # "text-embedding-3-small"
        means=results,        # {"recall@10": 0.850, "recall@5": 0.782, ...}
        ci_lower=ci_lower,    # {"recall@10": 0.820, "recall@5": 0.750, ...}
        ci_upper=ci_upper,    # {"recall@10": 0.880, "recall@5": 0.815, ...}
        stds=stds             # Variability measures
    )
```

**The 95% confidence interval means:**
"If I ran this experiment 100 times, 95 times the result would fall between ci_lower and ci_upper"

**Concrete example:**
```python
# Input: means_df['recall@10'] contains 1,000 bootstrap values:
[0.85, 0.83, 0.87, 0.84, 0.86, 0.82, 0.88, 0.85, 0.84, 0.87, ...]

# Output:
means['recall@10'] = 0.850       # Average of 1,000 values
ci_lower['recall@10'] = 0.820    # 25th lowest value out of 1,000
ci_upper['recall@10'] = 0.880    # 25th highest value out of 1,000

# Interpretation:
# "recall@10 is 0.850, but could reasonably be anywhere from 0.820 to 0.880"
```

### Step 5: Visualize Results

**What happens in this step:**
We take the compact statistics from Step 4 (`BootstrapResults`) and turn them into a visual plot with lines and shaded areas.

```python
def plot_recall_at_k(bootstrap_results):
    plt.figure(figsize=(12, 6))

    # bootstrap_results is a LIST of BootstrapResults objects
    # One for each experiment (e.g., [small_model_results, large_model_results])
    for results in bootstrap_results:
        k_values = [1, 3, 5, 10, 15, 20]

        # Extract the data for plotting
        recall_means = [results.means[f"recall@{k}"] for k in k_values]
        # Example: [0.75, 0.82, 0.85, 0.87, 0.89, 0.90]

        recall_lower = [results.ci_lower[f"recall@{k}"] for k in k_values]
        # Example: [0.70, 0.78, 0.82, 0.84, 0.86, 0.87]

        recall_upper = [results.ci_upper[f"recall@{k}"] for k in k_values]
        # Example: [0.80, 0.86, 0.88, 0.90, 0.92, 0.93]

        # Plot the line connecting the best estimates
        plt.plot(k_values, recall_means, marker="o", label=results.label)

        # Plot the shaded area showing uncertainty
        # This creates the "fat layer" between ci_lower and ci_upper
        plt.fill_between(k_values, recall_lower, recall_upper, alpha=0.2)

    plt.xlabel("k")
    plt.ylabel("Recall")
    plt.title("Recall@k with 95% Confidence Intervals")
    plt.legend()
    plt.grid(True)
```

**Visual breakdown of what gets plotted:**

```
Recall
  1.0 ┤
      │                                    ╱─────  ← recall_upper (e.g., 0.93)
  0.9 ┤                           ╱───────●
      │                  ╱────────●        │       ← Shaded area (uncertainty)
  0.8 ┤         ╱───────●         │       │
      │╱───────●        │        │       ╱─────  ← recall_lower (e.g., 0.87)
  0.7 ┤●       │        │        │
      └────────┴────────┴────────┴────────┴──────
       k=1    k=3     k=5     k=10    k=20

  ● = recall_means (best estimate)
  Shaded area = uncertainty (between ci_lower and ci_upper)
```

**How to read the plot:**

**1. The Line (dots connected):**
- Your best estimate of performance at each k value
- Example: "At k=10, recall is approximately 0.87"

**2. The Shaded Area (the "fat layer"):**
- Shows where the true value probably lies
- **Wide shaded area** = High uncertainty (results vary a lot between bootstrap samples)
- **Narrow shaded area** = Low uncertainty (results are stable)

**3. Comparing Two Models:**
```
Model A:  ═══════  (blue line + light blue shading)
Model B:  ═══════  (orange line + light orange shading)

Case 1: Non-overlapping shaded areas
    Model A: ────────
    Model B:         ────────
    ✅ Model B is CLEARLY better

Case 2: Overlapping shaded areas
    Model A: ────────
    Model B:    ────────
    ⚠️  Models might not be significantly different (check p-value!)

Case 3: Partially overlapping
    Model A: ────────
    Model B:      ────────
    ⚠️  Model B is probably better, but check p-value for confirmation
```

**Real example with numbers:**
```python
# After running bootstrap_samples and calculate_bootstrap_stats:

text-embedding-3-small:
  recall@10: mean=0.847, ci_lower=0.820, ci_upper=0.874

text-embedding-3-large:
  recall@10: mean=0.851, ci_lower=0.825, ci_upper=0.877

# The plot shows:
# - Small model line at 0.847 with shading from 0.820 to 0.874
# - Large model line at 0.851 with shading from 0.825 to 0.877
# - The shadings OVERLAP heavily → improvement might be noise!
# - Need to check p-value to confirm
```

### Step 6: Test Statistical Significance

**The final question:** "Is the difference between Model A and Model B real, or just random luck?"

The visualization helps, but you need a definitive answer. This is where the **paired t-test** comes in.

```python
# Get the 1,000 bootstrap samples for each experiment
result1_samples = bootstrap_samples(config1)[0]  # means_df (1,000 rows)
result2_samples = bootstrap_samples(config2)[0]  # means_df (1,000 rows)

# Run paired t-test for each metric
for metric in result1_samples.columns:
    values_model1 = result1_samples[metric]  # 1,000 bootstrap values
    values_model2 = result2_samples[metric]  # 1,000 bootstrap values

    # Paired t-test: tests if the DIFFERENCE between paired samples is significant
    t_stat, p_value = stats.ttest_rel(values_model1, values_model2)

    print(f"{metric}: p-value = {p_value:.4f}")
```

**What's happening behind the scenes:**

```python
# Example: Testing recall@10 between two models

Model 1 (small): [0.85, 0.83, 0.87, 0.84, 0.86, ...]  # 1,000 values
Model 2 (large): [0.86, 0.84, 0.88, 0.85, 0.87, ...]  # 1,000 values

# Step 1: Calculate differences for each bootstrap sample
differences = [0.86-0.85, 0.84-0.83, 0.88-0.87, ...]
            = [0.01, 0.01, 0.01, 0.01, 0.01, ...]  # 1,000 differences

# Step 2: Check if the average difference is significantly different from 0
average_difference = mean([0.01, 0.01, 0.01, ...]) = 0.01

# Step 3: T-test asks: "Is this 0.01 difference real, or could it be noise?"
# If most bootstrap samples show Model 2 > Model 1 → p-value will be LOW (significant)
# If differences are inconsistent (sometimes positive, sometimes negative) → p-value will be HIGH (not significant)
```

**Interpreting p-values:**
- **p < 0.01:** Highly significant (99% confident the difference is real)
- **p < 0.05:** Significant (95% confident the difference is real) ← **Standard threshold**
- **p > 0.05:** Not significant (could just be random noise - DON'T deploy!)

**What p-value actually means:**
- `p = 0.03` means: "If there were NO real difference between models, there's only a 3% chance we'd see results this different just by random luck"
- Low p-value = "Very unlikely this is just luck, the difference is probably real"
- High p-value = "This could easily happen by chance, no real difference"

**Example output:**
```
recall@1: p-value = 0.0143  ✅ Significant! Deploy the change.
recall@5: p-value = 0.1820  ❌ Not significant. Improvement is probably noise.
recall@10: p-value = 0.0003 ✅ Highly significant! Definitely deploy.
```

**Why we use "paired" t-test (`ttest_rel`):**
- We're comparing the SAME bootstrap samples from both models
- Bootstrap sample #1 uses the same resampled questions for both models
- Bootstrap sample #2 uses the same resampled questions for both models
- This "pairing" makes the test more sensitive to real differences

**Alternative approach (NOT recommended):**
```python
# WRONG: Independent t-test (ignores pairing)
t_stat, p_value = stats.ttest_ind(values_model1, values_model2)

# WHY WRONG?
# This treats the 1,000 samples as completely independent
# Ignores that bootstrap sample #1 for Model A and Model B used the same questions
# Less powerful, might miss real differences
```

**Concrete example with real numbers:**

```python
# After running both experiments and bootstrapping:

Small model - recall@10 bootstrap distribution (1,000 values):
[0.847, 0.843, 0.851, 0.845, 0.849, ..., 0.848]
Mean: 0.847

Large model - recall@10 bootstrap distribution (1,000 values):
[0.851, 0.847, 0.855, 0.849, 0.853, ..., 0.852]
Mean: 0.851

# Differences (paired):
[0.004, 0.004, 0.004, 0.004, 0.004, ..., 0.004]
Average difference: 0.004

# T-test result:
t_statistic = 1.23
p_value = 0.234

# Interpretation:
# The large model is 0.004 better on average, BUT
# p = 0.234 > 0.05 → NOT statistically significant
# Decision: DON'T switch to large model, the improvement is likely just noise
```

---

## Part 3: Complete Example Workflow

```python
# 1. Define configs
config_small = ExperimentConfig(
    experiment_id="uuid-small",
    project_name="RAG-System",
    label="text-embedding-3-small"
)
config_large = ExperimentConfig(
    experiment_id="uuid-large",
    project_name="RAG-System",
    label="text-embedding-3-large"
)

# 2. Run bootstrapping (this takes ~30 seconds per config)
bootstrap_dfs = [
    bootstrap_samples(config_small, n_samples=1000),
    bootstrap_samples(config_large, n_samples=1000)
]

# 3. Calculate statistics
bootstrap_results = [
    calculate_bootstrap_stats(dfs[0], dfs[1], config.label)
    for dfs, config in zip(bootstrap_dfs, [config_small, config_large])
]

# 4. Visualize
plot_recall_at_k(bootstrap_results)
plt.savefig("recall_comparison.png")

# 5. Test significance
result1, result2 = [dfs[0] for dfs in bootstrap_dfs]
for metric in result1.columns:
    t_stat, p_value = stats.ttest_rel(result1[metric], result2[metric])
    if p_value < 0.05:
        print(f"✅ {metric}: SIGNIFICANT (p={p_value:.4f})")
    else:
        print(f"❌ {metric}: NOT significant (p={p_value:.4f})")
```

---

## Part 4: When to Use Bootstrapping (Decision Guide)

### Quick Decision Tree

```
Should I use bootstrapping for this comparison?
    ↓
Q1: Is the difference OBVIOUS (>20% improvement)?
    YES → ❌ Skip bootstrapping, just deploy
    NO → Continue
    ↓
Q2: Do I have >10,000 test samples?
    YES → ❌ Skip bootstrapping, simple averages are fine
    NO → Continue
    ↓
Q3: Is this decision high-stakes (costs money/time/complexity)?
    NO → ❌ Skip bootstrapping, just deploy and monitor
    YES → Continue
    ↓
Q4: Do I have at least 50 test samples?
    NO → ⚠️  Collect more data first!
    YES → ✅ USE BOOTSTRAPPING
```

### Use Bootstrapping When:

| Scenario | Example | Why |
|----------|---------|-----|
| **A/B testing with costs** | Testing expensive model vs cheap model | Need proof to justify cost |
| **Small datasets** | 50-500 test questions | High uncertainty, need confidence intervals |
| **Close results** | Model A: 0.847, Model B: 0.851 (0.4% diff) | Can't tell if difference is real |
| **Noisy metrics** | LLM-judge scores, human ratings | Need to quantify uncertainty |
| **Justifying to stakeholders** | Presenting to management | Need rigorous statistical proof |

### Skip Bootstrapping When:

| Scenario | Example | Why |
|----------|---------|-----|
| **Huge differences** | Model A: 0.45, Model B: 0.89 (2x better!) | Obviously better, no stats needed |
| **Massive datasets** | 50,000+ test samples | Simple averages already stable |
| **Exploratory testing** | Quickly testing 5 prototypes | Too slow for rapid iteration |
| **Zero cost/risk** | Free upgrade, easy rollback | Just deploy and monitor |
| **Wrong test domain** | Testing on synthetic data for production | Stats won't transfer anyway |

### Sample Size Guidelines

**The Paradox:** Small samples need bootstrapping most BUT give the widest confidence intervals.

```python
# Example: Testing with only 20 questions

Bootstrap result:
  Model A: recall@10 = 0.85, CI = [0.65, 0.95]  ← HUGE uncertainty!
  Model B: recall@10 = 0.87, CI = [0.67, 0.97]  ← HUGE uncertainty!
  p-value = 0.45  ← Not significant

Translation: "With only 20 questions, you don't have enough data to conclude anything."
```

**Minimum Sample Sizes:**

| Sample Size | Confidence Interval Width | Recommendation |
|-------------|---------------------------|----------------|
| < 30 | Very wide (±0.15-0.20) | ⛔ Too small - collect more data |
| 30-50 | Wide (±0.10-0.15) | ⚠️  Marginal - be cautious with conclusions |
| 50-100 | Moderate (±0.05-0.10) | ✅ Reasonable - bootstrapping works |
| 100-200 | Narrow (±0.03-0.05) | ✅ Good - reliable results |
| 200+ | Very narrow (±0.02-0.03) | ✅ Excellent - high confidence |
| 10,000+ | Tiny (±0.005) | ⭐ Skip bootstrapping - simple avg is enough |

**What to do with small samples:**

1. **Best option:** Collect more data
   ```python
   "I have 20 questions but need 50+ for reliable results.
   Let me create 30 more test cases before deciding."
   ```

2. **If you can't collect more:** Use bootstrap but acknowledge limitations
   ```python
   "Results with 20 samples (⚠️  SMALL SAMPLE):
   - Model B shows 2% improvement
   - p-value = 0.45 (not significant)
   - Recommendation: Inconclusive - need 30+ more samples"
   ```

3. **Alternative:** Manual qualitative analysis
   ```python
   "With only 20 samples, let me manually analyze WHERE Model B is better:
   - Better on complex queries (15/20)
   - Worse on simple queries (3/20)
   - Hypothesis: B better for complex queries (test 50 more complex queries)"
   ```

---

## Part 5: Common Questions & Pitfalls

### Q1: How many bootstrap samples do I need?

**Answer:** 1,000 is the standard. Don't go below 500. Going above 10,000 wastes compute without much benefit.

### Q2: What if my results are NOT statistically significant (p > 0.05)?

**This is VERY common!** Here's what to do:

**Step 1: Understand what it means**
```
p = 0.234 means:
"If the models were actually identical, there's a 23.4% chance
we'd see this difference just by luck. Not convincing enough."
```

**Step 2: Choose your action**

| Your Situation | p-value | Action |
|----------------|---------|--------|
| Small sample (< 100) | > 0.05 | ⏸️  Collect more data, then re-test |
| Large sample (100+), high cost | > 0.05 | ❌ Don't deploy - improvement is noise |
| Large sample (100+), zero cost | > 0.05 | ✅ Deploy anyway - low risk, possible upside |
| Tiny improvement anyway | < 0.05 | ⚠️  Significant but not worth it |

**Step 3: Example decision framework**

```python
# Scenario: text-embedding-3-large vs small
# Result: 0.4% improvement, p = 0.234, large costs 6.5x more

if p_value >= 0.05 and cost_increase > 1.5:
    decision = "DON'T DEPLOY"
    reason = "Not statistically significant + too expensive"

elif p_value >= 0.05 and cost_increase == 0:
    decision = "DEPLOY with monitoring"
    reason = "Not significant BUT no cost penalty, low risk"

elif p_value < 0.05 and improvement < 0.02:
    decision = "EVALUATE business case"
    reason = "Significant but improvement is tiny - worth the cost?"
```

**Step 4: Don't p-hack!**

❌ **WRONG:** "Let me try different metrics until I find p < 0.05..."
✅ **RIGHT:** "p > 0.05 means no proof. Accept it and move on."

### Q3: Can I compare more than 2 experiments?

**Yes!** Just add more configs to your list. The visualization will show multiple lines with shaded areas. For statistical testing, compare pairs:
- Model A vs Model B
- Model A vs Model C
- Model B vs Model C

### Q4: What if my p-value is between 0.05-0.10?

**Borderline zone:**
- p = 0.06-0.10: Weakly significant (consider deploying if low risk)
- Collect more test data to reduce uncertainty
- Run the experiment again to confirm

### Q5: My confidence intervals overlap. Does that mean my improvement is useless?

**Not necessarily!** Check the p-value:
- Overlapping CIs + p > 0.05 → Improvement is probably noise
- Overlapping CIs + p < 0.05 → Improvement might still be real (run more tests)

### Q6: When should I NOT use bootstrapping?

**Don't use bootstrapping if:**
- You have < 30 test questions (not enough data to resample)
- Your test questions are not independent (e.g., all from the same document)
- You changed multiple variables at once (can't isolate what helped)
- The difference is huge (>20% improvement - obviously better)
- You have 10,000+ samples (simple averages are already stable)

### Q7: My metrics have high variance. What does that mean?

**High variance means:**
- Performance is inconsistent across questions
- Some questions work great, others fail badly
- Your system might need better handling of edge cases

**Fix by:**
- Analyzing which question types fail
- Adding more diverse training data
- Improving your chunking/retrieval strategy

### Q8: What about multiple comparisons? Testing 5 models at once?

**Be careful of multiple comparison problem!**

If you test 5 models and run 10 pairwise comparisons:
- By chance alone, you'd expect 0.5 comparisons to show p < 0.05 (false positives)
- Solution: Use Bonferroni correction: divide your threshold by number of comparisons
  - Instead of p < 0.05, use p < 0.005 (if doing 10 comparisons)
- Better: Test your top 2 candidates only (fewer comparisons)

---

## Part 6: Practical Decision Examples

### Example 1: Clear Deploy ✅

**Scenario:**
- Model A (small): recall@10 = 0.75
- Model B (large): recall@10 = 0.88 (+17% improvement!)
- Cost: Large is 6.5x more expensive
- Sample size: 100 questions
- p-value: 0.001

**Decision: DEPLOY Model B**

**Why:**
- Huge improvement (17%)
- Highly significant (p < 0.01)
- Worth the cost increase

---

### Example 2: Clear Reject ❌

**Scenario:**
- Model A (small): recall@10 = 0.847
- Model B (large): recall@10 = 0.851 (+0.4% improvement)
- Cost: Large is 6.5x more expensive
- Sample size: 100 questions
- p-value: 0.234

**Decision: DON'T DEPLOY Model B**

**Why:**
- Tiny improvement (0.4%)
- NOT significant (p > 0.05)
- Not worth 6.5x cost

---

### Example 3: Need More Data ⏸️

**Scenario:**
- Model A: recall@10 = 0.80
- Model B: recall@10 = 0.85 (+6% improvement)
- Cost: B is same price
- Sample size: 30 questions (SMALL!)
- p-value: 0.15
- Confidence intervals: Very wide

**Decision: COLLECT MORE DATA**

**Why:**
- Sample size too small (30 questions)
- Wide confidence intervals = high uncertainty
- Improvement looks promising (6%)
- No cost penalty
- Action: Test on 70 more questions (total 100)

---

### Example 4: Deploy Despite Non-Significance ✅

**Scenario:**
- Model A: recall@10 = 0.82
- Model B: recall@10 = 0.85 (+3.7% improvement)
- Cost: B is FREE (same API, same price)
- Sample size: 150 questions
- p-value: 0.08 (borderline)

**Decision: DEPLOY Model B with monitoring**

**Why:**
- Zero cost penalty
- Decent improvement (3.7%)
- Borderline significance (p = 0.08)
- Easy to rollback
- Low risk, potential upside

---

### Example 5: Significant But Not Worth It ⚠️

**Scenario:**
- Model A: recall@10 = 0.920
- Model B: recall@10 = 0.925 (+0.5% improvement)
- Cost: B adds 200ms latency per query
- Sample size: 200 questions
- p-value: 0.03 (significant!)

**Decision: DON'T DEPLOY Model B**

**Why:**
- Statistically significant BUT
- Improvement is tiny (0.5%)
- 200ms latency is unacceptable
- Already at 92% recall - diminishing returns
- Not worth the latency cost

---

## Part 7: Decision Framework

Use this flowchart to decide whether to deploy a change:

```
1. Run bootstrapping for both experiments
   ↓
2. Check confidence intervals on plot
   ↓
   Clearly separated?
   ├─ YES → ✅ Deploy (very confident)
   └─ NO → Continue to step 3
   ↓
3. Check p-value
   ↓
   p < 0.01?
   ├─ YES → ✅ Deploy (highly confident)
   └─ NO → Continue to step 4
   ↓
   p < 0.05?
   ├─ YES → ⚠️  Deploy (moderately confident)
   └─ NO → Continue to step 5
   ↓
5. p > 0.05
   └─ ❌ DON'T deploy (improvement is likely noise)
      Consider: More test data, better baseline, or different approach
```

---

## Part 8: Real-World Example

**Scenario:** You're deciding whether to upgrade from `text-embedding-3-small` ($0.02/1M tokens) to `text-embedding-3-large` ($0.13/1M tokens) - **6.5x more expensive**.

**Results after bootstrapping:**
```
Metric       Small (mean)  Large (mean)  p-value    Decision
recall@10    0.847         0.851         0.234      ❌ Not significant
mrr@10       0.723         0.745         0.003      ✅ Significant
recall@20    0.912         0.913         0.789      ❌ Not significant
```

**Analysis:**
- `recall@10` improved by 0.004 but p=0.234 → **just noise**
- `mrr@10` improved by 0.022 and p=0.003 → **real improvement!**
- `recall@20` barely changed → **not worth it**

**Decision:**
- If you care about ranking quality (`mrr`) → ✅ Deploy large model
- If you only care about recall → ❌ Stick with small model (save 6.5x cost)
- If cost is critical → ❌ 0.022 MRR improvement not worth 6.5x cost

---

## Part 9: Checklist Before Running

Before you run your statistical validation:

- [ ] I have at least 50 test questions (100+ is better)
- [ ] My test questions are diverse and representative
- [ ] I changed **only ONE variable** between experiments
- [ ] I have per-question results logged (not just aggregated metrics)
- [ ] I defined my success metrics clearly (recall@k, MRR, etc.)
- [ ] I know what p-value threshold I'll use (usually 0.05)
- [ ] I have a rollback plan if results are negative

---

## Summary

**What you learned:**
1. **Bootstrapping** gives you confidence intervals, not just point estimates
2. **Visualization** shows if improvements are meaningful (non-overlapping shaded areas)
3. **Statistical testing** gives you a yes/no answer (p < 0.05 = significant)
4. **Decision-making** combines all three: CIs + p-values + business context

**Key takeaway for engineers:**
> "Don't trust a single number. Get the distribution. Test significance. Make data-driven decisions."

**Next steps:**
1. Copy the code from `03_visualise_results.py`
2. Replace the experiment IDs with yours
3. Run it and interpret the results
4. Deploy only if p < 0.05 AND the business case makes sense

---

## Appendix: Complete Data Flow Diagram

Understanding how data transforms through each step:

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Run Experiments (02_bench_retrieve.py)            │
└─────────────────────────────────────────────────────────────┘

100 questions → RAG system retrieval
    ↓
For each question:
  Q1: predictions=[chunk_A, chunk_B, ..., chunk_Z] (40 chunks)
      correct_answer=[chunk_B]
      Pre-calculate: recall@1=0.0, recall@10=1.0, mrr@10=0.5, ...
  Q2: predictions=[...]
  ...
  Q100: predictions=[...]
    ↓
Store in Braintrust:
  - Raw predictions (40 chunks per question)
  - Pre-calculated metrics (for reference)
  - Metadata (correct answers)

┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Analyze & Validate (03_visualise_results.py)      │
└─────────────────────────────────────────────────────────────┘

STEP 1: Fetch Raw Data
    ↓
config.get_results() → [{preds: [...], labels: [...]}, ...] × 100 questions

STEP 2: Bootstrap Resampling (1,000 times)
    ↓
Bootstrap Sample #1: Resample 100 questions with replacement
  Example: [Q2, Q5, Q2, Q1, Q3, Q7, ...] (Q2 appears twice, Q4 is missing)
    ↓
  Calculate metrics for each question at different k values:
    Q2: recall@1=0.0, recall@5=1.0, recall@10=1.0, ...
    Q5: recall@1=1.0, recall@5=1.0, recall@10=1.0, ...
    Q2: recall@1=0.0, recall@5=1.0, recall@10=1.0, ... (duplicate!)
    ...
    ↓
  Average across 100 questions:
    recall@1 = 0.75
    recall@5 = 0.85
    recall@10 = 0.87
    ... (one row in means_df)

Bootstrap Sample #2: Different resampled questions
  → Different averages: recall@1=0.73, recall@5=0.84, recall@10=0.86

...repeat 998 more times...

Bootstrap Sample #1000:
  → Different averages: recall@1=0.76, recall@5=0.86, recall@10=0.88

    ↓
Result: means_df (1,000 rows × metrics columns)

STEP 3: Calculate Statistics
    ↓
For each metric (e.g., recall@10):
  - Take the 1,000 bootstrap values: [0.87, 0.86, 0.88, 0.85, ...]
  - Calculate:
      * mean = 0.850 (best estimate)
      * ci_lower = 0.820 (2.5th percentile)
      * ci_upper = 0.880 (97.5th percentile)
    ↓
Result: BootstrapResults object with means, ci_lower, ci_upper

STEP 4: Visualize
    ↓
For k = [1, 3, 5, 10, 15, 20]:
  - Plot line at means values: (1, 0.75), (3, 0.82), (5, 0.85), ...
  - Fill shaded area between ci_lower and ci_upper
    ↓
Result: Plot with lines and confidence interval bands

STEP 5: Statistical Testing
    ↓
Compare Model 1 vs Model 2:
  - Get 1,000 bootstrap values for each model
  - Model 1 recall@10: [0.85, 0.83, 0.87, ...]
  - Model 2 recall@10: [0.86, 0.84, 0.88, ...]
    ↓
  - Calculate differences: [0.01, 0.01, 0.01, ...]
  - Run paired t-test: Is average difference significantly > 0?
    ↓
Result: p-value (e.g., 0.234)
  - If p < 0.05 → Difference is significant! Deploy Model 2
  - If p > 0.05 → Difference is noise. Stick with Model 1
```

**Key transformation points:**

1. **Raw → Bootstrap Distribution**
   - From: 100 individual question results
   - To: 1,000 simulated experiment outcomes

2. **Bootstrap Distribution → Summary Stats**
   - From: 1,000 rows of averages
   - To: Single mean + confidence interval

3. **Summary Stats → Visual**
   - From: Numbers (mean=0.85, ci_lower=0.82, ci_upper=0.88)
   - To: Line plot with shaded area

4. **Bootstrap Distributions → p-value**
   - From: Two sets of 1,000 values
   - To: Single number answering "Is this different?"

Good luck! 🚀
