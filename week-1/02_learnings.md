<!--
Document Type: Learning Notes
Purpose: Comprehensive guide on building scalable retrieval pipeline evaluation systems
Context: Week 1 - Learning from 02_bench_retrieve.py sample implementation
Key Topics: Evaluation architecture, metrics design, configuration management, tidy data format, result analysis
Target Use: Reference guide for engineers implementing retrieval evaluation pipelines
-->

# Building Scalable Retrieval Pipeline Evaluations

## Overview

This document captures key learnings from building a scalable retrieval evaluation system. The goal is to create an evaluation framework where **adding new configurations requires minimal code changes** - just add entries to configuration dictionaries/lists.

## Core Design Principles

### 1. **Separation of Concerns**

Separate your evaluation into distinct, reusable components:

```python
#  Good: Separate metric functions
def calculate_recall(predictions: list[str], gt: list[str]):
    return len([label for label in gt if label in predictions]) / len(gt)

def calculate_mrr(predictions: list[str], gt: list[str]):
    mrr = 0
    for label in gt:
        if label in predictions:
            mrr = max(mrr, 1 / (predictions.index(label) + 1))
    return mrr

# Store them as tuples for easy iteration
metrics = [("recall", calculate_recall), ("mrr", calculate_mrr)]
```

**Why this matters:**
- Each metric is independently testable
- Easy to add new metrics without touching evaluation logic
- Metrics can be reused across different evaluation frameworks

---

### 2. **Configuration-Driven Evaluation**

Define all evaluation parameters as data structures that can be easily modified:

```python
# Configuration: just add/remove entries here
metrics = [("recall", calculate_recall), ("mrr", calculate_mrr)]
k = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]

# Inside your evaluation function:
available_rerankers = {
    "rerank-english-v3.0": CohereReranker(
        model_name="rerank-english-v3.0", column="query"
    ),
    "none": None,
}

search_query_modes = ["hybrid", "vector"]

embedding_model_to_table = {
    "text-embedding-3-small": table_small,
    "text-embedding-3-large": table_large,
}
```

**Adding a new configuration is trivial:**
```python
# Want to test another Cohere reranker? Just add one entry:
available_rerankers = {
    "rerank-english-v3.0": CohereReranker(
        model_name="rerank-english-v3.0", column="query"
    ),
    "rerank-multilingual-v3.0": CohereReranker(
        model_name="rerank-multilingual-v3.0", column="query"
    ),
    "none": None,
}

# Want to test FTS search? Just add to the list:
search_query_modes = ["hybrid", "vector", "fts"]

# Want to test a new embedding model? Just add one entry:
embedding_model_to_table = {
    "text-embedding-3-small": table_small,
    "text-embedding-3-large": table_large,
    "text-embedding-ada-002": table_ada,  # New model
}
```

---

### 3. **Wrapper Functions for Framework Integration**

Evaluation frameworks (like Braintrust) have specific interfaces. Use wrapper functions to adapt your metrics:

```python
def evaluate_braintrust(input, output, **kwargs):
    """
    Braintrust scoring wrapper that generates all metric@k combinations.

    Purpose: Adapts our metric functions to Braintrust's required interface.
    Returns: List of Score objects (2 metrics � 10 k values = 20 scores per evaluation)
    """
    predictions = [item["id"] for item in output]
    labels = [kwargs["metadata"]["chunk_id"]]

    scores = []
    # Generate all combinations dynamically
    for metric, score_fn in metrics:
        for subset_k in k:
            scores.append(
                Score(
                    name=f"{metric}@{subset_k}",
                    score=score_fn(predictions[:subset_k], labels),
                    metadata={"query": input, "result": output, **kwargs["metadata"]},
                )
            )

    return scores
```

**Key insight:**
- The wrapper creates **all metric@k combinations** (recall@1, recall@3, ..., mrr@1, mrr@3, etc.)
- This allows measuring performance at different cutoff points **without re-running retrieval**
- Retrieve once with `max_k=80`, then compute metrics for k=1,3,5,...,40 from the same results

---

### 4. **Systematic Configuration Combination Testing**

Use `itertools.product()` to test **all combinations** of configurations:

```python
from itertools import product

# This automatically tests ALL combinations:
# 2 rerankers � 2 search modes � 2 embedding models = 8 configurations
for reranker_name, search_mode, embedding_model in product(
    available_rerankers,        # Dict: iterates over keys
    search_query_modes,          # List: iterates over elements
    embedding_model_to_table     # Dict: iterates over keys
):
    current_reranker = available_rerankers[reranker_name]
    current_table = embedding_model_to_table[embedding_model]

    # Run evaluation for this specific configuration
    benchmark_result = await Eval(
        name="Text-2-SQL",
        experiment_name=f"{experiment_id}-{reranker_name}-{search_mode}-{embedding_model}",
        task=lambda query: retrieve(
            question=query,
            max_k=80,
            table=current_table,
            mode=search_mode,
            reranker=current_reranker,
        ),
        data=evaluation_queries,
        scores=[evaluate_braintrust],
        metadata={
            "embedding_model": embedding_model,
            "reranker": reranker_name,
            "query_mode": search_mode,
        },
    )
```

**Important note about `product()` with dicts:**
- When iterating over a dictionary in Python, you get the **keys** (not values)
- `product()` works with any iterable (lists, dicts, tuples, etc.)
- This is why we can use dict keys as lookup values later: `available_rerankers[reranker_name]`

---

## Tidy Data Format for Results

### What is Tidy Data?

Following the [tidy data principles](https://kiwidamien.github.io/what-is-tidy-data.html):
1. **Each column represents a single variable**
2. **Each row represents a single observation**

### Why This Matters

Tidy format makes analysis and visualization **dramatically easier**:

```python
# Process results into tidy format
performance_scores = benchmark_result.summary.scores
for metric_name, score_data in performance_scores.items():
    metric_type, top_k = metric_name.split("@")  # Parse "recall@5" -> ("recall", "5")
    evaluation_results.append({
        "metric": metric_type,           # Single variable per column
        "k": int(top_k),                 # Single variable per column
        "reranker": reranker_name,       # Single variable per column
        "embedding_model": embedding_model,  # Single variable per column
        "query_type": search_mode,       # Single variable per column
        "score": score_data.score,       # Single variable per column
    })
```

**Result DataFrame looks like:**
```
   metric  k           reranker        embedding_model  query_type  score
0  recall  1   rerank-english-v3.0  text-embedding-3-small  hybrid   0.85
1  recall  3   rerank-english-v3.0  text-embedding-3-small  hybrid   0.92
2  recall  5   rerank-english-v3.0  text-embedding-3-small  hybrid   0.95
...
```

### Benefits of Tidy Format

1. **Easy filtering:**
   ```python
   # Get all hybrid search results
   hybrid_data = df[df["query_type"] == "hybrid"]

   # Get specific configuration
   config_data = df[
       (df["reranker"] == "rerank-english-v3.0") &
       (df["embedding_model"] == "text-embedding-3-large") &
       (df["metric"] == "recall")
   ]
   ```

2. **Simple plotting:**
   ```python
   # Plot performance curves for different configurations
   for model in df["embedding_model"].unique():
       for reranker in df["reranker"].unique():
           data = filtered_data[
               (filtered_data["reranker"] == reranker) &
               (filtered_data["embedding_model"] == model)
           ]
           ax.plot(data["k"], data["score"], label=f"{model}\n{reranker}")
   ```

3. **Easy to add new observations:**
   - Just append a new row with the same column structure
   - No need to reshape or restructure existing data

---

## Result Analysis and Visualization

### Comparative Analysis Setup

The script creates **side-by-side comparisons** to understand the impact of different configurations:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Compare: Hybrid vs Vector search
# For each subplot, iterate through all model/reranker combinations
for model in sorted(df["embedding_model"].unique()):
    for reranker in sorted(df["reranker"].unique()):
        data = filtered_data[
            (filtered_data["reranker"] == reranker) &
            (filtered_data["embedding_model"] == model)
        ]
        if len(data) > 0:
            # Use different markers to distinguish configurations
            marker = "o" if reranker != "none" else "^"
            ax.plot(data["k"], data["score"], marker=marker, label=f"{model}\n{reranker}")
```

### Visualization Best Practices

1. **Use markers to differentiate:**
   - Circles (`"o"`) for reranked results
   - Triangles (`"^"`) for non-reranked results
   - Makes it easy to visually identify patterns

2. **Group comparisons logically:**
   - One subplot per search mode (hybrid vs vector)
   - Allows direct comparison of the same metric under different conditions

3. **Clear labeling:**
   - Include configuration in legend: `f"{model}\n{reranker}"`
   - Set descriptive titles: `"Recall@k for Hybrid Search"`
   - Label axes clearly: `"k"`, `"Recall"`

---

## Complete Evaluation Workflow

### Step 1: Define Your Metrics
```python
def calculate_metric(predictions, ground_truth):
    # Implement your metric logic
    return score

metrics = [("metric_name", calculate_metric), ...]
```

### Step 2: Define Evaluation Parameters
```python
k = [1, 3, 5, 10, 15, 20]  # Cutoff values to test
```

### Step 3: Define Configurations
```python
# All configurations you want to test
rerankers = {"none": None, "cohere": CohereReranker(), ...}
search_modes = ["vector", "hybrid", "fts"]
embedding_models = {"small": table_small, "large": table_large, ...}
```

### Step 4: Create Framework Wrapper
```python
def evaluate_framework(input, output, **kwargs):
    predictions = extract_predictions(output)
    labels = extract_labels(kwargs)

    scores = []
    for metric, score_fn in metrics:
        for subset_k in k:
            scores.append(FrameworkScore(
                name=f"{metric}@{subset_k}",
                score=score_fn(predictions[:subset_k], labels)
            ))
    return scores
```

### Step 5: Run All Combinations
```python
results = []
for reranker, search_mode, model in product(rerankers, search_modes, embedding_models):
    result = run_eval(
        config=(reranker, search_mode, model),
        scorer=evaluate_framework
    )

    # Store in tidy format
    for metric_name, score_data in result.scores.items():
        metric_type, top_k = metric_name.split("@")
        results.append({
            "metric": metric_type,
            "k": int(top_k),
            "reranker": reranker,
            "embedding_model": model,
            "query_type": search_mode,
            "score": score_data.score,
        })
```

### Step 6: Analyze Results
```python
df = pd.DataFrame(results)

# Filter and plot
for config in configurations:
    filtered = df[df["config"] == config]
    plt.plot(filtered["k"], filtered["score"], label=config)
```

---

## Key Takeaways

### For Scalability:
1.  Define metrics as separate, reusable functions
2.  Store configurations in data structures (dicts/lists)
3.  Use `itertools.product()` to generate all combinations
4.  Create wrapper functions to adapt to framework interfaces
5.  Store results in tidy data format

### For Analysis:
1.  Use tidy data format (one observation per row)
2.  Separate variables into individual columns
3.  Enable easy filtering with pandas
4.  Create comparative visualizations
5.  Use visual markers to distinguish configurations

### Adding New Configurations:
```python
# Just add to the configuration dictionaries/lists:
metrics.append(("precision", calculate_precision))
k.append(50)
rerankers["new-reranker"] = NewReranker()
search_modes.append("bm25")
embedding_models["new-model"] = new_table

# No other code changes needed! <�
```

---

## Common Pitfalls to Avoid

### L Don't hardcode configurations
```python
# Bad: Have to duplicate code for each config
result1 = eval(reranker="cohere", model="small")
result2 = eval(reranker="cohere", model="large")
result3 = eval(reranker="none", model="small")
# ... this doesn't scale!
```

### L Don't create nested/wide data formats
```python
# Bad: Hard to analyze
{
    "cohere": {
        "small": {"recall@5": 0.85, "recall@10": 0.92},
        "large": {"recall@5": 0.88, "recall@10": 0.94}
    }
}

# Good: Tidy format (flat, one observation per row)
[
    {"reranker": "cohere", "model": "small", "metric": "recall", "k": 5, "score": 0.85},
    {"reranker": "cohere", "model": "small", "metric": "recall", "k": 10, "score": 0.92},
    ...
]
```

### L Don't rerun retrieval for each k value
```python
# Bad: Inefficient
for k_val in [5, 10, 20]:
    results = retrieve(query, max_k=k_val)  # Retrieves k_val items
    score = calculate_metric(results)

# Good: Retrieve once, evaluate multiple k values
results = retrieve(query, max_k=80)  # Retrieve once
for k_val in [5, 10, 20, 30, 40]:
    score = calculate_metric(results[:k_val])  # Slice existing results
```

---

## Additional Resources

- [Tidy Data Principles](https://kiwidamien.github.io/what-is-tidy-data.html)
- [Python itertools.product() Documentation](https://docs.python.org/3/library/itertools.html#itertools.product)
- [Pandas DataFrame Filtering Guide](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)
