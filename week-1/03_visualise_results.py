"""
Help me have a solid understanding on how to validate your rack improvements and understand the impact of different techniques on your system's performance.
"""

from dataclasses import dataclass
from typing import Dict, List

import braintrust
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


# I have code that's easily adaptable if you want to compare results of different experiments. So a good approach is to define a configuration plus that can easily scale to other experiments as seen below. This makes adding new experiments to compare as simple as adding a new configuration object. Because we have logged our experiment results with BrainTrust, it's easy to fetch the results.
@dataclass
class ExperimentConfig:
    experiment_id: str
    project_name: str
    label: str

    def get_results(self):
        return [
            {
                "preds": [item["id"] for item in row["output"]],
                "labels": [row["metadata"]["chunk_id"]],
            }
            for row in braintrust.init(
                project=self.project_name, experiment=self.experiment_id, open=True
            )
            if row["root_span_id"] == row["span_id"]
        ]


configs = [
    ExperimentConfig(
        experiment_id="85944371-4b07-4c2b-a862-a6af7bbcfda5-none-vector-text-embedding-3-small",
        project_name="Text-2-SQL",
        label="text-embedding-3-small",
    ),
    ExperimentConfig(
        experiment_id="85944371-4b07-4c2b-a862-a6af7bbcfda5-none-vector-text-embedding-3-large",
        project_name="Text-2-SQL",
        label="text-embedding-3-large",
    ),
]

# Define some helper functions to calculate mrr and recall


def calculate_mrr(predictions: list[str], gt: list[str]):
    mrr = 0
    for label in gt:
        if label in predictions:
            # Find the relevant item that has the smallest index
            mrr = max(mrr, 1 / (predictions.index(label) + 1))
    return mrr


def calculate_recall(predictions: list[str], gt: list[str]):
    # Calculate the proportion of relevant items that were retrieved
    return len([label for label in gt if label in predictions]) / len(gt)


metrics = [("recall", calculate_recall), ("mrr", calculate_mrr)]
k = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]

# Simulate multiple runs with bootstrapping


def bootstrap_samples(config: ExperimentConfig, n_samples: int = 1000):
    """
    Create bootstrap distribution of metrics to estimate uncertainty in experiment results.

    BOOTSTRAPPING CONCEPT:
    Instead of getting one single "recall@10 = 0.85", we want to know:
    "How confident are we in this 0.85? Could it reasonably be 0.82 or 0.88?"

    THE PROCESS:
    1. Fetch raw per-question results from Braintrust (e.g., 100 questions)
    2. Create 1,000 "alternate reality" experiments by resampling with replacement
       - Each resample: randomly pick 100 questions (some duplicated, some missing)
       - This simulates "what if we had tested on a slightly different question set?"
    3. For each of the 1,000 resamples:
       - Calculate metrics (recall@k, mrr@k) for each question
       - Average across the 100 questions
    4. Return 1,000 different average scores (one per bootstrap sample)

    WHAT YOU GET BACK:
    - means_df: 1,000 rows × metrics columns (e.g., recall@1, recall@3, mrr@10)
      Each row = one possible average outcome if you had different questions
    - stds_df: 1,000 rows × metrics columns
      Each row = standard deviation across questions for that bootstrap sample

    This distribution lets us:
    - Calculate confidence intervals (e.g., "recall@10 is between 0.82-0.88 with 95% confidence")
    - Test if differences between experiments are statistically significant
    - Understand how stable our metrics are
    """
    means_list = []
    stds_list = []

    # STEP 1: Get raw per-question results from Braintrust
    # Returns: [{preds: [chunk_ids...], labels: [correct_chunk_id]}, ...] for all questions
    items = config.get_results()

    # STEP 2: Create n_samples (default 1,000) alternate experiments
    for _ in range(n_samples):
        # STEP 3: Resample with replacement (bootstrap magic!)
        # Randomly pick len(items) questions, allowing duplicates
        # Example: If you have 100 questions, you'll get 100 questions back,
        #          but some might appear 2-3 times, others might be missing
        sample = np.random.choice(items, size=len(items), replace=True)

        sample_scores = []

        # STEP 4: Calculate metrics for THIS bootstrap sample
        for row in sample:  # For each of the resampled questions
            row_scores = {}

            # Calculate both recall and MRR at different k values
            for metric_name, metric_fn in metrics:  # [("recall", calculate_recall), ("mrr", calculate_mrr)]
                for subset_k in k:  # [1, 3, 5, 10, 15, 20, ...]
                    metric_key = f"{metric_name}@{subset_k}"

                    # THIS is where the helper functions (calculate_recall, calculate_mrr) get used!
                    row_scores[metric_key] = metric_fn(
                        row["preds"][:subset_k],  # Top k predictions for this question
                        row["labels"]              # Correct answer(s)
                    )
            sample_scores.append(row_scores)

        # STEP 5: Calculate mean and std for THIS bootstrap sample
        # sample_scores is a list of dicts, one per question in this resample
        sample_df = pd.DataFrame(sample_scores)  # Shape: (num_questions, num_metrics)
        means_list.append(sample_df.mean().to_dict())  # Average across all questions
        stds_list.append(sample_df.std().to_dict())    # Std dev across all questions

    # STEP 6: Convert to DataFrames for easy analysis
    # Shape: (n_samples, num_metrics) - e.g., (1000, 20) if you have 2 metrics × 10 k values
    means_df = pd.DataFrame(means_list)  # Each row = one bootstrap sample's average scores
    stds_df = pd.DataFrame(stds_list)     # Each row = one bootstrap sample's std devs

    return means_df, stds_df


# Visualising confidence intervals


@dataclass
class BootstrapResults:
    label: str
    means: Dict[str, float]
    stds: Dict[str, float]
    ci_lower: Dict[str, float]
    ci_upper: Dict[str, float]


def calculate_bootstrap_stats(
    means_df: pd.DataFrame, stds_df: pd.DataFrame, label: str
) -> BootstrapResults:
    """Calculate bootstrap statistics for a given experiment config"""
    # Calculate mean and confidence intervals for each metric
    results = {}
    ci_lower = {}
    ci_upper = {}
    stds = {}

    # For each metric column
    for col in means_df.columns:
        results[col] = means_df[col].mean()
        stds[col] = stds_df[col].std()
        ci_lower[col] = np.percentile(means_df[col], 2.5)
        ci_upper[col] = np.percentile(means_df[col], 97.5)

    return BootstrapResults(
        means=results,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        label=label,
        stds=stds,
    )


bootstrap_dfs = [bootstrap_samples(config, n_samples=1000) for config in configs]
bootstrap_results = [
    calculate_bootstrap_stats(dfs[0], dfs[1], config.label)
    for dfs, config in zip(bootstrap_dfs, configs)
]


def plot_recall_at_k(bootstrap_results: List[BootstrapResults]):
    plt.figure(figsize=(12, 6))

    # Extract k values from the metrics in the first result
    k_values = []
    for key in bootstrap_results[0].means.keys():
        if key.startswith("recall@"):
            k = int(key.split("@")[1])
            k_values.append(k)
    k_values.sort()

    for results in bootstrap_results:
        # Extract recall metrics
        recall_means = [results.means[f"recall@{k}"] for k in k_values]
        recall_lower = [results.ci_lower[f"recall@{k}"] for k in k_values]
        recall_upper = [results.ci_upper[f"recall@{k}"] for k in k_values]

        # Plot line and confidence interval
        plt.plot(k_values, recall_means, marker="o", label=results.label)
        plt.fill_between(k_values, recall_lower, recall_upper, alpha=0.2)

    plt.xlabel("k")
    plt.ylabel("Recall")
    plt.title("Recall@k with 95% Confidence Intervals")
    plt.grid(True)
    plt.legend()


# Plot the results
plot_recall_at_k(bootstrap_results)

# Calculate variances for each bootstrap result
variances_list = []
for i, result in enumerate(bootstrap_results):
    variances = {metric: f"{std**2:.5f}" for metric, std in result.stds.items()}
    variances["model"] = result.label
    variances_list.append(variances)

# Combine into single dataframe
pd.DataFrame(variances_list).set_index("model")

# Are the means statistically different?

bootstrap_means = [item[0] for item in bootstrap_dfs]
# Get the first two bootstrap results
result1, result2 = bootstrap_means

# Perform t-test between the two models for each metric
t_test_results = {}

for metric in result1.keys():
    # Extract values for both models for this metric
    values_model1 = np.array(result1[metric])
    values_model2 = np.array(result2[metric])

    # Check if values are nearly identical
    if np.allclose(values_model1, values_model2, rtol=1e-10):
        print(
            f"Warning: Values for {metric} are nearly identical, t-test results may be unreliable"
        )
        t_test_results[metric] = {"t_statistic": np.nan, "p_value": np.nan}
        continue

    # Perform related t-test
    t_stat, p_value = stats.ttest_rel(values_model1, values_model2)

    t_test_results[metric] = {"t_statistic": t_stat, "p_value": p_value}

# Convert to DataFrame for better visualization
t_test_df = pd.DataFrame(t_test_results).transpose()
print("T-test results between models:")
print(t_test_df)
