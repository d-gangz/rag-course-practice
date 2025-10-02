"""
Please note that your exact scores might differ slightly, but in general, Text Embedding 3 Large should outperform Text Embedding 3 Small along with the reranker for this use case he is testing

Check out 02_benchmark_retrieval.ipynb. A lot of learnings in terms of how to organise your data for analysis too.

Read this tidy data to understand how to organise your data for analysis.
https://kiwidamien.github.io/what-is-tidy-data.html

In tidy format:
1. Each column represents a single variable
2. Each row represents a single observation

So adding a new row is easier
"""

import asyncio
import uuid
from itertools import product
from typing import Literal, Optional

import datasets
import lancedb
import matplotlib.pyplot as plt
import pandas as pd
from braintrust import Eval, Score, init_dataset
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CohereReranker, Reranker
from lancedb.table import Table


# Function to get or create a LanceDB table based on the embedding model
def get_or_create_lancedb_table(db: Table, table_name: str, embedding_model: str):
    if table_name in db.table_names():
        print(f"Table {table_name} already exists")
        table = db.open_table(table_name)
        table.create_fts_index("query", replace=True)
        return table

    func = get_registry().get("openai").create(name=embedding_model)

    class Chunk(LanceModel):
        id: str
        query: str = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()

    table = db.create_table(table_name, schema=Chunk, mode="overwrite")
    dataset = datasets.load_dataset("567-labs/bird-rag")["train"]
    formatted_dataset = [{"id": item["id"], "query": item["query"]} for item in dataset]
    table.add(formatted_dataset)

    table.create_fts_index("query", replace=True)
    print(f"{table.count_rows()} chunks ingested into the database")
    return table


# Create LanceDB Instance
db = lancedb.connect("./lancedb")

table_small = get_or_create_lancedb_table(
    db, "chunks_text_embedding_3_small", "text-embedding-3-small"
)
table_large = get_or_create_lancedb_table(
    db, "chunks_text_embedding_3_large", "text-embedding-3-large"
)


# ---- Defining metrics ----
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


# ---- Retrieve function - modifyable ----

metrics = [("recall", calculate_recall), ("mrr", calculate_mrr)]
k = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]


def retrieve(
    question: str,
    table: Table,
    max_k=25,
    mode: Literal["vector", "fts", "hybrid"] = "vector",
    reranker: Optional[Reranker] = None,
    hooks=None,
):
    results = table.search(question, query_type=mode).limit(max_k)
    if reranker:
        results = results.rerank(reranker=reranker)
    return [
        {"id": result["id"], "query": result["query"]} for result in results.to_list()
    ]


# Braintrust scoring wrapper function that generates all metric@k combinations.
# For each evaluation, this returns 20 Score objects (2 metrics × 10 k values).
# This allows us to measure how retrieval performance changes at different cutoff points
# (e.g., recall@5 vs recall@20) without re-running the retrieval.
def evaluate_braintrust(input, output, **kwargs):
    # We first get the predictions ( what we retrieved ) and the labels
    predictions = [item["id"] for item in output]
    labels = [kwargs["metadata"]["chunk_id"]]

    scores = []
    # Generate all combinations of metrics (recall, mrr) and k values (1, 3, 5, ..., 40)
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


# ---- Running the Evals ----


async def run_evaluations():
    """Main async function to run all evaluations"""
    # Load subset of evaluation queries
    evaluation_queries = [
        item for item in init_dataset(project="Text-2-SQL", name="Bird-Bench-Questions")
    ]

    # Evaluation configurations
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

    # Run evaluations
    evaluation_results = []
    experiment_id = str(uuid.uuid4())
    # Test all combinations of retrieval configurations:
    # 2 rerankers × 2 search modes × 2 embedding models = 8 total configurations
    # For each configuration, we'll get 20 scores (from evaluate_braintrust)
    #
    # Note: product() works with any iterable:
    # - For dicts (available_rerankers, embedding_model_to_table): iterates over keys
    # - For lists (search_query_modes): iterates over elements
    # This gives us all combinations of (reranker_name, search_mode, embedding_model)
    for reranker_name, search_mode, embedding_model in product(
        available_rerankers, search_query_modes, embedding_model_to_table
    ):
        # Get model instances
        current_reranker = available_rerankers[reranker_name]
        current_table = embedding_model_to_table[embedding_model]

        # Run evaluation (async - needs await)
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

        # Process benchmark results in tidy format
        performance_scores = benchmark_result.summary.scores
        for metric_name, score_data in performance_scores.items():
            metric_type, top_k = metric_name.split("@")
            evaluation_results.append(
                {
                    "metric": metric_type,
                    "k": int(top_k),
                    "reranker": reranker_name,
                    "embedding_model": embedding_model,
                    "query_type": search_mode,
                    "score": score_data.score,
                }
            )

    return evaluation_results


# Run the async function
if __name__ == "__main__":
    evaluation_results = asyncio.run(run_evaluations())
    df = pd.DataFrame(evaluation_results)
    print(df)

    # ----- Plotting the results -----

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot recall for hybrid search
    hybrid_data = df[(df["metric"] == "recall") & (df["query_type"] == "hybrid")]
    for model in sorted(df["embedding_model"].unique()):
        for reranker in sorted(df["reranker"].unique()):
            data = hybrid_data[
                (hybrid_data["reranker"] == reranker)
                & (hybrid_data["embedding_model"] == model)
            ]
            if len(data) > 0:
                marker = "o" if reranker != "none" else "^"
                ax1.plot(
                    data["k"],
                    data["score"],
                    marker=marker,
                    label=f"{model}\n{reranker}",
                )

    ax1.set_title("Recall@k for Hybrid Search")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Recall")
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot recall for vector search
    vector_data = df[(df["metric"] == "recall") & (df["query_type"] == "vector")]
    for model in sorted(df["embedding_model"].unique()):
        for reranker in sorted(df["reranker"].unique()):
            data = vector_data[
                (vector_data["reranker"] == reranker)
                & (vector_data["embedding_model"] == model)
            ]
            if len(data) > 0:
                marker = "o" if reranker != "none" else "^"
                ax2.plot(
                    data["k"],
                    data["score"],
                    marker=marker,
                    label=f"{model}\n{reranker}",
                )

    ax2.set_title("Recall@k for Vector Search")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Recall")
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
