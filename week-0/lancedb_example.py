"""
Refer to this documentation for more details: https://lancedb.com/docs/search/hybrid-search/

It should explain the below code better.
"""

import lancedb
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.rerankers import CohereReranker

load_dotenv()

db = lancedb.connect(
    uri="db://alix-partners-48x6ob",
    region="us-east-1",
)

embeddings = (
    get_registry()
    .get("openai")
    .create(
        name="text-embedding-3-small",
        dim=512,  # Can be any value from 128 to 1536 for text-embedding-3-small
    )
)


# Define a Schema
class Words(LanceModel):
    # This is the source field that will be used as input to the OpenAI Embedding API
    text: str = embeddings.SourceField()

    # This is the vector field that will store the output of the OpenAI Embedding API
    vector: Vector(embeddings.ndims()) = embeddings.VectorField()


table = db.create_table("words", schema=Words, mode="overwrite")

# First let's create some more complex documents
documents = [
    {"text": "rebel spaceships striking from a hidden base"},
    {"text": "have won their first victory against the evil Galactic Empire"},
    {"text": "during the battle rebel spies managed to steal secret plans"},
    {"text": "to the Empire's ultimate weapon the Death Star"},
]

# add the data inside.
table.add(documents)

# verify that data was ingested correctly
# Note: For LanceDB Cloud, use search().to_list() to view data
print(table.search().limit(10).to_pandas())

# Need this to activate the full test search
table.create_fts_index("text", replace=True)
index_name = "text_idx"
table.wait_for_index([index_name])

# Doing a simple full text search
for item in table.search("rebel", query_type="fts").to_list():
    print(item["text"])

# Doing a simple hybrid search
results = (
    table.search(
        "flower moon",
        query_type="hybrid",
        vector_column_name="vector",
        fts_columns="text",
    )
    .limit(10)
    .to_pandas()
)

print("Hybrid search results:")
print(results)

# Then Define a Cohere Reranker
# reranker = CohereReranker(model_name="rerank-english-v3.0")

# results = (
#     table.search(
#         "words",
#         query_type="hybrid",
#         vector_column_name="vector",
#         fts_columns="text",
#     )
#     .rerank(reranker)
#     .limit(10)
#     .to_pandas()
# )
