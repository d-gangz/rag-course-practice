"""
This is for doing topic modeling, which is useful for discovering themes or patterns in large collections of text. Think of it like sorting a massive pile of documents into folders based on what they are about, except that the computer figures out both what the folder should be and which documents belong there.

Check out this link to unserstand Kura better: https://0d156a8f.kura-4ma.pages.dev/getting-started/tutorial/
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from kura.checkpoints import JSONLCheckpointManager
from kura.types import Conversation, Message
from kura.visualization import visualise_pipeline_results

load_dotenv()

# examine how the data looks like
# Use Path to get directory where this script is located
script_dir = Path(__file__).parent
conversations_path = script_dir / "conversations.json"

with open(conversations_path) as f:
    conversations_raw = json.load(f)

conversations_raw[0]

# Converts each query-document pair into a Kura conversation object


def process_query_obj(obj: dict):
    return Conversation(
        chat_id=obj["query_id"],
        created_at=datetime.now(),
        messages=[
            Message(
                created_at=datetime.now(),
                role="user",
                content=f"""
User Query: {obj["query"]}
Retrieved Information : {obj["matching_document"]}
""",
            )
        ],
        # enrich data with metadata to set foundation for topic modelling.
        metadata={"query_id": obj["query_id"]},
    )


# print(process_query_obj(conversations_raw[0]))

# Using the procedural API to analyse conversations


async def analyze_conversations(conversations, checkpoint_manager):
    from kura.cluster import (
        ClusterDescriptionModel,
        generate_base_clusters_from_conversation_summaries,
    )
    from kura.dimensionality import HDBUMAP, reduce_dimensionality_from_clusters
    from kura.meta_cluster import MetaClusterModel, reduce_clusters_from_base_clusters
    from kura.summarisation import SummaryModel, summarise_conversations

    # Set up models
    summary_model = SummaryModel()
    cluster_model = ClusterDescriptionModel()
    meta_cluster_model = MetaClusterModel()
    dimensionality_model = HDBUMAP()

    # Run pipeline steps
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=checkpoint_manager
    )

    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries, model=cluster_model, checkpoint_manager=checkpoint_manager
    )

    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters, model=meta_cluster_model, checkpoint_manager=checkpoint_manager
    )

    projected = await reduce_dimensionality_from_clusters(
        reduced_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager,
    )

    return projected


# Convert raw data to Conversation objects
conversations = [process_query_obj(obj) for obj in conversations_raw]

# Use absolute path for checkpoints to avoid creating multiple folders
checkpoint_dir = script_dir.parent / "checkpoints"
checkpoint_manager = JSONLCheckpointManager(str(checkpoint_dir), enabled=True)
checkpoint_manager.save_checkpoint("conversations", conversations)

if __name__ == "__main__":
    # Run async analysis
    clusters = asyncio.run(
        analyze_conversations(conversations, checkpoint_manager=checkpoint_manager)
    )

    visualise_pipeline_results(clusters, style="enhanced")

# Note use uv run uv run kura --dir ./checkpoints to open the web viewer. And then select the respective files accordingly.
