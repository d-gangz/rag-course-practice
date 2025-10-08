<!--
Document Type: Technical Documentation
Purpose: Explains how Kura uses checkpoint files to create cluster hierarchies and visualizations
Context: Created during Kura topic modeling implementation to understand the data flow
Key Topics: JSONL checkpoint files, clustering pipeline, hierarchy construction, data relationships
Target Use: Reference guide for understanding Kura's internal data flow and file structure
-->

# How Kura Creates Cluster Hierarchies from Checkpoint Files

## Overview

Kura's web UI displays a hierarchical cluster visualization by processing three key checkpoint files. This document explains how these files work together to create the cluster hierarchy you see in the interface.

## The Three Core Files

### 1. `conversations.jsonl` - Raw Data Source

**Purpose**: Contains the original conversation data that needs to be analyzed.

**Structure**:
```json
{
  "chat_id": "5e878c76-25c1-4bad-8cae-6a40ca4c8138",
  "created_at": "2025-10-08T09:53:00",
  "messages": [
    {
      "role": "user",
      "content": "User Query: experiment tracking\nRetrieved Information: ..."
    }
  ],
  "metadata": {"query_id": "..."}
}
```

**Key Fields**:
- `chat_id`: Unique identifier for each conversation
- `messages`: Array of conversation messages (in our case, query-document pairs)
- `metadata`: Additional context for analysis

**Role in Pipeline**: Source data that gets summarized and clustered.

---

### 2. `summaries.jsonl` - AI-Generated Summaries

**Purpose**: Contains condensed versions of each conversation, created by an LLM analyzing the conversation content.

**Full Structure**:
```json
{
  "summary": "The user is seeking guidance on tracking machine learning experiments using a specific tool, detailing the steps and pseudocode involved in the process.",
  "request": "The user's overall request for the assistant is to provide information on experiment tracking in machine learning.",
  "topic": null,
  "languages": ["english", "python"],
  "task": "The task is to explain how to track machine learning experiments using a specific tool and provide pseudocode examples.",
  "concerning_score": 1,
  "user_frustration": 1,
  "assistant_errors": null,
  "chat_id": "5e878c76-25c1-4bad-8cae-6a40ca4c8138",
  "metadata": {
    "conversation_turns": 1,
    "query_id": "5e878c76-25c1-4bad-8cae-6a40ca4c8138"
  },
  "embedding": null
}
```

**Key Fields**:

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `summary` | `string` | **LLM generated** | Concise summary of the conversation |
| `request` | `string` | **LLM generated** | User's overall intent/request |
| `topic` | `string` or `null` | **LLM generated** | Main topic (if identifiable) |
| `languages` | `array[string]` | **LLM generated** | Languages detected in conversation |
| `task` | `string` | **LLM generated** | Specific task user wants accomplished |
| `concerning_score` | `integer` (1-10) | **LLM generated** | How concerning/problematic the conversation is |
| `user_frustration` | `integer` (1-10) | **LLM generated** | User frustration level detected |
| `assistant_errors` | `string` or `null` | **LLM generated** | Any errors the assistant made |
| `chat_id` | `string` | **Preserved from input** | Links back to original conversation |
| `metadata` | `object` | **Preserved from input** | Your custom metadata from the conversation |
| `embedding` | `array[float]` or `null` | **Added later** | Vector embedding (populated during clustering) |

**How SummaryModel Works**:

1. **Takes as input**: The `Conversation` object from `conversations.jsonl`
   - Reads **ALL messages** in the `messages` array (not just the first one)
   - Analyzes the entire conversation thread for context
   - Preserves `chat_id` and your custom `metadata`

2. **LLM analyzes the entire conversation** and generates:
   - `summary`: What the conversation is about
   - `request`: What the user wants
   - `task`: Specific task to accomplish
   - `languages`: Languages/code detected
   - `concerning_score`: 1-10 scale (1 = normal, 10 = very concerning)
   - `user_frustration`: 1-10 scale (1 = calm, 10 = very frustrated)
   - `assistant_errors`: Any mistakes the assistant made (if applicable)

3. **Outputs**: Complete summary object with both LLM-generated fields and preserved metadata

**Role in Pipeline**:
- The `summary` field is converted to **vector embeddings**
- Similar summaries have similar embeddings
- These embeddings are used for clustering similar conversations together
- Other fields (`languages`, `task`, scores) provide metadata for analysis but aren't used for clustering

---

### 3. `dimensionality.jsonl` - Final Cluster Hierarchy

**Purpose**: Contains the complete cluster structure with hierarchical relationships and 2D coordinates for visualization.

**Structure**:
```json
{
  "id": "fe6c16f3d43b43dd840a25df616bf4c3",
  "name": "Optimize machine learning processes and tools",
  "description": "Users explored techniques for optimizing ML processes...",
  "slug": "ml_process_optimization",
  "chat_ids": ["023fca2d-...", "cbbf4fe7-...", ...],
  "parent_id": null,
  "x_coord": 2.1106297969818115,
  "y_coord": 7.4166059494018555,
  "level": 0,
  "count": 179
}
```

**Key Fields**:
- `id`: Unique cluster identifier
- `name`: Human-readable cluster name (AI-generated)
- `description`: Detailed explanation of what this cluster represents
- `chat_ids`: Array of conversation IDs belonging to this cluster
- `parent_id`: ID of parent cluster (null for top-level clusters)
- `x_coord`, `y_coord`: 2D coordinates for visualization
- `level`: Hierarchy depth (0 = top level, 1 = sub-cluster, etc.)
- `count`: Number of conversations in this cluster

**Role in Pipeline**: The final output used by the UI to render the cluster hierarchy.

---

## The Processing Pipeline

### Step-by-Step Flow

```
conversations.jsonl
        ↓
    [Summarize with LLM]
        ↓
summaries.jsonl
        ↓
    [Generate embeddings]
        ↓
    [Cluster similar embeddings]
        ↓
    clusters.jsonl (initial groupings)
        ↓
    [Meta-clustering: group similar clusters]
        ↓
    meta_clusters.jsonl (hierarchical groupings)
        ↓
    [Dimensionality reduction: project to 2D]
        ↓
dimensionality.jsonl (final hierarchy + coordinates)
```

### Detailed Stages

#### Stage 1: Summarization
```python
# From cluster_convo.py
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=checkpoint_manager
)
```
- Reads each conversation from `conversations.jsonl`
- LLM creates a concise summary
- Saves to `summaries.jsonl`

#### Stage 2: Base Clustering
```python
clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    model=cluster_model,
    checkpoint_manager=checkpoint_manager
)
```
- Converts summaries to embeddings (vectors)
- Groups similar embeddings into initial clusters
- Saves to `clusters.jsonl`

#### Stage 3: Meta-Clustering
```python
reduced_clusters = await reduce_clusters_from_base_clusters(
    clusters,
    model=meta_cluster_model,
    checkpoint_manager=checkpoint_manager
)
```
- Groups similar clusters into parent clusters
- Creates hierarchy (parent-child relationships)
- Saves to `meta_clusters.jsonl`

#### Stage 4: Dimensionality Reduction
```python
projected = await reduce_dimensionality_from_clusters(
    reduced_clusters,
    model=dimensionality_model,
    checkpoint_manager=checkpoint_manager
)
```
- Projects high-dimensional embeddings to 2D space
- Adds `x_coord` and `y_coord` for visualization
- Saves to `dimensionality.jsonl`

---

## How the UI Builds the Hierarchy

### Hierarchy Construction

The UI reads `dimensionality.jsonl` and builds a tree structure using `parent_id`:

**Example Hierarchy**:
```
Optimize machine learning processes and tools (parent_id: null, level: 0)
├── Help me optimize ML training processes (parent_id: fe6c16f3..., level: 1)
├── Enhance GPU utilization for efficient model training (parent_id: fe6c16f3..., level: 1)
└── Support tracking and managing ML projects (parent_id: fe6c16f3..., level: 1)
```

**The Tree Algorithm**:
1. Find all clusters where `parent_id == null` → These are root nodes
2. For each root node, find all clusters where `parent_id == root.id` → These are children
3. Recursively repeat for each child
4. Display with indentation based on `level`

### Linking Back to Source Data

When a user clicks on a cluster in the UI:

1. UI reads `chat_ids` from the cluster in `dimensionality.jsonl`
2. Looks up those `chat_id`s in `conversations.jsonl` to show original content
3. Optionally shows summaries from `summaries.jsonl` for quick overview

**Example**:
```
User clicks: "Optimize ML processes" cluster
  ↓
UI gets: chat_ids = ["5e878c76-...", "cbbf4fe7-...", ...]
  ↓
UI fetches from conversations.jsonl:
  - chat_id: "5e878c76-..."
  - content: "User Query: experiment tracking\n..."
  ↓
UI displays: Original query and document
```

---

## Visualization on 2D Map

The `x_coord` and `y_coord` in `dimensionality.jsonl` are used to position clusters on a 2D scatter plot:

- **Similar clusters** are positioned close together
- **Dissimilar clusters** are far apart
- Users can visually see topic relationships

**Dimensionality Reduction** (UMAP/t-SNE):
- Original embeddings are high-dimensional (e.g., 1536 dimensions)
- Reduced to 2D while preserving similarity relationships
- Allows visual exploration of topic space

---

## File Relationships Summary

| File | Contains | Links To | Used For |
|------|----------|----------|----------|
| `conversations.jsonl` | Original conversations | - | Source data, display content |
| `summaries.jsonl` | AI summaries | `chat_id` → conversations | Clustering input |
| `dimensionality.jsonl` | Cluster hierarchy | `chat_ids` → conversations<br>`parent_id` → other clusters | UI tree structure, visualization |

---

## Key Takeaways

1. **conversations.jsonl** provides the raw data (what users actually said)
2. **summaries.jsonl** provides condensed versions (what the conversation is about)
3. **dimensionality.jsonl** provides the structure (how conversations are grouped and related)

4. The hierarchy is built through **iterative clustering**:
   - Similar conversations → clusters
   - Similar clusters → meta-clusters
   - Forms a tree structure via `parent_id`

5. All files use **JSONL format** (one JSON object per line) for:
   - Easy streaming/reading line-by-line
   - Efficient processing of large datasets
   - Simple analysis with pandas: `pd.read_json('file.jsonl', lines=True)`

6. The UI **needs all three files** because:
   - `dimensionality.jsonl`: Provides structure and coordinates
   - `summaries.jsonl`: Shows what each cluster is about
   - `conversations.jsonl`: Shows actual content when user clicks

---

## How to Structure Your Input Data

### Understanding the Data Flow

To use Kura effectively, you need to understand what happens to your data at each stage:

```
YOUR INPUT DATA
     ↓
Conversation object (what you create)
     ↓
SummaryModel (LLM analyzes your messages)
     ↓
Summary object (LLM-generated fields + your metadata preserved)
     ↓
Clustering (groups similar summaries)
     ↓
Cluster hierarchy (final output)
```

### What You Control (Input Structure)

When creating conversation objects in your code, you have control over:

#### 1. **Conversation Messages** (Required)
This is what the LLM will analyze:

```python
Conversation(
    chat_id="unique-id-123",
    created_at=datetime.now(),
    messages=[
        Message(
            role="user",
            content="Your actual conversation content here"
        )
    ],
    metadata={"your": "custom", "fields": "here"}  # Optional
)
```

**Important**:
- **ALL messages** in the `messages` array are analyzed by the LLM (not just the first one)
- The LLM sees the entire conversation thread: user messages, assistant responses, and any follow-ups
- For single-turn conversations (like our RAG queries), you'll only have one message
- For multi-turn conversations, include all messages for full context
- Make sure the messages contain the **actual text you want to cluster**

#### 2. **Custom Metadata** (Optional but Preserved)

Your custom metadata flows through the entire pipeline unchanged:

```python
# Your input metadata
metadata = {
    "query_id": "abc-123",
    "source": "customer_support",
    "timestamp": "2025-10-08",
    "user_type": "premium"
}

# This metadata will appear in:
# - conversations.jsonl
# - summaries.jsonl
# And you can use it for filtering/analysis later
```

**Key insight**: Your metadata doesn't affect clustering, but it's preserved for your analysis needs.

### What the LLM Generates (Output Structure)

You **cannot control** these fields - they're generated by analyzing your conversation content:

| Generated Field | What It Means | How to Influence It |
|-----------------|---------------|---------------------|
| `summary` | Main summary | Write clear, focused conversation content |
| `request` | User's intent | Make the user's goal explicit in the message |
| `task` | Specific task | Clearly state what needs to be done |
| `languages` | Detected languages | Include code/language you want detected |
| `concerning_score` | 1-10 concern level | Content determines this (1 = normal) |
| `user_frustration` | 1-10 frustration | Tone of the conversation affects this |
| `assistant_errors` | Errors made | Actual errors in the conversation |

### Best Practices for Input Data

#### ✅ DO: Structure messages for clear analysis

```python
# GOOD: Clear, focused content
Message(
    role="user",
    content=f"""
User Query: {query}
Retrieved Information: {document_text}
Context: {additional_context}
"""
)
```

#### ❌ DON'T: Mix unrelated information

```python
# BAD: Confusing, mixed content
Message(
    role="user",
    content="Query: ML training. Also check API docs. User reported bug #123."
)
```

#### ✅ DO: Use metadata for structured data

```python
# GOOD: Structured info in metadata
metadata = {
    "query_id": query["id"],
    "document_id": doc["id"],
    "relevance_score": 0.85,
    "source": "knowledge_base"
}
```

#### ❌ DON'T: Put all info in message content

```python
# BAD: Metadata buried in text
content = "Query ID: 123, Doc ID: 456, Score: 0.85, Source: KB - Actual query here..."
```

### Example: RAG System Input Structure

For a RAG (Retrieval-Augmented Generation) system like ours:

```python
def process_query_obj(obj: dict):
    return Conversation(
        # Use query_id as chat_id for tracking
        chat_id=obj["query_id"],
        created_at=datetime.now(),

        # Structure the message clearly
        messages=[
            Message(
                created_at=datetime.now(),
                role="user",
                content=f"""
User Query: {obj["query"]}
Retrieved Information: {obj["matching_document"]}
"""
            )
        ],

        # Store identifiers and metrics in metadata
        metadata={
            "query_id": obj["query_id"],
            "document_id": obj.get("document_id"),
            "retrieval_score": obj.get("score"),
            "source": "rag_system"
        }
    )
```

**Why this structure works**:
1. **Clear content**: LLM can easily understand the query and document
2. **Preserved IDs**: `query_id` in metadata lets you trace back to source data
3. **Additional metrics**: Retrieval score available for later analysis
4. **Consistent format**: Every conversation follows the same pattern

### Multi-Turn Conversation Example

If you have actual multi-turn conversations (not just single query-document pairs), include all messages:

```python
# Example: Customer support conversation with multiple turns
Conversation(
    chat_id="support-ticket-789",
    created_at=datetime.now(),
    messages=[
        Message(role="user", content="How do I reset my password?"),
        Message(role="assistant", content="You can reset it by clicking 'Forgot Password' on the login page."),
        Message(role="user", content="I tried that but didn't receive an email"),
        Message(role="assistant", content="Let me check your email settings. Can you confirm your email address?"),
        Message(role="user", content="Yes, it's user@example.com"),
        Message(role="assistant", content="I've resent the reset email. Please check your spam folder too.")
    ],
    metadata={"ticket_id": "789", "category": "password_reset"}
)
```

**The LLM will analyze ALL 6 messages** to understand:
- The user's issue (password reset)
- The progression of the conversation
- Whether the issue was resolved
- The user's frustration level across multiple turns

This results in a more accurate summary than just analyzing the first message alone.

### Using Metadata for Analysis

After clustering, you can analyze patterns using your preserved metadata:

```python
import pandas as pd
import json

# Load summaries with your custom metadata
with open('checkpoints/summaries.jsonl') as f:
    summaries = [json.loads(line) for line in f]

# Extract metadata for analysis
df = pd.DataFrame([
    {
        'chat_id': s['chat_id'],
        'summary': s['summary'],
        'query_id': s['metadata']['query_id'],
        'source': s['metadata'].get('source'),
        'frustration': s['user_frustration']
    }
    for s in summaries
])

# Analyze: Which sources have highest frustration?
df.groupby('source')['frustration'].mean()
```

### Key Takeaway

**What you control**:
- Message content (what gets analyzed)
- Custom metadata (preserved throughout)
- Conversation structure

**What the LLM controls**:
- Summary quality
- Detected languages, tasks, intent
- Concern and frustration scores

**Best approach**: Focus on creating **clear, consistent message content** and use **metadata for your tracking needs**. The LLM will handle the analysis and generate appropriate summaries for clustering.
