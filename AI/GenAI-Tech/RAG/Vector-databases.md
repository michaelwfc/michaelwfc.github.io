---
title: Vector-databases
source: "https://app.notion.com/p/Vector-database-191198c00c32808d9dd7f76bd297d8ff"
author:
published:
created: 2026-07-18
description: "A collaborative AI workspace, built on your company context. Build and orchestrate agents right alongside your team's projects, meetings, and connected apps."
tags:
  - "clippings"
---


## Vector database

Got it! Here's the revised table where each item in the pros and cons list starts on a new line.

| Database | Pros | Cons |
| --- | --- | --- |
| Faiss | Extremely fast for similarity search, optimized for high-speed nearest neighbor search. Supports GPU acceleration (CUDA support for large-scale searches). Multiple indexing options (IVF, HNSW, PQ, OPQ) for different use cases. Lightweight and easy to integrate with Python and C++. | Not a full database (no persistence, filtering, or real-time updates). Requires manual data management (must reload and re-index for new data). Single-node only (no built-in distributed computing support). |
| Milvus | Full-fledged vector database with persistence and scalability. Hybrid search (supports metadata filtering + vector similarity search). Distributed and cloud-native (supports large-scale AI applications). Supports multiple storage backends (local disk, S3, MinIO, etc.). | More complex setup compared to Faiss (requires database management). Slightly slower than Faiss for pure in-memory searches. Higher resource usage due to distributed nature. |
| Weaviate | Schema-based vector database (structured queries via GraphQL). Multi-modal search (supports images, text, and metadata filtering). Real-time updates and indexing. Supports cloud deployment (Kubernetes, Docker). | Memory-intensive (can consume a lot of RAM). Not as fast as Faiss for large-scale vector-only searches. Smaller community compared to Milvus and Faiss. |
| Qdrant | Vector search with extended filtering (good for faceted search). Optimized for memory efficiency. Can be used as an embedded library or a full database. Real-time insert and delete support. | Fewer indexing options compared to Faiss or Milvus. Newer project with a smaller ecosystem. Not as optimized for pure speed as Faiss. |
| Chroma | AI-native vector database (built for LLM applications like RAG). Simple and easy to set up (Python-native APIs). Great for handling text-based embeddings. Lightweight and easy to use for developers. | Limited scalability compared to distributed databases like Milvus. Not well-optimized for image or audio embeddings. Less mature than other databases in terms of indexing options. |

#### Summary: Best Database for Different Needs

| Use Case | Best Choice |
| --- | --- |
| Fastest pure similarity search (GPU support, in-memory) | Faiss |
| Scalable vector search (big data, distributed applications) | Milvus |
| Hybrid search (vector + metadata filtering, multi-modal search) | Weaviate / Qdrant |
| LLM embedding storage (RAG, AI-native applications) | Chroma |
| Lightweight, embedded search | Faiss / Qdrant |

This should now display each pro and con on a separate line. Let me know if you need further refinements!

### Faiss is "not a full-fledged database"

When I say Faiss is "not a full-fledged database", I mean that Faiss is a vector search library, not a complete database system. Here’s what that means in practical terms:

#### 1\. Faiss Lacks Traditional Database Features

A typical database (like MySQL, PostgreSQL, or Milvus) includes:

Data Persistence: Stores data permanently on disk.

Indexing & Querying: Supports structured queries (e.g., SQL) with filtering.

Real-time Updates: Allows inserting, updating, and deleting data dynamically.

Concurrency Control: Manages multiple users and processes accessing data at the same time.

Faiss lacks these features because it is designed only for vector similarity search, not general-purpose data management.

#### 2\. How Faiss Works Differently from a Database

Faiss loads all data into memory (RAM) for fast search—there’s no built-in disk storage or persistence.

If you insert new data, you must rebuild the entire index, unlike a database that supports real-time inserts.

There’s no metadata filtering—it only searches for the nearest vectors but cannot filter based on extra attributes (e.g., “find vectors close to X where category = ‘sports’”).

No built-in API for querying—you use Faiss through Python or C++ instead of SQL-like commands.

#### 3\. When This Matters

If you only need fast similarity search, Faiss is great.

If you need a database-like system with storage, filtering, and real-time updates, you need something like Milvus, Weaviate, or Qdrant.

Think of Faiss as a "search engine" rather than a database. It’s powerful for finding similar vectors, but it doesn’t handle storage, indexing, or updates like a traditional database.

Would you like recommendations based on your specific use case?