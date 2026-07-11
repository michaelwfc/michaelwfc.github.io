---
title: LLM Wiki Methodology
type: concept
tags:
  - llm
  - wiki
  - methodology
date_created: 2026-07-11
---

# LLM Wiki Methodology

## Definition

The LLM wiki methodology uses a language model to maintain a structured knowledge base that sits between raw documents and user questions. The model reads sources, synthesizes the important points, and writes them into linked markdown pages that can be reused over time.

## How It Works

1. A new source is ingested into a raw collection.
2. The model identifies the main ideas, entities, and concepts that matter.
3. It writes or updates wiki pages for the source, relevant concepts, and the overall overview.
4. It keeps the index and log current so future questions can be answered from the accumulated knowledge layer.

## Why it matters

This approach reduces the repeated work of rediscovering facts from scratch. It also helps preserve context, contradictions, and cross-links as a knowledge base grows.

## See Also

- [LLM Wiki Domain Overview](../overview.md)
- [Karpathy AI LLM Wiki](../sources/karpathy-ai-llm-wiki.md)
