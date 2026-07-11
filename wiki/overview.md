---
title: LLM Wiki Domain Overview
type: overview
tags:
  - llm
  - wiki
  - knowledge-management
date_created: 2026-07-11
---

# LLM Wiki Domain Overview

This domain captures the core idea behind Karpathy's LLM wiki pattern: use an LLM to build and maintain a persistent, interlinked knowledge base that compounds over time instead of re-deriving answers from raw documents on every query.

## Core idea

- Treat the wiki as a durable artifact that grows with each ingested source.
- Keep raw sources immutable and use the wiki for synthesized knowledge.
- Maintain cross-links, summaries, and a navigable index so the knowledge base remains useful as it scales.

## Key workflow

1. Ingest a source into the raw collection.
2. Extract the signal into the wiki as summaries, concepts, entities, and source notes.
3. Update the index and activity log so future queries can navigate the knowledge layer efficiently.

## Pages in this domain

- [LLM Wiki Methodology](concepts/llm-wiki-methodology.md)
- [Karpathy AI LLM Wiki](sources/karpathy-ai-llm-wiki.md)
