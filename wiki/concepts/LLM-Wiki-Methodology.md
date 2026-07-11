---
title: LLM Wiki Methodology
type: concept
tags:
  - llm
  - wiki
  - knowledge-management
date_created: 2026-07-11
---

# LLM Wiki Methodology

## Definition

An LLM wiki is a persistent knowledge base maintained by an LLM agent. Instead of re-deriving answers from raw documents on every request, the system builds and updates a structured wiki that grows over time.

## How It Works

1. Keep raw sources immutable and use them as the reference layer.
2. Ingest each new source by extracting the core takeaways and updating relevant pages.
3. Create or revise concept, entity, source, and comparison pages as the knowledge base expands.
4. Maintain a navigable index and an append-only activity log so future queries can reuse the accumulated work.
5. Run periodic linting passes to catch contradictions, orphan pages, and stale claims.

## Why It Matters

- It reduces repeated work by letting the system accumulate and refine knowledge instead of rediscovering it from scratch.
- It makes relationships and contradictions explicit through links and revisions.
- It turns knowledge accumulation into a compounding, maintainable process.

## See Also

- [Karpathy AI LLM Wiki](../sources/karpathy-ai-llm-wiki.md)
- [LLM Wiki Domain Overview](../overview.md)
