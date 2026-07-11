---
title: Karpathy AI LLM Wiki
type: source
tags:
  - llm
  - wiki
  - knowledge-management
date_created: 2026-07-11
---

# Karpathy AI LLM Wiki

## Source
[Karpathy-LLM-wiki](raw/blog/Karpathy-LLM-wiki.md)
## Key Takeaways

- The source proposes using an LLM to maintain a persistent, interlinked markdown wiki rather than relying only on retrieval over raw documents.
- The wiki should compound knowledge over time by ingesting new sources, updating summaries, linking related concepts, and recording contradictions.
- The workflow centers on three recurring operations: ingest, query, and lint.

## Notes

- The document is intentionally abstract and focuses on the pattern rather than one specific implementation.
- It emphasizes keeping raw sources immutable while treating the wiki as the maintained synthesis layer.
- It also suggests practical tooling such as Obsidian, graph views, and simple logging conventions.

## Pages Created/Updated

- [LLM Wiki Methodology](../concepts/llm-wiki-methodology.md)
- [LLM Wiki Domain Overview](../overview.md)

## See Also

- [LLM Wiki Methodology](../concepts/llm-wiki-methodology.md)
- [LLM Wiki Domain Overview](../overview.md)
