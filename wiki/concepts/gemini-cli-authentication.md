---
title: Gemini CLI Authentication
type: concept
tags:
  - gemini
  - cli
  - authentication
date_created: 2026-07-11
---

# Gemini CLI Authentication

## Definition

Gemini CLI supports multiple authentication paths, including Google sign-in, Gemini API key, and Vertex AI. The right choice depends on account eligibility, regional access, and whether a Google Cloud project is available.

## How It Works

1. Google sign-in uses Gemini Code Assist and is free for eligible accounts.
2. Gemini API key uses the Gemini Developer API and is commonly used when Google sign-in is unavailable.
3. Vertex AI uses Google Cloud project configuration and is suited to enterprise or project-based workflows.

## Why It Matters

- Authentication mode affects both availability and cost model.
- API key mode is often the most reliable fallback for personal developers.
- Vertex AI is appropriate when project-level Google Cloud infrastructure is already in place.

## See Also

- [Gemini CLI](../sources/gemini-cli.md)
- [Vertex AI for Gemini CLI](vertex-ai-gemini-cli.md)
