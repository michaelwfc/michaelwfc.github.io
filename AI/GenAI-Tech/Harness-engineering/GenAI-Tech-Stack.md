---
title: GenAI-Tech-Stack
source: https://app.notion.com/p/Tech-Stack-192198c00c3280bba33dc7f90d97641c
author:
published:
created: 2026-07-18
description: A collaborative AI workspace, built on your company context. Build and orchestrate agents right alongside your team's projects, meetings, and connected apps.
tags:
  - clippings
---
## Tech Stack

### 完整技术栈

| 组件 | 技术方案 |  |
| --- | --- | --- |
| llm model | Llama 2-7B, Mistral-7B, Qwen-7B |  |
| Serving | vLLM, TGI, llama.cpp, hugging face | 优化: Flash Attention, GPTQ, 4-bit 量化 |
| API | FastAPI/Flask |  |
| Orchestrator | langchain, | memory |
| vector database | Faiss, Milvus (向量数据库)/sqlite, snowflake |  |
| sql connector | sqlachemy |  |
| llm cache | Redis |  |
| Web UI | Gradio, Streamlit, Open WebUI |  |
| rag | llamda index, ragflow |  |
| application framework | dify |  |
| monitor | langfuse,Arize AI phoenix/Application Insight |  |
| evaluator | lm-evaluation-harness |  |
| document | sphnix |  |
| unit test | pytest |  |
| lint | flake8 |  |
| package managerment |  |  |