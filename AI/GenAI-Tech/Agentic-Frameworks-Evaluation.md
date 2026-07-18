---
title: Agentic -Frameworks-Evaluation
source: "https://app.notion.com/p/Agentic-Frameworks-Evaluation-2f7198c00c3280b18d43fe493cd07055"
author:
published:
created: 2026-07-18
description: "A collaborative AI workspace, built on your company context. Build and orchestrate agents right alongside your team's projects, meetings, and connected apps."
tags:
  - "clippings"
---
![🕥](https://notion-emojis.s3-us-west-2.amazonaws.com/prod/svg-twitter/1f565.svg)

## Agentic Frameworks Evaluation

## Prompt:

what are the critical criterias for evaluating an agentic framework for production?

Can you write a comprehensive report about agentic framework/capabilities from Google Vertex AI, AWS, and Azure

## Agentic Framework Evaluation Criteria Table

> Added the new criterion Open-Source Support to measure how each platform interoperates with popular OSS agent frameworks like LangChain, LangGraph, LlamaIndex, etc.

| Criteria | Features to Evaluate | Why It Matters in Production |
| --- | --- | --- |
| LLM Model Support | • Multiple model types & providers • Model routing • Fallback strategies | Avoids vendor lock-in and allows cost/quality tuning |
| Human-in-the-Loop (HITL) | • Step-level pause/resume • Explicit approval workflows • Audit logs | Needed for high-risk actions and compliance |
| Framework Approach | • Code-first/low-code • Native vs agnostic agent runtime • Multi-agent orchestrations | Determines flexibility and learning curve |
| Engine | • Managed session persistence • Support for long-running workflows • Async execution | Keeps agents coherent across days and tasks |
| Orchestration & Workflow Control | • DAGs & conditional logic • Retries & timeouts • Idempotency controls | Essential for reliable production behavior |
| State Management & Memory | • Short-term vs long-term • Vector/KV/graph memory • State serialization & recovery | Persistent context reduces hallucinations |
| Tooling Usage | • Standard protocols (MCP/A2A) support • Tool schema & sandboxing • Permission boundaries | Ensures safe, predictable tool interactions |
| Observability & Monitoring | • Step-level traces • Token usage & latency breakdown • Dashboard/alerts | Debugging traceability and ROI insights |
| Evaluation | • LLM-as-Judge • Golden dataset tests • Regression tests | Prevent regressions and quality drift |
| Deployment & Scalability | • Horizontal scaling • Async task handling • Session isolation | Agents must scale reliably under load |
| Security & Governance | • IAM per agent • Policy enforcement • Data residency controls | Prevents misbehavior and ensures compliance |
| Cost Control | • Token budgets • Rate limits • Step caps • Cost attribution | Prevent runaway bills from loops or heavy models |
| Ecosystem & Extensibility | • SDKs • Tool integration • Model portability • Open-source options | Determines long-term viability and portability |
| Open-Source Support | • First-party OSS integration • Compatibility with LangChain, LangGraph, LlamaIndex, Strands, etc. • Standard protocol support (MCP, A2A) | OSS support accelerates innovation and reduces lock-in |
| Maturity | • Production references • API stability • Community engagement | More mature frameworks tend to cause fewer surprises |

## Google Vertex AI Agentic Framework

| Criteria | Components | Features | Pros | Cons |
| --- | --- | --- | --- | --- |
| LLM Model Support | Gemini + Model Garden | Native support for Gemini models + ability to plug external models | Advanced reasoning and grounding | More GCP-centric ecosystem |
| HITL | Agent Builder approval UI | Response-level review & gating | Easy to add human review | Not step-level pause/resume |
| Framework Approach | ADK (Agent Dev Kit) | Code-first + low-code templates | Flexible & extensible | Less explicit workflow logic |
| Engine | Vertex Agent Engine | Managed long-running sessions | Auto scaling & infra freedom | Less explicit session control |
| Orchestration | ADK logic + templates | Workflow orchestration with templates | Fast prototyping | Weak deterministic DAG support |
| State & Memory | Memory Bank | Topic-based memory, long/short memory | Effective RAG grounding | Less explicit memory control |
| Tooling Usage | A2A, MCP, connectors | Model Context Protocol + tool connectors | Strong grounding options | Mostly Google tools first |
| Observability | Vertex logging + traces | Monitoring & tracing dashboards | Rich observability | Limited replay APIs |
| Evaluation | Agent eval tools | Built-in test harness | Easy answer validation | Not deep workflow regression |
| Deployment & Scalability | adk deploy  , auto scale | Single command production deploy | Fast iteration | Possible GCP lock-in |
| Security & Governance | IAM + API registry | Google Cloud policy & governance | Strong enterprise controls | GCP-centric enforcement |
| Cost Control | Quota/budget alerts | Token & resource tracking | Predictable spend | Less granular step-level |
| Ecosystem & Extensibility | LangChain, LangGraph, ADK | OSS integrations plus templates ([Google Cloud Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview?authuser=19&hl=ja&utm_source=chatgpt.com)) | Supports many OSS frameworks | Less mature integration than AWS |
| Open-Source Support | ADK + LangChain/LangGraph | Full integration with ADK, various templates ([Google Cloud Documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview?authuser=19&hl=ja&utm_source=chatgpt.com)) | Flexible, OSS-friendly | Core Builder still cloud-native only |
| Maturity | Vertex AI (est. 2021) | Stable core, agent features newer | Trusted platform | Agentic features evolving |

### AWS Bedrock AgentCore Framework

| Criteria | Components | Features | Pros | Cons |
| --- | --- | --- | --- | --- |
| LLM Model Support | Bedrock models + external | Supports models from many providers ([Amazon Web Services, Inc.](https://aws.amazon.com/th/bedrock/faqs/?utm_source=chatgpt.com)) | Huge model diversity | Not all frontier latest |
| HITL | Step Functions | Full pause/resume + approval flows | True production HITL | More engineering effort |
| Framework Approach | AgentCore + Bedrock Agents | Framework-agnostic support ([AWS 文档](https://docs.aws.amazon.com/ja_jp/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html?utm_source=chatgpt.com)) | Most flexible | Larger surface area |
| Engine | AgentCore Runtime | Long-running async sessions ([AWS 文档](https://docs.aws.amazon.com/ja_jp/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html?utm_source=chatgpt.com)) | Scales & isolates tasks | Requires infra planning |
| Orchestration | Step Functions + AgentCore | Deterministic workflows | Best control | Higher setup cost |
| State & Memory | AgentCore Memory | Session + long-term memory ([AWS 文档](https://docs.aws.amazon.com/ja_jp/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html?utm_source=chatgpt.com)) | Explicit session state | Requires modeling |
| Tooling Usage | MCP Gateway + Lambda tools | Secure tool wrapping & policies ([Amazon Web Services, Inc.](https://aws.amazon.com/jp/bedrock/agentcore/faqs/?utm_source=chatgpt.com)) | Very safe | Lambda overhead |
| Observability | CloudWatch + OpenTelemetry | Step traces & logs ([Amazon Web Services, Inc.](https://aws.amazon.com/th/bedrock/faqs/?utm_source=chatgpt.com)) | Excellent visibility | Tool config needed |
| Evaluation | Evaluations service | Metrics & testing ([AWS 文档](https://docs.aws.amazon.com/ja_jp/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html?utm_source=chatgpt.com)) | Production-grade testing | No built-in judge agents |
| Deployment & Scalability | Managed serverless | Auto scale + session isolation ([AWS 文档](https://docs.aws.amazon.com/ja_jp/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html?utm_source=chatgpt.com)) | Robust scaling | Requires AWS expertise |
| Security & Governance | IAM + policy | Policy enforcement & identity ([Amazon Web Services, Inc.](https://aws.amazon.com/th/bedrock/faqs/?utm_source=chatgpt.com)) | Strongest governance | Complex policy design |
| Cost Control | Cloud cost tools | Token & usage quotas | Good visibility | Needs monitoring discipline |
| Ecosystem & Extensibility | LangChain, LangGraph, LlamaIndex, CrewAI | Full OSS support ([Amazon Web Services, Inc.](https://aws.amazon.com/jp/bedrock/agentcore/faqs/?utm_source=chatgpt.com)) | Best OSS ecosystem support | Requires adaptation |
| Open-Source Support | Best of all | Integrates LangChain, LangGraph, LlamaIndex, etc. ([Amazon Web Services, Inc.](https://aws.amazon.com/jp/bedrock/agentcore/faqs/?utm_source=chatgpt.com)) | Maximum flexibility | Some frameworks require glue |
| Maturity | AWS infra heritage | Enterprise focus | Very mature | Agent features are newer |

### Azure AI Foundry Agent Framework

| Criteria | Components | Features | Pros | Cons |
| --- | --- | --- | --- | --- |
| LLM Model Support | Azure OpenAI + Foundry | Broad model range ([Itho](https://s.itho.me/ccms_slides/2025/7/3/ba864d4f-4209-411c-bcb3-6be193352aef.pdf?utm_source=chatgpt.com)) | Frontier models + Anthropic | Azure bind |
| HITL | Logic Apps + Prompt Flow | Approval patterns | Easy low-code Enterprise workflows | Fragmented |
| Framework Approach | Semantic Kernel + Foundry | Agent/Skills abstractions | Strong SDKs | Higher complexity |
| Engine | Foundry Agent Service | Managed runtime ([Itho](https://s.itho.me/ccms_slides/2025/7/3/ba864d4f-4209-411c-bcb3-6be193352aef.pdf?utm_source=chatgpt.com)) | Easy enterprise integration | Must learn Azure way |
| Orchestration | Logic Apps | Workflow graphs | Enterprise connectivity | Less expressive |
| State & Memory | Built-in episodic memory | Memory + context ([Itho](https://s.itho.me/ccms_slides/2025/7/3/ba864d4f-4209-411c-bcb3-6be193352aef.pdf?utm_source=chatgpt.com)) | Good context | Implicit behaviors |
| Tooling Usage | Webhook/REST | Tool catalogs + connectors | Broad enterprise tools | Glue required |
| Observability | Prompt Flow | Step traces & diagnostics | Developer-friendly | Limited replay |
| Evaluation | Prompt Flow | Judgment tools | Integrated tests | Smaller ecosystem |
| Deployment & Scalability | Hosted agents | Auto-scale mode ([Itho](https://s.itho.me/ccms_slides/2025/7/3/ba864d4f-4209-411c-bcb3-6be193352aef.pdf?utm_source=chatgpt.com)) | Fast enterprise rollout | Young platform |
| Security & Governance | RBAC + compliance | Mature controls | Enterprise ready | Azure-centric |
| Cost Control | Model Router | Cost-aware balancing | Smart defaults | Attribution maybe opaque |
| Ecosystem & Extensibility | LangChain, LlamaIndex, Semantic Kernel | Strong OSS support | Broad OSS compatibility | Tool gaps |
| Open-Source Support | Medium-High | LangChain, LlamaIndex supported | Flexible | Fragmented approach |
| Maturity | Fresh 2025 GA | Modern designs | Future-ready | Young |

## Agentic Framework Summary Table

This table uses Low / Medium / High to give you a quick comparison.

| Criteria | Google Vertex AI | AWS Bedrock / AgentCore | Azure AI Foundry |
| --- | --- | --- | --- |
| LLM Model Support | Medium | High | High |
| HITL | Medium | High | Medium |
| Framework Approach | Medium | High | Medium |
| Engine (Sessions & Async) | Medium | High | Medium |
| Orchestration & Workflow | Medium | High | Medium |
| State & Memory | Medium | High | Medium |
| Tooling Usage | High | High | Medium |
| Observability & Monitoring | Medium | High | Medium |
| Evaluation | Medium | High | High |
| Deployment & Scalability | High | High | High |
| Security & Governance | High | Very High | Very High |
| Cost Control | Medium | High | Medium |
| Ecosystem & Extensibility | Medium | High | High |
| Open-Source Support | Medium | High | High |
| Maturity | High (platform) | High | Medium (new) |

#### Bottom Line (Engineering Lens)

AWS Bedrock / AgentCore: Best for regulated, mission-critical systems with full OSS support and deterministic workflows.

Google Vertex AI: Best for rich grounding, enterprise RAG, and fast prototyping on GCP.

Azure AI Foundry: Best for deep enterprise integration with Microsoft tools and strong OSS fusion.