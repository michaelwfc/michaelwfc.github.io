---
tags:
  - wiki
---

#  🚀 实战落地：在当前仓库中实现 "LLM Wiki" 架构
  你在 Karpathy AI LLM Wiki 知识库.md 中记录了通过 AI Agent 协同维护持久化知识库的设想。你现在正在使用的 Gemini CLI 恰恰是最完美的落地工具！

  你可以直接在当前仓库中构建这套体系：
   - 创建本地配置 (GEMINI.md)：在仓库根目录新建 GEMINI.md，写入专门针对你这个知识库的 Prompt 规则。Gemini CLI 启动时会自动加载该文件，使其化身为你专属的
     "Wiki 维护官"。
   - 建立索引与日志：
     - 在根目录新建 index.md（内容索引，列出所有主题和 MOC）。
     - 新建 log.md（追加式的摄入与查询日志）。
   - 让 Gemini 帮你干重活：当你阅读了新的 AI 论文或微信推文（如 Genai course.md 中记录的链接），可以直接通过 Gemini CLI 运行类似以下的指令：
    > "请帮我阅读这篇关于自监督学习的文章，提炼要点写入 AI-tech/，并自动更新 index.md 和 log.md，同时检查是否与现有的 NLP/知识图谱.md 有关联。"



