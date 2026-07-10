





[Obsidian入门保姆级教程：20分钟轻松上手Obsidian！](https://www.bilibili.com/video/BV1Xi4y1h76C/?spm_id_from=333.337.search-card.all.click&vd_source=b3d4057adb36b9b243dc8d7a6fc41295)
[Obsidian邪修用法，免费云同步，AI，手机端，还有进阶技巧](https://www.youtube.com/watch?v=IlNOhNeWGgY)

# Obsidian Setting

## + github(同步) + git 插件

* 你的 .gitignore 已经配置得非常棒（正确忽略了容易引起冲突的 workspace*.json 和缓存文件）
* 
## Gemini CLI
- [[Gemini CLI]]
`npm install -g @google/gemini-cli`
如果你**直接使用 Google 账号登录 Gemini CLI（OAuth 登录）**，**不会产生 API 或 Token 费用**。Google 给开发者提供了一套免费的配额，不按 token 计费



## Wiki Link 链接 

| 场景                 | 推荐方式                                        | 原因                              |
| ------------------ | ------------------------------------------- | ------------------------------- |
| 链接到另一篇笔记           | `[[TCP]]`                                   | 支持反向链接、重命名、Graph、别名等完整功能        |
| 链接到标题              | `[[TCP#Flow Control]]`                      | 能直接跳转到指定章节                      |
| 使用显示别名             | `[[Transmission Control Protocol\|TCP]]`    | 文件名保持规范，显示文本更自然                 |
| 链接到网页              | `[RFC 793](https://...)`                    | 外部资源使用标准 Markdown               |
| 插入图片               | `![](image.png)` 或 `![[image.png]]`         | 根据是否希望使用 Obsidian 的嵌入功能选择       |
| 插入 PDF、Markdown 内容 | `![[Lecture1.pdf]]`、`![[TCP#Flow Control]]` | 使用 Obsidian 的嵌入（Transclusion）能力 |

## Settings

* 本地附件托管：你在 Karpathy 的笔记里提到了 “Download images locally” 的技巧，但在目录中我看到 images/ 目录下只有 wiki_control.png 等几张图。
* 建议在 Obsidian 设置 → 文件与链接 中，将“附件默认存放路径”设置为指定的文件夹（例如 images/ 或者是你规划好的 raw/assets/），并使用 Ctrl+Shift+D
 快捷键，确保剪藏网页时图片能自动下载到本地，防止在线图床失效。



## 元数据 & Frontmatter 
引入标准 Frontmatter 与 Dataview 插件
你只有在 Karpathy AI LLM Wiki 知识库.md 中使用了 YAML Frontmatter。建议在全库推广标准化的元数据格式，配合 Dataview 插件 实现自动化汇总。

- **Frontmatter** 提供稳定的“数据库字段”，便于 Dataview 查询和自动生成看板。
- **Wiki Link (`[[...]]`)** 则负责表达知识之间的语义关系，构建真正的知识网络。

这两者结合起来，既能形成知识图谱，又能像数据库一样自动汇总和管理内容，是 Obsidian 最强大的工作流之一。


### 什么是 Frontmatter？
**Frontmatter**（前置元数据）就是放在 Markdown 文件开头的一段 **YAML**。
**Markdown = 文档**
**Frontmatter = 数据库字段（Metadata）**

加入 Frontmatter：Obsidian（以及插件）就知道：它已经变成了一条数据库记录。

YAML Frontmatter **只能出现在文件最开头**。


现在 Obsidian 官方已经把 Frontmatter 和 **Properties** 深度整合，不需要手工编辑 YAML 也可以维护这些字段。

### 什么是 Dataview？

Dataview 可以理解为：

> **SQL + Markdown**

或者：

> **Excel 查询 + Markdown**

它扫描整个 Vault 的 Frontmatter，然后查询。



   * 推荐的 YAML 模板：

   1     ---
   2     tags:
   3       - tech/ai     # 或者是 project, course, tutorial
   4     status: idea    # idea (想法), ongoing (进行中), completed (已完成)
   5     created: 2026-07-10
   6     ---
   
   * Dataview 自动看板：
      在 欢迎.md 或项目管理页面中，插入以下代码段，即可自动拉取所有正在进行的 DIY 项目：

   1     TABLE status, created
   2     FROM #project
   3     WHERE status = "ongoing"

## Obsidian 插件
- Obsidian web clipper: website -> markdown
- 
----


