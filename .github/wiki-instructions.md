# LLM Wiki

You are the maintainer of this wiki — a persistent, LLM-maintained knowledge base. Read this file before every operation.



---

## Structure

```
copilot-llm-wiki/
├── raw/        ← source documents (immutable — you read, never modify)
└── wiki/       ← everything here is yours to create and maintain
    ├── index.md
    ├── log.md
    ├── overview.md
    ├── entities/
    ├── concepts/
    ├── comparisons/
    ├── sources/
    └── qa/
```

- `raw/` is read-only source material (voice memos, meeting transcripts, API docs).
- `wiki/` is the maintained knowledge layer (the "Company Intranet").
- `wiki/index.md` is the content catalog.
- `wiki/log.md` is the append-only activity log.

## Directory rules

### Raw sources

- `raw/sources/` stores immutable source documents.
- `raw/assets/` stores downloaded local images or attachments referenced by sources.
- Never modify source content in `raw/` unless the user explicitly asks for file hygiene work unrelated to the knowledge layer.

When ingesting a URL, save its markdown content to `raw/websites` before processing. Never edit files in `raw/`.

### Wiki content
- `wiki/` is your persistent knowledge layer. You own it entirely. Pages are organized by category:
- `wiki/overview.md` top-level synthesis: its core aspects, a map to the rest of the wiki
- `wiki/sources/` contains one summary page per ingested raw document
- `wiki/concepts/` foundational ideas (e.g., how it works, core principles,topic, thesis, method, and theme pages.)
- `wiki/entities/` named things (e.g., features, products, people, organizations,  places, works, etc)
- `wiki/qa/` filed answers from multi-page query syntheses which contains reusable outputs produced in response to questions.
- `comparisons` — side-by-side tables
- `wiki/archive/` contains pages that are merged, demoted, or kept only for traceability.
- `wiki/staging/` contains sources or drafts pending human review before ingestion.

---

## General writing rules

### Page conventions
Every substantive wiki page should try to include:

- Use markdown.
- Filenames: lowercase kebab-case, no spaces
- All new wiki pages MUST contain YAML frontmatter at the top (e.g., tags, date_created) matching the files in `wiki/templates/`.
- Prefer concise, high-signal pages over long raw notes.
- Cross-link aggressively using relative markdown links.
- Preserve uncertainty explicitly.
- Distinguish facts, interpretations, open questions, and contradictions.
- Update existing pages when possible instead of creating duplicates.
- When a new page is created, ensure it is linked from at least one other page and listed in `wiki/index.md`.
- Keep the active wiki smaller than the total archive. If a page is weakly sourced, duplicated, merged, or no longer worth surfacing, move it to `wiki/archive/`.
- log.md entries are never edited or deleted

### Quality bar

- Do not dump large excerpts from sources.
- Do not restate the same idea across many pages without adding page-specific value.
- Prefer editing an existing page if the knowledge belongs there.
- Keep links and indexes in sync.
- Leave the wiki in a more connected state after every operation.
- Prefer source-backed pages in the active wiki. Review artifacts and vague uploads can guide cleanup, but they should not dominate the canonical layer.



### Prohibitions

- Never modify or delete files in `raw/` — it is read-only source material
- Never edit or delete past entries in `log.md` — append only
- Never write a wiki page without first reading `index.md` — check before creating
- Never write a page that contradicts an existing page without flagging the contradiction to the user

---


## Operations
### Trigger phrases

| Workflow | Phrases |
|----------|---------|
| Ingest | "ingest X", "add X to the wiki", "process this", "add this source" |
| Query | any question about the wiki |
| Lint | "lint the wiki", "check the wiki", "find orphans" |


### Staging (Human in the Loop)

When the user asks to stage a large document or book:

1. Read the raw source in `wiki/staging/`.
2. Generate an implementation plan detailing the proposed changes to the wiki.
3. Await user approval.
4. Once approved, execute the ingest workflow and move the source to `raw/sources/`.


### Ingest

Triggered by: "ingest X", "add X to the wiki", "process this", "add this source", or simply "ingest" (auto-detects files in `raw/`).

1. **Source Retrieval**: 
   - If a source is provided (URL or path), use it. 
   - If no source is provided, scan the `raw/` directory for any new markdown files (excluding `.gitkeep`).
   - If the source is a URL, save the content to `raw/websites/<slug>.md` first.
2. **State key takeaways** before writing anything: important facts, new entities/concepts to create, existing pages to update.
3. **Contradiction check.** Read any existing pages the source touches. If a claim contradicts an existing page, flag it to the user and do not write until resolved.
4. **Write a source summary** at `wiki/sources/<slug>.md` — key takeaways, notable details, cross-links to pages created or updated.
5. **Create or update relevant (ex:entities/concepts) pages.** A single source typically touches 5–15 pages. New pages start as stubs; fill in what the source supports. Existing pages get new sections or updated facts.
   If the source materially changes the top-level picture (new plan tier, new capability, new product area), update `wiki/overview.md` as well.
6. **Update index.md** add new pages, refresh stale descriptions, keep sections sorted.
7. **Append to log.md.** Append an entry to `wiki/log.md`. 
8. **Cleanup**: You MUST delete the source file from `raw/` after successfully updating the log and index. **Requirement**: Use the `execute` tool with a terminal command (like `rm` or `Remove-Item`) to perform the deletion; do NOT use standard file-editing tools for this step.

Discuss takeaways with the user before writing. Prefer ingesting one source at a time.

Expected result: one source can legitimately touch many wiki pages.
When a relevant page already exists, prefer reconciling and rewriting it coherently rather than merely appending a note.

---

### Query
Triggered by: any question about Copilot

1. Read `wiki/index.md` to find relevant pages.
2. Search ONLY the` wiki/` directory content. Read those pages; follow cross-links as needed.
3. If the wiki can't answer, say which source would fill the gap and ask whether to ingest it.
   "I couldn't find an answer in the wiki. Would you like me to ingest a new source for this?"
4. Answer with citations: list the pages consulted at the end, citing page paths inline.
   Follow the citation format: Pages consulted: [page1.md], [page2.md]
5. If the answer synthesizes across pages in a reusable way, save it under `wiki/qa/`
6. Update `wiki/index.md` and append a `query` entry to `wiki/log.md` if a durable artifact was created.

---

### Lint

Triggered by: "lint the wiki", "check the wiki", "find orphans",When the user asks for a cleanup, health check, or lint pass:

Check for:
- **Orphans pages** Pages in `wiki/` with no entry in `index.md` (orphans pages)
- **Broken cross-links**  Cross-links pointing to pages that don't exist
- **Entities without pages** Concepts or entities mentioned across multiple pages but lacking their own page
- **Stale claims** - facts in older pages that are contradicted by more recently ingested sources (compare dates in log.md)
- **Missing cross-references** Missing cross-references between related pages
- **Contradictions** - claims on one wiki page that directly contradict claims on another page


After reporting all six checks, offer to fix any issues found.

Expected result: the lint pass should leave behind an inspectable report, not just a transient chat answer.
The lint pass should also identify pages that belong in `wiki/archive/` because they are weakly sourced, duplicated, stale, or artifact-level noise.
---

### index.md

The catalog of every wiki page. Updated on every ingest. The LLM reads this first when answering any query.

`wiki/index.md` is the primary navigation surface. Keep it compact and skimmable.

- Organize by section: overview, concepts,  entities, queries, staging.
- Exclude `wiki/archive/` from the main index.
- Exclude legacy flat folders from the main index once migration is complete.
- Each entry should include:
  - page link
  - one-line description
  - optional metadata such as updated date, source count, or confidence


### log.md

Append-only. Never edit past entries.

Format: `## [YYYY-MM-DD] operation | description`

Operations: `init` `ingest` `query` `lint`

- Summary of what changed
- Pages touched: [page](relative/path.md), [page](relative/path.md)

Example:
```
## [2026-04-09] init | Wiki initialized

## [2026-04-10] ingest | What is GitHub Copilot
Saved raw/what-is-github-copilot.md. Created wiki/sources/what-is-github-copilot.md,
wiki/entities/copilot-free.md, wiki/overview.md (updated). Updated index.md.

## [2026-04-10] query | What IDEs support agent mode?
Read entities/copilot-chat.md, comparisons/ide-support.md. Filed answer as wiki/comparisons/agent-mode-ides.md.
```

---

### Page formats

| Type | Required sections |
|------|-------------------|
| entity | ## Overview, ## Key Facts, ## See Also |
| concept | ## Definition, ## How It Works, ## See Also |
| comparison | ## Summary Table, ## See Also |
| source | ## Key Takeaways, ## Pages Created/Updated, ## See Also |
| qa | ## Question, ## Answer, ## Pages Consulted, ## See Also |

