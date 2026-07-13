Here's a summary of every PDF→Markdown tool covered in this conversation, what each one actually gives you, and when to reach for which.

## Quick comparison

|Tool|Headers?|Bold/Bullets?|Tables?|Effort|Best for|
|---|---|---|---|---|---|
|`pdftotext`|❌ No|❌ No|❌ No|Trivial|Flat searchable text, quick greps|
|`pdftotext` + line-length heuristic|⚠️ Guessed|❌ No|❌ No|Low|Never — too unreliable, skip this|
|`pdftohtml -xml`|✅ Font-size based|⚠️ Partial|❌ No|Medium|Better structure without extra deps|
|`pdfplumber` (Python)|⚠️ Manual|⚠️ Manual|✅ Yes|Medium|PDFs with real tables/grids|
|`pymupdf` / `fitz` (Python)|✅ Yes|✅ Yes|⚠️ Manual|Higher|Slide decks — best structure recovery|
|`marker` (ML-based)|✅ Yes|✅ Yes|✅ Yes|GPU recommended|Highest quality, scanned/complex PDFs|

---

## 1. `pdftotext` (poppler-utils) — baseline

```bash
pdftotext -nopgbrk -enc UTF-8 lecture.pdf lecture.txt
```

Extracts a flat character stream. No font size, no bold, no bullets — everything looks the same. Fast and dependency-free (just `apt install poppler-utils` on Ubuntu, or `winget install oschwartz10612.Poppler` on Windows). Good for search/grep, bad for structured Markdown.

## 2. `pdftohtml -xml` — cheap structure upgrade

```bash
pdftohtml -xml -i lecture.pdf lecture.xml
```

Same poppler toolkit, but exposes per-text-block **font size** via `<fontspec>` tags. You can map large sizes → `#`/`##` headings with a small script. Still no bullet detection, no bold flags, and word-wrapped paragraphs come out as fragmented `<text>` elements.

## 3. `pdfplumber` — best for tables

```python
import pdfplumber
```

Wraps `pdfminer`, adds precise bounding-box extraction and `extract_tables()`, which recovers grid structures as Python lists you can render as Markdown pipe tables. Use this when slides/documents contain real tabular data (like the CS144 forwarding-table slide).

## 4. `pymupdf` (`fitz`) — the one that actually works well for slides

```python
import fitz
```

This is the tool we ended up building out in full for your CS144 slides. It exposes font **size, bold flags, and bounding boxes per span**, which is what let us fix:

- Real `#`/`##`/`###` headings from font-size ratios
- **Bold** text via the flags bitmask
- Bullet markers split across two PDF lines, merged into clean `- item` lines
- Word-wrapped prose joined back into single paragraphs (both within and across PDF blocks)
- Font-encoding bugs specific to Keynote exports (ligature glyphs mis-mapped to `/` and `$`)
- Page numbers/footers/diagram-icon-label noise filtered out

This is the highest-effort option among the local (non-ML) tools, but it's the only one that reliably reconstructs the semantic structure of the original slide.

## 5. `marker` — ML-based, highest fidelity

```bash
pip install marker-pdf
marker_single lecture.pdf output/ --langs English
```

Runs a layout-detection + OCR pipeline. Handles multi-column layouts, equations, and scanned pages best of any tool discussed — but needs a GPU for reasonable speed and is a heavier dependency.

---

## Bottom line

- **Just need searchable text** → `pdftotext`
- **Need real headings with minimal effort** → `pdftohtml -xml`
- **PDF has actual tables** → `pdfplumber`
- **Slide decks with headings, bold, bullets** (your CS144 case) → `pymupdf` — this is what the script in this conversation uses
- **Scanned PDFs or need best-in-class output, GPU available** → `marker`