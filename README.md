# claude-knowledge

A cross-project persistent knowledge base for Claude Code. Stores solutions, patterns, and environment-specific fixes as semantic-searchable entries with built-in duplicate and conflict detection.

## Features

- **Semantic search** — Finds entries by meaning, not just keywords. Uses `paraphrase-multilingual-MiniLM-L12-v2` for multilingual embeddings with hybrid ranking (vector similarity + keyword match). Search results automatically update reference counts.
- **LLM-assisted conflict judgment** — Vector screening finds candidate matches, then Claude (the LLM) judges whether they are real contradictions — no hardcoded word lists.
- **Auto-capture** — `knowledge_capture` auto-generates name, type, tags, and description from raw text. Vector-screens against existing entries. Use at session end or after fixing bugs to avoid losing experiences.
- **Crystallize** — `knowledge_crystallize` extracts structured knowledge from conversation context into a draft entry ready for `knowledge_add` review.
- **Health check** — `knowledge_health` runs structural integrity checks (index sync, empty entries, cache staleness, stale entries). Zero LLM calls — safe to run every session.
- **Lint** — `knowledge_lint` reports content-quality issues: stale entries, merge candidates, missing tags, short content.
- **Date-grouped storage** — Entries are stored as markdown files grouped by date (`~/.claude/knowledge/entries/YYYY-MM-DD.md`), human-readable and version-control friendly.
- **MCP integration** — Exposes 11 MCP tools for use within Claude Code: `knowledge_search`, `knowledge_add`, `knowledge_confirm`, `knowledge_list`, `knowledge_check`, `knowledge_health`, `knowledge_capture`, `knowledge_crystallize`, `knowledge_lint`, `knowledge_stats`, `knowledge_show`.
- **Dual-path import** — Explicit path (user says "remember this") and implicit path (AI auto-detects → buffers to temp → user confirms).
- **Reference tracking** — Each entry tracks `reference_count` and `last_referenced`. Stale entries are flagged by health/lint.
- **Operation log** — Append-only `log.jsonl` records every add/search/confirm/discard operation for auditability.
- **Embedding cache** — Caches computed embeddings to disk, only recomputes when the index changes.

## How It Works

```
                     knowledge_add(name, content, tags)
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Vector screening    │
                    │  (MiniLM embeddings) │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              No candidates          Candidates found
              (all < 0.70)          (>= 0.70 similarity)
                    │                     │
                    ▼                     ▼
             Write directly      ┌──────────────────┐
                                 │ Return PENDING_ID │
                                 │ + candidate list  │
                                 └────────┬─────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │ Claude (LLM)     │
                                 │ judges each      │
                                 │ candidate:       │
                                 │ real conflict?   │
                                 └────────┬─────────┘
                                          │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                          True conflict          Safe / no contradiction
                               │                     │
                               ▼                     ▼
                          Report to user     knowledge_confirm(
                          for decision        pending_id, action="write")
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Claude Code                                     │
│                                                  │
│  ┌──────────┐  ┌───────────────┐                 │
│  │/remember │  │ auto-detect   │                 │
│  └────┬─────┘  └───────┬───────┘                 │
│       │                │                          │
│       ▼                ▼                          │
│  ┌──────────────────────────────────────────┐    │
│  │  knowledge_mcp_server.py                 │    │
│  │  (FastMCP stdio server)                  │    │
│  │                                          │    │
│  │  knowledge_add ──→ vector screening ──→  │    │
│  │    │                                     │    │
│  │    ├── clean ──→ write directly          │    │
│  │    └── candidates ──→ PENDING_ID          │    │
│  │         │                                 │    │
│  │         ▼ (Claude judges)                 │    │
│  │  knowledge_confirm(pending_id, "write")   │    │
│  │                                          │    │
│  │  knowledge_search ──→ bump ref_count     │    │
│  │  knowledge_health ──→ zero LLM            │    │
│  │  knowledge_capture ──→ auto-metadata     │    │
│  │  knowledge_crystallize ──→ draft         │    │
│  │  knowledge_lint ──→ quality report       │    │
│  └──────────────┬───────────────────────────┘    │
│                 │                                 │
└─────────────────┼─────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────┐
│  knowledge_base.py               │
│  ┌────────────────────────────┐  │
│  │  Search Engine             │  │
│  │   - Semantic (MiniLM)      │  │
│  │   - Keyword (jieba)        │  │
│  │   - Hybrid merge           │  │
│  ├────────────────────────────┤  │
│  │  Candidate Finder          │  │
│  │   - DUP_THRESHOLD 0.85     │  │
│  │   - CONFLICT_THRESHOLD 0.70│  │
│  │   - Labels candidates      │  │
│  │   - LLM does final judge   │  │
│  ├────────────────────────────┤  │
│  │  Health Checker            │  │
│  │   - Index integrity        │  │
│  │   - Empty/stale entries    │  │
│  │   - Cache staleness        │  │
│  ├────────────────────────────┤  │
│  │  Storage                   │  │
│  │   ~/.claude/knowledge/     │  │
│  │   ├── entries/             │  │
│  │   ├── INDEX.json           │  │
│  │   ├── log.jsonl            │  │
│  │   └── .embedding_cache     │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Claude Code (or any MCP-compatible client)

### Step 1: Clone and install dependencies

```bash
git clone https://github.com/xyushi0809/claude-knowledge.git
cd claude-knowledge
pip install sentence-transformers jieba numpy mcp
```

### Step 2: Register as MCP server

Add to `~/.claude/mcp.json` (global applies to all projects):

```json
{
  "mcpServers": {
    "claude-knowledge": {
      "type": "stdio",
      "command": "python",
      "args": ["<absolute-path-to>/claude-knowledge/knowledge_mcp_server.py"]
    }
  }
}
```

Or add to `.mcp.json` in a specific project directory.

### Step 3: Verify

Start a Claude Code session and ask Claude to search the knowledge base:

> "Search claude-knowledge for 'python3'"

Or test directly via CLI:

```bash
python knowledge_base.py search "python3"
```

If the knowledge base is empty, you'll see "Knowledge base is empty." The tools are now ready to use.

## MCP Tools

### `knowledge_search`

Search entries by semantic, keyword, or hybrid query.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | (required) | Search query |
| `top_n` | int | 5 | Number of results |
| `mode` | string | `"hybrid"` | `"hybrid"`, `"semantic"`, or `"keyword"` |

### `knowledge_add`

Stage a new entry. Performs vector screening; if similar entries exist, returns them with a `PENDING_ID` for LLM review. If clean (or `force=true`), writes directly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | (required) | Entry name (hyphenated, e.g. `"windows-python3-stub"`) |
| `content` | string | (required) | Full entry body (markdown) |
| `tags` | string | `""` | Comma-separated tags |
| `entry_type` | string | `"reference"` | `"environment"`, `"bugfix"`, `"pattern"`, `"reference"`, `"tip"` |
| `description` | string | `""` | Short description |
| `source` | string | `"conversation"` | Source label (e.g. `"auto-capture"`, `"conversation"`) |
| `force` | bool | `false` | Skip screening — write directly |

### `knowledge_confirm`

Finalize a pending entry after LLM review.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pending_id` | string | (required) | The `PENDING_ID` returned by `knowledge_add` |
| `action` | string | `"write"` | `"write"` (save) or `"discard"` (drop the entry) |

### `knowledge_list`

List all entries, optionally filtered by tag.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tag` | string | `""` | Optional tag filter |

### `knowledge_check`

Audit all entries for duplicates (similarity >= 0.85) and conflicts (similarity >= 0.70 + contradictory language).

No parameters.

### `knowledge_stats`

Show usage statistics: total entries, type distribution, top tags.

No parameters.

### `knowledge_show`

Display the full content of a specific entry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | (required) | Entry name as shown in `knowledge_list` |

### `knowledge_health`

Run structural integrity checks. Zero LLM calls — safe to run every session.

No parameters.

Checks: index integrity, empty entries, cache staleness, stale entries (>90 days without reference).

### `knowledge_capture`

Auto-capture a potential entry from raw text. Auto-generates name, type, tags, and description. Vector-screens and either writes directly (if clean) or returns `PENDING_ID` with candidates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raw_text` | string | (required) | Raw experience text (error message, fix, pattern, etc.) |
| `source` | string | `"auto-capture"` | Source label |
| `context_hint` | string | `""` | One-line hint to help generate better name/tags |

### `knowledge_crystallize`

Extract structured knowledge from conversation context. Returns a draft entry ready for review and `knowledge_add`. Does NOT save — the LLM reviews and forwards to `knowledge_add`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_text` | string | (required) | Conversation excerpt to crystallize |
| `hint_name` | string | `""` | Suggested entry name |
| `hint_type` | string | `""` | Suggested type (from VALID_TYPES) |

### `knowledge_lint`

Content-quality audit. Reports stale entries, merge candidates (similar pairs), missing tags, and short content.

No parameters.

## Two-Step Entry Flow

When `knowledge_add` is called, one of two paths is taken:

### Path A: Clean entry (no similar entries found)

```
knowledge_add(name, content, tags)
  → vector screening (similarity < 0.70 for all)
  → writes directly
  → returns "Entry saved: ..."
```

### Path B: Candidates found

```
knowledge_add(name, content, tags)
  → vector screening finds candidates (>= 0.70 similarity)
  → returns PENDING_ID + candidate list (with full text)
  → Claude (LLM) judges each candidate:
      - Is this truly the same information? (→ duplicate, ask user)
      - Do these two entries say opposite things? (→ conflict, ask user)
      - Same topic, different advice, both valid? (→ safe, proceed)
  → knowledge_confirm(pending_id, action="write" | "discard")
```

This delegates semantic judgment to the LLM — no hardcoded word lists, no false positives from surface-level pattern matching.

## Health / Lint Workflow

Borrowing from llm-wiki-agent's design:

| | Health (`knowledge_health`) | Lint (`knowledge_lint`) |
|---|---|---|
| Scope | Structural integrity | Content quality |
| LLM calls | Zero | Yes (semantic analysis) |
| Cost | Free | Tokens |
| Frequency | Every session | Every 10-15 adds |
| Checks | Index sync, empty entries, cache staleness, stale refs | Stale entries, merge candidates, missing tags, short content |
| Tool | `knowledge_base.py health` | `knowledge_lint` MCP tool |

Run `knowledge_health` first — linting a corrupt index wastes tokens.

## Dual-Path Import

### Explicit path (user-initiated)

User says "remember this" → Claude calls `knowledge_add`. If clean, written directly. If candidates found, Claude reviews and calls `knowledge_confirm`.

### Implicit path (AI-detected)

1. Claude detects a potential cross-project experience during conversation
2. Writes it to a temp buffer file (`temp_memory.md`) as a pending item
3. At a natural pause, reports to user: "I noticed the following experiences worth saving..."
4. User confirms or rejects each item
5. Confirmed items → `knowledge_add` → `knowledge_confirm` if needed; rejected items → removed from buffer

## Entry Format

Entries are stored as markdown files grouped by date:

```markdown
# 2026-05-08

## windows-python3-stub
> type: bugfix
> tags: windows, python, claude-code
> source: conversation
> description: python3 on Windows opens Microsoft Store instead of running Python

On Windows, `python3` is a Store app stub located at
`%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\python3`.
Calling it opens the Microsoft Store and returns exit code 49.

**Fix:** Always use `python` instead of `python3` in all scripts,
hooks, and configuration files on Windows.

---
```

## Search Ranking Formula

```
Final Score = vector_similarity × 10 + keyword_hits × 5 + time_recency × 1.0 + log10(ref_count) × 0.5
```

- Vector similarity dominates — semantic match is the primary signal
- Keyword hits provide a safety net against semantic drift
- Time recency and reference count act as fine-tuning terms

## Duplicate & Conflict Detection

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `DUP_THRESHOLD` | 0.85 | Cosine similarity >= this → reported as likely duplicate |
| `CONFLICT_THRESHOLD` | 0.70 | Similarity in [0.70, 0.85) → reported as potential conflict candidate |

Vector screening identifies similar entries. The final semantic judgment — "are these two really saying contradictory things?" — is made by Claude (the LLM), not by pattern matching. This means:

- No false positives from surface-level word overlap
- Can recognize nuanced contradictions ("use Redis" vs "use local cache")
- Can distinguish "different advice for different contexts" from true conflicts
- Works in any language the LLM understands

Both MCP and CLI paths use vector screening only — semantic judgment is delegated to the LLM.

## Storage

All data is stored locally under `~/.claude/knowledge/`:

```
~/.claude/knowledge/
├── entries/
│   ├── 2026-05-07.md
│   └── 2026-05-08.md
├── INDEX.json
└── .embedding_cache.pkl
```

- **entries/** — Human-readable markdown, one file per date, one `## section` per entry
- **INDEX.json** — Entry metadata index (names, tags, types, timestamps, reference counts)
- **.embedding_cache.pkl** — Pickled embedding vectors, invalidated when index changes
- **log.jsonl** — Append-only operation log (add, search, confirm, discard)

## CLI Usage

`knowledge_base.py` can also be used directly from the command line:

```bash
# Search
python knowledge_base.py search "MCP shell wrapper fix" --top 5 --mode hybrid

# Add (content from stdin)
echo "Always use python not python3 on Windows" | python knowledge_base.py add windows-python3 --tags windows,python --type bugfix

# List all entries
python knowledge_base.py list
python knowledge_base.py list --tag windows

# Audit for duplicates and conflicts
python knowledge_base.py check

# Structural health check (zero LLM)
python knowledge_base.py health

# Statistics (with reference tracking)
python knowledge_base.py stats

# Show full entry
python knowledge_base.py show windows-python3-stub
```

## Dependencies

- `sentence-transformers` — Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- `jieba` — Chinese text segmentation for keyword search
- `numpy` — Vector operations for similarity computation
- `mcp` — FastMCP server framework for Claude Code integration

## Limitations

- **First-run model download** — The first `knowledge_search` or `knowledge_add` call will download the embedding model (~120 MB for `paraphrase-multilingual-MiniLM-L12-v2`). This is a one-time cost.

## License

MIT
