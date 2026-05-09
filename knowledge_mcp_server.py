#!/usr/bin/env python3
"""
claude-knowledge MCP Server — wraps knowledge_base.py as MCP tools.

Register in .mcp.json or ~/.claude/mcp.json:
    "mcpServers": {
        "claude-knowledge": {
            "command": "python",
            "args": ["<path-to>/knowledge_mcp_server.py"]
        }
    }
"""

import sys
import os

# Force UTF-8 for stdout (MCP stdio transport)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
elif os.environ.get("PYTHONIOENCODING") != "utf-8":
    os.environ["PYTHONIOENCODING"] = "utf-8"

import uuid

from mcp.server.fastmcp import FastMCP

# Reuse knowledge_base internals
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knowledge_base as kb

mcp = FastMCP("claude-knowledge", instructions="Cross-project experience knowledge base for Claude Code.")

# In-memory cache for entries awaiting LLM conflict judgment
_pending = {}  # pending_id -> entry_meta


def _write_entry(name, content, tag_list, entry_type, source, description):
    """Internal: write entry to disk and rebuild index/cache."""
    meta = {
        "name": name,
        "type": entry_type,
        "tags": tag_list,
        "source": source,
        "description": description or content[:80].replace("\n", " "),
    }
    path = kb.append_to_date_file(meta, content)
    entries = kb.read_all_entries()
    kb.rebuild_index(entries)
    kb._log_operation("add", entry_name=name, detail=f"type={entry_type} tags={','.join(tag_list)}")
    if entries:
        texts = [kb._content_for_search(e) for e in entries]
        embeddings = kb.compute_embeddings(texts)
        kb.save_embedding_cache(embeddings, [e["name"] for e in entries], kb.load_index())
    return path


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="knowledge_search",
    description="Search claude-knowledge entries by semantic/keyword/hybrid query."
)
def search(query: str, top_n: int = 5, mode: str = "hybrid") -> str:
    """Search knowledge entries.

    Args:
        query: Search query string
        top_n: Number of results (default 5)
        mode: Search mode — "hybrid" (default), "semantic", or "keyword"
    """
    kb.ensure_dirs()
    entries = kb.read_all_entries()
    if not entries:
        return "Knowledge base is empty."

    index = kb.load_index()
    entry_names = [e["name"] for e in entries]
    embeddings = kb.load_embedding_cache(index)
    if embeddings is None:
        texts = [kb._content_for_search(e) for e in entries]
        embeddings = kb.compute_embeddings(texts)
        kb.save_embedding_cache(embeddings, entry_names, index)

    valid_modes = {"semantic", "keyword", "hybrid"}
    if mode not in valid_modes:
        mode = "hybrid"

    results = []
    if mode == "keyword":
        results = kb.search_keyword(query, entries, top_n)
    else:
        results = kb.search_hybrid(query, entries, embeddings, top_n)

    if not results:
        return "No matching results found."

    kb._bump_reference([r["name"] for r in results])
    kb._log_operation("search", detail=query[:200])

    lines = [f"Found {len(results)} result(s):\n"]
    for r in results:
        score = r.get("score", 0)
        score_str = f"[Relevance: {score:.2f}]" if score else ""
        tags = " ".join(f"#{t}" for t in r.get("tags", []))
        lines.append(f"  **{r['name']}** {score_str}")
        if tags:
            lines.append(f"  Tags: {tags}")
        lines.append(f"  Type: {r.get('type', 'reference')}")
        desc = r.get("description", "") or ""
        lines.append(f"  {desc[:150]}")
        body = r.get("body", "")
        sol = kb._extract_solution(body)
        if sol:
            lines.append(f"  → {sol[:200]}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool(
    name="knowledge_add",
    description="Stage a new entry. Returns candidate conflicts for LLM review. "
                "If clean or force=true, writes directly. Otherwise returns "
                "a pending_id — use knowledge_confirm to finalize."
)
def add_entry(name: str, content: str, tags: str = "", entry_type: str = "reference",
              source: str = "conversation", description: str = "", force: bool = False) -> str:
    """Stage a new knowledge entry.

    If no similar entries are found, writes directly. If candidates are found,
    returns them with a pending_id so the LLM (Claude) can judge whether they
    are real conflicts before calling knowledge_confirm.

    Args:
        name: Entry name (hyphenated, e.g. "my-tip")
        content: Full entry body text
        tags: Comma-separated tags
        entry_type: One of: environment, bugfix, pattern, reference, tip
        source: Source of the knowledge
        description: Short description (optional)
        force: Skip screening — write directly
    """
    kb.ensure_dirs()

    name = kb._sanitize_name(name)
    if not name:
        return "Error: Invalid name."

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    if entry_type not in kb.VALID_TYPES:
        entry_type = "reference"

    # Force mode — write directly, no screening
    if force:
        path = _write_entry(name, content, tag_list, entry_type, source, description)
        return f"Entry saved: {name} ({path})"

    entries = kb.read_all_entries()
    index = kb.load_index()

    new_entry = {
        "name": name,
        "description": description or content[:80].replace("\n", " "),
        "type": entry_type,
        "tags": tag_list,
        "body": content,
    }

    # Compute embeddings and find candidates
    embeddings = None
    if entries:
        embeddings = kb.load_embedding_cache(index)
        if embeddings is None:
            texts = [kb._content_for_search(e) for e in entries]
            embeddings = kb.compute_embeddings(texts)
            kb.save_embedding_cache(embeddings, [e["name"] for e in entries], index)

    if entries and embeddings is not None:
        new_emb = kb.compute_embeddings([kb._content_for_search(new_entry)])[0]
        candidates = kb.find_candidates(new_entry, entries, embeddings, new_emb)

        if candidates:
            # Split into duplicates (>= 0.85) and potential conflicts (0.70–0.85)
            duplicates = [(e, s) for e, s, k in candidates if k == "duplicate"]
            conflicts = [(e, s) for e, s, k in candidates if k == "potential_conflict"]

            pending_id = str(uuid.uuid4())[:8]
            _pending[pending_id] = {
                "name": name,
                "content": content,
                "tags": tags,
                "entry_type": entry_type,
                "source": source,
                "description": description,
            }

            msg = [f"PENDING_ID: {pending_id}"]
            msg.append(f"New entry **{name}** has {len(candidates)} candidate(s). "
                       "Review each — are these true conflicts or safe to write?\n")

            if duplicates:
                msg.append(f"## Duplicates (similarity >= 0.85)\n"
                           "These are very similar. Decide: overwrite / keep both / discard new.\n")
                for e, s in duplicates:
                    msg.append(f"### Existing: **{e['name']}** (similarity: {s:.3f})")
                    msg.append(f"Type: {e.get('type', '')} | Tags: {', '.join(e.get('tags', []))}")
                    msg.append(f"Description: {e.get('description', '')}")
                    msg.append(f"```\n{kb._extract_solution(e.get('body', ''))}\n```\n")

            if conflicts:
                msg.append(f"## Potential Conflicts (similarity {kb.CONFLICT_THRESHOLD:.2f}–{kb.DUP_THRESHOLD - 0.01:.2f})\n"
                           "Topic overlap detected. Judge whether these contradict:\n")
                for e, s in conflicts:
                    msg.append(f"### Existing: **{e['name']}** (similarity: {s:.3f})")
                    msg.append(f"Type: {e.get('type', '')} | Tags: {', '.join(e.get('tags', []))}")
                    msg.append(f"Description: {e.get('description', '')}")
                    msg.append(f"```\n{kb._extract_solution(e.get('body', ''))}\n```\n")

            msg.append(f"### New entry preview")
            msg.append(f"```\n{content[:500]}\n```")
            msg.append(f"\nTo finalize: `knowledge_confirm(pending_id=\"{pending_id}\", action=\"write\")` "
                       "or `action=\"discard\"`")

            return "\n".join(msg)

    # No candidates — write directly
    path = _write_entry(name, content, tag_list, entry_type, source, description)
    return f"Entry saved: {name} ({path})"


@mcp.tool(
    name="knowledge_confirm",
    description="Finalize a pending entry from knowledge_add. Call after LLM review of candidate conflicts."
)
def confirm_entry(pending_id: str, action: str = "write") -> str:
    """Confirm or discard a pending entry.

    Args:
        pending_id: The PENDING_ID returned by knowledge_add
        action: "write" (save to knowledge base) or "discard" (drop the entry)
    """
    entry = _pending.pop(pending_id, None)
    if entry is None:
        return f"Error: No pending entry with id '{pending_id}'. It may have already been confirmed or expired."

    if action == "discard":
        kb._log_operation("discard", entry_name=entry["name"])
        return f"Discarded: {entry['name']}"

    # Write the entry (conflict judgment was already done by LLM)
    tag_list = [t.strip() for t in entry["tags"].split(",")] if entry["tags"] else []
    path = _write_entry(
        entry["name"], entry["content"], tag_list,
        entry["entry_type"], entry["source"], entry["description"]
    )
    return f"Entry saved: {entry['name']} ({path})"


@mcp.tool(
    name="knowledge_list",
    description="List all claude-knowledge entries, optionally filtered by tag."
)
def list_entries(tag: str = "") -> str:
    """List knowledge entries.

    Args:
        tag: Optional tag filter
    """
    index = kb.load_index()
    entries_data = index.get("entries", {})
    if not entries_data:
        return "Knowledge base is empty."

    items = list(entries_data.values())
    if tag:
        tag_lower = tag.lower()
        items = [e for e in items if tag_lower in [t.lower() for t in e.get("tags", [])]]

    items.sort(key=lambda e: e.get("created", ""), reverse=True)

    lines = [f"Total: {len(items)} entries\n"]
    for e in items:
        tags = " ".join(f"#{t}" for t in e.get("tags", []))
        lines.append(f"  {e['name']}")
        lines.append(f"  {e.get('description', '')[:120]}")
        if tags:
            lines.append(f"  Tags: {tags}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool(
    name="knowledge_check",
    description="Audit all claude-knowledge entries for duplicates and contradictions."
)
def check_entries() -> str:
    """Run the duplicate/conflict audit across all entries."""
    kb.ensure_dirs()
    entries = kb.read_all_entries()
    if not entries:
        return "Knowledge base is empty."

    texts = [kb._content_for_search(e) for e in entries]
    embeddings = kb.compute_embeddings(texts)

    import numpy as np
    embs = np.array(embeddings)
    n = len(entries)
    dup_pairs = []
    conflict_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(embs[i] @ embs[j])
            if sim >= kb.DUP_THRESHOLD:
                dup_pairs.append((entries[i], entries[j], sim))
            elif sim >= kb.CONFLICT_THRESHOLD:
                conflict_pairs.append((entries[i], entries[j], sim))

    lines = []
    if dup_pairs:
        lines.append(f"\nFound {len(dup_pairs)} duplicate pair(s):\n")
        for a, b, sim in dup_pairs:
            lines.append(f"  {a['name']} ↔ {b['name']} (similarity: {sim:.3f})")
            lines.append(f"    A: {a.get('description', '')[:80]}")
            lines.append(f"    B: {b.get('description', '')[:80]}")
            lines.append("")

    if conflict_pairs:
        lines.append(f"\nFound {len(conflict_pairs)} candidate pair(s) (similarity {kb.CONFLICT_THRESHOLD:.2f}–{kb.DUP_THRESHOLD - 0.01:.2f}):\n")
        for a, b, sim in conflict_pairs:
            lines.append(f"  {a['name']} ↔ {b['name']} (similarity: {sim:.3f})")
            lines.append(f"    A: {kb._extract_solution(a.get('body', ''))[:200]}")
            lines.append(f"    B: {kb._extract_solution(b.get('body', ''))[:200]}")
            lines.append("")

    if not dup_pairs and not conflict_pairs:
        lines.append(f"✓ Audit complete: {len(entries)} entries, no duplicates or conflicts.")
    else:
        lines.append(f"\nTotal: {len(entries)} entries, {len(dup_pairs)} duplicate(s), {len(conflict_pairs)} candidate(s)")
        lines.append("Note: candidates are vector-screened only. Use knowledge_add for LLM-assisted judgment.")

    return "\n".join(lines)


@mcp.tool(
    name="knowledge_stats",
    description="Show claude-knowledge usage statistics."
)
def stats() -> str:
    """Show knowledge base statistics."""
    index = kb.load_index()
    entries = index.get("entries", {})
    if not entries:
        return "Knowledge base is empty."

    types = {}
    all_tags = {}
    for e in entries.values():
        t = e.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
        for tag in e.get("tags", []):
            all_tags[tag] = all_tags.get(tag, 0) + 1

    created_dates = [e.get("created", "") for e in entries.values() if e.get("created")]
    created_dates.sort()

    lines = ["=== claude-knowledge Statistics ==="]
    lines.append(f"Total entries: {len(entries)}")
    lines.append(f"Earliest: {created_dates[0][:10] if created_dates else 'N/A'}")
    lines.append(f"Latest: {created_dates[-1][:10] if created_dates else 'N/A'}")
    lines.append("\nBy type:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        lines.append(f"  {t}: {c}")
    top_tags = sorted(all_tags.items(), key=lambda x: -x[1])[:10]
    if top_tags:
        lines.append("\nTop tags:")
        for tag, count in top_tags:
            lines.append(f"  #{tag}: {count}")
    return "\n".join(lines)


@mcp.tool(
    name="knowledge_show",
    description="Show the full content of a specific claude-knowledge entry by name."
)
def show_entry(name: str) -> str:
    """Show a complete entry.

    Args:
        name: Entry name (as shown in knowledge_list)
    """
    kb.ensure_dirs()
    entries = kb.read_all_entries()
    for e in entries:
        if e["name"] == name:
            lines = [f"# {e['name']}\n"]
            if e.get("tags"):
                lines.append("Tags: " + " ".join(f"#{t}" for t in e["tags"]))
            lines.append(f"Type: {e.get('type', 'reference')}")
            lines.append(f"Source: {e.get('source', 'N/A')}")
            if e.get("description"):
                lines.append(f"Description: {e['description']}")
            lines.append("")
            lines.append(e.get("body", ""))
            return "\n".join(lines)
    return f"Entry not found: {name}"


@mcp.tool(
    name="knowledge_health",
    description="Run structural health checks on claude-knowledge. Zero LLM calls — safe to run every session."
)
def health() -> str:
    """Run health check: index integrity, empty entries, cache staleness, stale entries."""
    kb.ensure_dirs()
    result = kb.check_health()
    issues = result["issues"]

    lines = ["=== claude-knowledge Health Check ===",
             f"Entries on disk: {result['total_entries']}",
             f"Entries in index: {result['index_entries']}"]

    if not issues:
        lines.append("✓ All checks passed.")
    else:
        lines.append(f"\n{len(issues)} issue(s) found:\n")
        for issue in issues:
            sev = issue["severity"].upper()
            lines.append(f"  [{sev}] {issue['check']}")
            lines.append(f"  {issue['message']}")
            if issue.get("fix"):
                lines.append(f"  → {issue['fix']}")
            lines.append("")
    return "\n".join(lines)


@mcp.tool(
    name="knowledge_capture",
    description="Auto-capture a potential knowledge entry from raw text. "
                "Vector-screens, auto-generates name/tags/description, "
                "returns PENDING_ID if candidates found or writes directly if clean. "
                "Use at session end or after fixing a bug to avoid missing experiences."
)
def capture(raw_text: str, source: str = "auto-capture", context_hint: str = "") -> str:
    """Stage an auto-captured entry from raw text.

    Args:
        raw_text: The raw experience text (error message, fix, pattern, etc.)
        source: Source label (default "auto-capture")
        context_hint: One-line hint to help generate better name/tags
    """
    kb.ensure_dirs()

    if not raw_text or len(raw_text.strip()) < 20:
        return "Error: Text too short to capture (< 20 chars)."

    # Auto-generate metadata
    # Name: use context_hint if given, else first meaningful line
    text = raw_text.strip()
    if context_hint:
        name = kb._sanitize_name(context_hint)
    else:
        first_line = text.split("\n")[0].strip()
        name = kb._sanitize_name(first_line[:60])
    if not name or len(name) < 3:
        name = f"auto-{str(uuid.uuid4())[:8]}"

    # Auto-detect type from keywords
    lower_text = text.lower()
    if any(w in lower_text for w in ("error", "bug", "fix", "failed", "solution", "修复")):
        entry_type = "bugfix"
    elif any(w in lower_text for w in ("setup", "install", "config", "path", "environment", "python3", "配置")):
        entry_type = "environment"
    elif any(w in lower_text for w in ("pattern", "approach", "workflow", "pipeline", "模式")):
        entry_type = "pattern"
    elif any(w in lower_text for w in ("tip", "trick", "note", "技巧")):
        entry_type = "tip"
    else:
        entry_type = "reference"

    # Auto-generate description
    description = text[:120].replace("\n", " ")

    # Auto-extract tags from common categories
    tags = []
    if "windows" in lower_text:
        tags.append("windows")
    if "git" in lower_text or "bash" in lower_text:
        tags.append("git")
    if "python" in lower_text or "pip" in lower_text:
        tags.append("python")
    if "mcp" in lower_text:
        tags.append("mcp")
    if "claude" in lower_text:
        tags.append("claude-code")
    if "stata" in lower_text:
        tags.append("stata")

    name = kb._sanitize_name(name)
    if not name:
        return "Error: Could not generate valid name."

    tag_list = tags if tags else []
    tag_str = ",".join(tag_list)

    # Check for candidates
    entries = kb.read_all_entries()
    index = kb.load_index()

    new_entry = {
        "name": name,
        "description": description,
        "type": entry_type,
        "tags": tag_list,
        "body": text,
    }

    embeddings = None
    if entries:
        embeddings = kb.load_embedding_cache(index)
        if embeddings is None:
            texts_emb = [kb._content_for_search(e) for e in entries]
            embeddings = kb.compute_embeddings(texts_emb)
            kb.save_embedding_cache(embeddings, [e["name"] for e in entries], index)

    if entries and embeddings is not None:
        new_emb = kb.compute_embeddings([kb._content_for_search(new_entry)])[0]
        candidates = kb.find_candidates(new_entry, entries, embeddings, new_emb)

        if candidates:
            duplicates = [(e, s) for e, s, k in candidates if k == "duplicate"]
            conflicts = [(e, s) for e, s, k in candidates if k == "potential_conflict"]

            pending_id = str(uuid.uuid4())[:8]
            _pending[pending_id] = {
                "name": name,
                "content": text,
                "tags": tag_str,
                "entry_type": entry_type,
                "source": source,
                "description": description,
            }

            msg = [f"PENDING_ID: {pending_id}",
                   f"Auto-captured: **{name}** (type={entry_type}, tags={tag_str or 'none'})"]
            if duplicates:
                msg.append(f"\n{len(duplicates)} potential duplicate(s) — similar entries already exist:")
                for e, s in duplicates:
                    msg.append(f"  - **{e['name']}** (similarity: {s:.3f}): {e.get('description', '')[:80]}")
            if conflicts:
                msg.append(f"\n{len(conflicts)} topic overlap(s) — judge if these contradict:")
                for e, s in conflicts:
                    msg.append(f"  - **{e['name']}** (similarity: {s:.3f}): {e.get('description', '')[:80]}")
            msg.append(f"\n### Captured content preview")
            msg.append(f"```\n{text[:400]}\n```")
            msg.append(f"\nTo save: `knowledge_confirm(pending_id=\"{pending_id}\", action=\"write\")`")
            msg.append(f"To edit name/tags first, pass to `knowledge_add` with adjusted params instead.")

            return "\n".join(msg)

    # No candidates — write directly
    path = _write_entry(name, text, tag_list, entry_type, source, description)
    return f"Auto-captured and saved: **{name}** ({path})\nType: {entry_type} | Tags: {tag_str or 'none'}"


@mcp.tool(
    name="knowledge_crystallize",
    description="Extract structured knowledge from conversation context. "
                "Takes a conversation excerpt and returns a draft entry ready "
                "for knowledge_add. Does NOT save — returns the structured draft "
                "for the LLM to review, adjust, and then pass to knowledge_add."
)
def crystallize(conversation_text: str, hint_name: str = "", hint_type: str = "") -> str:
    """Extract a structured knowledge entry from raw conversation text.

    This is a pre-processor: it formats the conversation insight into the
    structure knowledge_add expects. The LLM should review the output,
    adjust as needed, then call knowledge_add with the final version.

    Args:
        conversation_text: The conversation excerpt to crystallize
        hint_name: Suggested entry name (if empty, auto-generated)
        hint_type: Suggested type (if empty, auto-detected)
    """
    if not conversation_text or len(conversation_text.strip()) < 30:
        return "Error: Conversation text too short to crystallize (< 30 chars)."

    text = conversation_text.strip()

    # Auto-detect type
    if hint_type and hint_type in kb.VALID_TYPES:
        entry_type = hint_type
    else:
        lower_text = text.lower()
        if any(w in lower_text for w in ("error", "bug", "fix", "failed", "solution", "修复")):
            entry_type = "bugfix"
        elif any(w in lower_text for w in ("setup", "install", "config", "path", "environment", "python3", "配置")):
            entry_type = "environment"
        elif any(w in lower_text for w in ("pattern", "approach", "workflow", "pipeline", "模式")):
            entry_type = "pattern"
        elif any(w in lower_text for w in ("tip", "trick", "note", "技巧")):
            entry_type = "tip"
        else:
            entry_type = "reference"

    # Extract a description (first meaningful sentence under 150 chars)
    description = text[:150].replace("\n", " ")

    # Generate default name
    name = kb._sanitize_name(hint_name) if hint_name else ""
    if not name or len(name) < 3:
        name = f"crystallized-{str(uuid.uuid4())[:8]}"

    # Auto-suggest tags
    lower_text = text.lower()
    tag_suggestions = []
    if "windows" in lower_text: tag_suggestions.append("windows")
    if "git" in lower_text or "bash" in lower_text: tag_suggestions.append("git")
    if "python" in lower_text or "pip" in lower_text: tag_suggestions.append("python")
    if "mcp" in lower_text: tag_suggestions.append("mcp")
    if "claude" in lower_text: tag_suggestions.append("claude-code")
    if "stata" in lower_text: tag_suggestions.append("stata")

    lines = [
        "=== Crystallized Knowledge Draft ===",
        "",
        "Review this draft, adjust as needed, then pass to `knowledge_add`.",
        "",
        f"  name: {name}",
        f"  entry_type: {entry_type}",
        f"  tags: {','.join(tag_suggestions)}",
        f"  description: {description[:120]}",
        "",
        "### Content:",
        "```",
        text[:2000],
        "```",
        "",
        "To save: `knowledge_add(name=\"...\", content=\"...\", tags=\"...\", entry_type=\"...\")`",
    ]
    return "\n".join(lines)


@mcp.tool(
    name="knowledge_lint",
    description="Content-quality audit of claude-knowledge. Reports stale entries, "
                "merge candidates, missing tags, and short content. Use after "
                "knowledge_health for a full picture."
)
def lint() -> str:
    """Run content-quality linting on all entries.

    Reports:
      - Entries with no tags (harder to discover)
      - Entries with very short content (<50 chars body)
      - Entries with high similarity that may need merging
      - Stale entries not referenced in >90 days
    """
    kb.ensure_dirs()
    entries = kb.read_all_entries()
    if not entries:
        return "Knowledge base is empty."

    index = kb.load_index()
    index_entries = index.get("entries", {})
    issues = []

    # Missing tags
    no_tags = [e["name"] for e in entries if not e.get("tags")]
    if no_tags:
        issues.append(f"**{len(no_tags)} entries with no tags** (harder to search):")
        issues.append("  " + ", ".join(no_tags[:15]))
        issues.append("  → Tip: add tags like #windows, #python, #git for better discoverability\n")

    # Short content
    short = [e["name"] for e in entries if len((e.get("body") or "").strip()) < 50]
    if short:
        issues.append(f"**{len(short)} entries with very short content** (<50 chars):")
        issues.append("  " + ", ".join(short))
        issues.append("  → Review: brief entries may need expanding\n")

    # Merge candidates (high similarity pairs)
    if len(entries) >= 2:
        texts = [kb._content_for_search(e) for e in entries]
        embeddings = kb.compute_embeddings(texts)
        import numpy as np
        embs = np.array(embeddings)
        n = len(entries)
        merge_candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(embs[i] @ embs[j])
                if 0.70 <= sim < kb.DUP_THRESHOLD:  # same range as conflict candidates
                    merge_candidates.append((entries[i]["name"], entries[j]["name"], sim))
        if merge_candidates:
            # Sort by similarity descending
            merge_candidates.sort(key=lambda x: -x[2])
            issues.append(f"**{len(merge_candidates)} merge candidate pairs** (similarity 0.70–0.85):")
            for a, b, sim in merge_candidates[:10]:
                issues.append(f"  - {a} ↔ {b} (similarity: {sim:.3f})")
            issues.append("  → Review: can these be merged or one archived?\n")

    # Stale entries
    from datetime import datetime as dt, timezone as tz, timedelta
    cutoff = (dt.now(tz.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    stale = []
    for e in index_entries.values():
        last_ref = e.get("last_referenced", "") or e.get("created", "")
        ref_count = e.get("reference_count", 0)
        if last_ref and last_ref < cutoff and ref_count < 2:
            stale.append((e["name"], last_ref, ref_count))
    if stale:
        issues.append(f"**{len(stale)} stale entries** (not referenced in >90 days, low engagement):")
        for name, last_ref, ref_count in stale[:10]:
            issues.append(f"  - {name} (last referenced: {last_ref}, refs: {ref_count})")
        issues.append("  → Review: archive, update, or delete\n")

    if not issues:
        return f"✓ Lint complete: {len(entries)} entries, no issues found."

    header = [f"=== claude-knowledge Lint Report ===\n",
              f"Total entries: {len(entries)}\n",
              f"{len(issues)} section(s) found:\n"]
    return "\n".join(header + issues)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
