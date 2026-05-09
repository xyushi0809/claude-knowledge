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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
