#!/usr/bin/env python3
"""
claude-knowledge: Cross-project knowledge management system.

Stores experiences as semantic-searchable entries. Supports conversational
import, duplicate/conflict detection, and user-facing search.

Usage:
    python knowledge_base.py add <name> --tags t1,t2 --type t < content.md
    python knowledge_base.py search <query> [--top 5] [--mode hybrid]
    python knowledge_base.py check [--fix]
    python knowledge_base.py health
    python knowledge_base.py list [--tag t]
    python knowledge_base.py stats
    python knowledge_base.py show <name>
"""

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
elif os.environ.get("PYTHONIOENCODING") != "utf-8":
    os.environ["PYTHONIOENCODING"] = "utf-8"

import jieba

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KNOWLEDGE_DIR = Path.home() / ".claude" / "knowledge"
ENTRIES_DIR = KNOWLEDGE_DIR / "entries"
INDEX_FILE = KNOWLEDGE_DIR / "INDEX.json"
EMBEDDING_CACHE = KNOWLEDGE_DIR / ".embedding_cache.pkl"
LOG_FILE = KNOWLEDGE_DIR / "log.jsonl"

VALID_TYPES = {"environment", "bugfix", "pattern", "reference", "tip"}
DUP_THRESHOLD = 0.85       # similarity above this = duplicate
CONFLICT_THRESHOLD = 0.70  # similarity above this = potential conflict

# ---------------------------------------------------------------------------
# Date-based Entry I/O
# Format: entries/YYYY-MM-DD.md
# Each entry within a file:
#   ## entry-name
#   > type: ...
#   > tags: tag1, tag2
#   > source: ...
#   > description: ...
#
#   body...
# ---------------------------------------------------------------------------

def ensure_dirs():
    ENTRIES_DIR.mkdir(parents=True, exist_ok=True)


def _date_files() -> list[Path]:
    """Return sorted list of YYYY-MM-DD.md files."""
    if not ENTRIES_DIR.exists():
        return []
    return sorted(ENTRIES_DIR.glob("????-??-??.md"))


def _parse_date_file(path: Path) -> list[dict]:
    """Parse a date-grouped entry file into individual entries."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []

    entries = []
    lines = text.split("\n")
    i = 0

    # Skip date header (# YYYY-MM-DD)
    while i < len(lines) and lines[i].startswith("# ") and not lines[i].startswith("## "):
        i += 1

    while i < len(lines):
        line = lines[i]
        m = re.match(r"^## ([a-z][a-z0-9_-]+)", line)
        if m:
            entry = {
                "name": m.group(1),
                "type": "reference",
                "tags": [],
                "source": "",
                "description": "",
                "file_path": str(path),
            }
            i += 1

            # Parse blockquote metadata lines
            while i < len(lines) and lines[i].startswith("> "):
                meta_line = lines[i].strip("> ").strip()
                if ":" in meta_line:
                    key, val = meta_line.split(":", 1)
                    key = key.strip().lower()
                    val = val.strip()
                    if key == "tags":
                        entry["tags"] = [t.strip() for t in val.split(",") if t.strip()]
                    else:
                        entry[key] = val
                i += 1

            # Body until next entry-level ## or EOF
            body_lines = []
            while i < len(lines) and not re.match(r"^## [a-z][a-z0-9_-]+", lines[i]):
                body_lines.append(lines[i])
                i += 1
            entry["body"] = "\n".join(body_lines).strip()
            # Trim trailing separator lines from body
            entry["body"] = re.sub(r"\n---\s*$", "", entry["body"]).strip()
            entry["_full_text"] = text
            entries.append(entry)
        else:
            i += 1

    return entries


def read_all_entries() -> list[dict]:
    """Read all entries across all date files."""
    entries = []
    for fpath in _date_files():
        entries.extend(_parse_date_file(fpath))
    return entries


def append_to_date_file(meta: dict, body: str) -> Path:
    """Append a new entry to today's date file."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = ENTRIES_DIR / f"{today}.md"

    lines = [f"## {meta['name']}"]
    lines.append(f"> type: {meta.get('type', 'reference')}")
    if meta.get("tags"):
        lines.append(f"> tags: {', '.join(meta['tags'])}")
    if meta.get("source"):
        lines.append(f"> source: {meta['source']}")
    if meta.get("description"):
        lines.append(f"> description: {meta['description']}")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    lines.append("---")
    lines.append("")

    text = "\n".join(lines) + "\n"

    if path.exists():
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + text)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {today}\n\n")
            f.write(text)

    return path


def _content_for_search(entry: dict) -> str:
    parts = [entry.get("description") or ""]
    body = entry.get("body", "") or ""
    parts.append(re.sub(r"[#*_~`>|\[\]]", " ", body))
    return " ".join(parts)


def _sanitize_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "-")
    name = re.sub(r"[^a-z0-9_-]", "", name)
    return name[:80]


# ---------------------------------------------------------------------------
# INDEX.json
# ---------------------------------------------------------------------------

def load_index() -> dict:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 1, "entries": {}}


def save_index(index: dict):
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_index(entries: list[dict]) -> dict:
    """Rebuild INDEX.json from parsed entries. Preserves existing reference tracking."""
    old_index = load_index()
    old_entries = old_index.get("entries", {})
    index = {"version": 1, "entries": {}}
    for e in entries:
        name = e["name"]
        old = old_entries.get(name, {})
        index["entries"][name] = {
            "name": name,
            "description": e.get("description", ""),
            "type": e.get("type", "reference"),
            "tags": e.get("tags", []),
            "source": e.get("source", ""),
            "created": old.get("created") or e.get("created") or _date_from_path(e.get("file_path", "")),
            "updated": _date_from_path(e.get("file_path", "")),
            "last_referenced": old.get("last_referenced", ""),
            "reference_count": old.get("reference_count", 0),
            "conflicts": old.get("conflicts", []),
        }
    save_index(index)
    return index


def _date_from_path(path: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    return m.group(1) if m else ""


def _log_operation(op: str, entry_name: str = "", detail: str = ""):
    """Append an operation record to log.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": op,
        "entry": entry_name,
        "detail": detail[:500] if detail else "",
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _bump_reference(names: list[str]):
    """Increment reference_count and update last_referenced for listed entries."""
    if not names:
        return
    index = load_index()
    entries = index.get("entries", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    changed = False
    for name in names:
        if name in entries:
            entries[name]["last_referenced"] = now
            entries[name]["reference_count"] = entries[name].get("reference_count", 0) + 1
            changed = True
    if changed:
        save_index(index)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def compute_embeddings(texts: list[str]) -> list:
    model = get_model()
    texts = [t[:4096] for t in texts]
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def load_embedding_cache(index: dict):
    if not EMBEDDING_CACHE.exists():
        return None
    try:
        with open(EMBEDDING_CACHE, "rb") as f:
            cache = pickle.load(f)
        if cache.get("index_version") == index.get("version") and \
           set(cache.get("entry_names", [])) == set(index.get("entries", {}).keys()):
            return cache.get("embeddings")
    except Exception:
        pass
    return None


def save_embedding_cache(embeddings, entry_names: list[str], index: dict):
    try:
        with open(EMBEDDING_CACHE, "wb") as f:
            pickle.dump({
                "embeddings": embeddings,
                "entry_names": entry_names,
                "index_version": index.get("version", 1),
            }, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_semantic(query: str, entries: list[dict], embeddings, top_k: int = 5) -> list:
    if not entries or embeddings is None:
        return []
    model = get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec
    top = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top:
        if scores[idx] > 0.1:
            results.append({**entries[idx], "score": float(scores[idx])})
    return results


def search_keyword(query: str, entries: list[dict], top_k: int = 5) -> list:
    query_words = set(jieba.lcut_for_search(query.lower()))
    for token in query.lower().split():
        if re.match(r"^[a-z0-9]+", token):
            query_words.add(token)
    scored = []
    for entry in entries:
        text = _content_for_search(entry).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**e, "score": s} for s, e in scored[:top_k]]


def search_hybrid(query: str, entries: list[dict], embeddings, top_k: int = 5) -> list:
    semantic = search_semantic(query, entries, embeddings, top_k)
    keyword = search_keyword(query, entries, top_k * 2)
    seen = set(id(r) for r in semantic)
    for r in keyword:
        if id(r) not in seen:
            semantic.append(r)
            seen.add(id(r))
    return semantic[:top_k]


# ---------------------------------------------------------------------------
# Duplicate & Conflict Detection
# ---------------------------------------------------------------------------

def _extract_solution(body: str) -> str:
    m = re.search(r"##\s*Solution\s*\n(.*?)(?:\n##|\Z)", body, re.DOTALL)
    return m.group(1).strip() if m else body


def detect_duplicates(new_entry: dict, entries: list[dict], embeddings, new_emb) -> list:
    """Alias for find_candidates — kept for CLI backward compatibility."""
    return find_candidates(new_entry, entries, embeddings, new_emb)


def find_candidates(new_entry: dict, entries: list[dict], embeddings, new_emb) -> list:
    """Return all entries >= CONFLICT_THRESHOLD, labeled as 'duplicate' or 'potential_conflict'.

    Contradiction detection is NOT performed here — it's delegated to the LLM
    (Claude Code) via the two-step knowledge_add → knowledge_confirm flow.
    """
    if not entries or embeddings is None:
        return []
    all_embs = list(embeddings)
    all_embs.append(new_emb)
    import numpy as np
    stack = np.array(all_embs)
    scores = stack[:-1] @ stack[-1]
    results = []
    for idx in range(len(entries)):
        sim = float(scores[idx])
        if sim < CONFLICT_THRESHOLD:
            continue
        kind = "duplicate" if sim >= DUP_THRESHOLD else "potential_conflict"
        results.append((entries[idx], sim, kind))
    return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_add(args):
    """Add a new knowledge entry from stdin."""
    content = sys.stdin.read().strip()
    if not content:
        print("Error: No content provided (stdin is empty).", file=sys.stderr)
        sys.exit(1)

    name = _sanitize_name(args.name)
    if not name:
        print("Error: Invalid name.", file=sys.stderr)
        sys.exit(1)

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
    entry_type = args.type if args.type in VALID_TYPES else "reference"

    # Read existing
    entries = read_all_entries()
    index = load_index()

    # Build embeddings for dedup
    embeddings = None
    if entries:
        embeddings = load_embedding_cache(index)
        if embeddings is None:
            texts = [_content_for_search(e) for e in entries]
            embeddings = compute_embeddings(texts)
            save_embedding_cache(embeddings, [e["name"] for e in entries], index)

    new_entry = {
        "name": name,
        "description": args.description or "",
        "type": entry_type,
        "tags": tags,
        "body": content,
    }

    # Duplicate check
    if entries and embeddings is not None:
        new_emb = compute_embeddings([_content_for_search(new_entry)])[0]
        matches = detect_duplicates(new_entry, entries, embeddings, new_emb)

        duplicates = [(e, s) for e, s, k in matches if k == "duplicate"]
        conflicts = [(e, s) for e, s, k in matches if k == "potential_conflict"]

        if duplicates:
            print(f"Found {len(duplicates)} potential duplicate(s):", file=sys.stderr)
            for existing, sim in duplicates:
                print(f"  - {existing.get('name')} (similarity: {sim:.3f})", file=sys.stderr)
            print("\nUse --force to add anyway.", file=sys.stderr)
            sys.exit(0)

        if conflicts:
            print("=" * 60, file=sys.stderr)
            print("POTENTIAL CONFLICTS (review needed)", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            for existing, sim in conflicts:
                print(f"\nEntry: {name}", file=sys.stderr)
                print(f"Candidate: {existing.get('name')} (similarity: {sim:.3f})", file=sys.stderr)
                print(f"  Existing: {_extract_solution(existing.get('body', ''))[:200]}", file=sys.stderr)
                print(f"  New:      {_extract_solution(content)[:200]}", file=sys.stderr)
            print("\nThese are vector-screened candidates. Use --force to add anyway,", file=sys.stderr)
            print("or use the MCP tool for LLM-assisted conflict judgment.", file=sys.stderr)
            sys.exit(2)

    # Save to date file
    meta = {
        "name": name,
        "type": entry_type,
        "tags": tags,
        "source": args.source or "conversation",
        "description": args.description or content[:80].replace("\n", " "),
    }
    path = append_to_date_file(meta, content)
    print(f"Saved: {path}", file=sys.stderr)

    # Rebuild index and cache
    entries = read_all_entries()
    index = rebuild_index(entries)
    if entries:
        texts = [_content_for_search(e) for e in entries]
        embeddings = compute_embeddings(texts)
        save_embedding_cache(embeddings, [e["name"] for e in entries], index)

    print(f"Entry saved: {name}")


def cmd_search(args):
    """Search knowledge entries."""
    entries = read_all_entries()
    if not entries:
        print("Knowledge base is empty.")
        return

    index = load_index()
    entry_names = [e["name"] for e in entries]
    embeddings = load_embedding_cache(index)
    if embeddings is None:
        texts = [_content_for_search(e) for e in entries]
        embeddings = compute_embeddings(texts)
        save_embedding_cache(embeddings, entry_names, index)

    if args.mode in ("semantic", "hybrid") and embeddings is not None:
        results = search_hybrid(args.query, entries, embeddings, args.top)
    else:
        results = search_keyword(args.query, entries, args.top)

    if not results:
        print("No matching results found.")
        return

    _bump_reference([r["name"] for r in results])

    print(f"Found {len(results)} result(s):\n")
    for r in results:
        score = r.get("score", 0)
        score_str = f"[Relevance: {score:.2f}]" if score else ""
        tags = " ".join(f"#{t}" for t in r.get("tags", []))
        print(f"  **{r['name']}** {score_str}")
        if tags:
            print(f"  Tags: {tags}")
        print(f"  Type: {r.get('type', 'reference')}")
        desc = r.get("description", "") or ""
        print(f"  {desc[:150]}")
        body = r.get("body", "")
        sol = _extract_solution(body)
        if sol:
            print(f"  → {sol[:200]}")
        print()


def cmd_check(args):
    """Audit all entries for duplicates and conflicts."""
    entries = read_all_entries()
    if not entries:
        print("Knowledge base is empty.")
        return

    texts = [_content_for_search(e) for e in entries]
    embeddings = compute_embeddings(texts)

    import numpy as np
    embs = np.array(embeddings)
    n = len(entries)
    dup_pairs = []
    conflict_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(embs[i] @ embs[j])
            if sim >= DUP_THRESHOLD:
                dup_pairs.append((entries[i], entries[j], sim))
            elif sim >= CONFLICT_THRESHOLD:
                conflict_pairs.append((entries[i], entries[j], sim))

    if dup_pairs:
        print(f"\nFound {len(dup_pairs)} duplicate pair(s) (similarity >= {DUP_THRESHOLD}):\n")
        for a, b, sim in dup_pairs:
            print(f"  {a['name']} ↔ {b['name']} (similarity: {sim:.3f})")
            print(f"    A: {a.get('description', '')[:80]}")
            print(f"    B: {b.get('description', '')[:80]}")
            print()

    if conflict_pairs:
        print(f"\nFound {len(conflict_pairs)} candidate pair(s) (similarity {CONFLICT_THRESHOLD}–{DUP_THRESHOLD - 0.01:.2f}):\n")
        for a, b, sim in conflict_pairs:
            print(f"  {a['name']} ↔ {b['name']} (similarity: {sim:.3f})")
            print(f"    A: {_extract_solution(a.get('body', ''))[:200]}")
            print(f"    B: {_extract_solution(b.get('body', ''))[:200]}")
            print()

    if not dup_pairs and not conflict_pairs:
        print(f"✓ Audit complete: {len(entries)} entries, no duplicates or conflicts.")
    else:
        print(f"\nTotal: {len(entries)} entries, {len(dup_pairs)} duplicate(s), {len(conflict_pairs)} candidate(s)")
        print("Note: candidates are vector-screened only. Use the MCP tool for LLM-assisted judgment.")


# ---------------------------------------------------------------------------
# Health Check (zero LLM calls — safe to run every session)
# ---------------------------------------------------------------------------

def check_health() -> dict:
    """Run structural health checks. Returns dict with issues found.

    Checks:
      - Index integrity (orphan entries in either direction)
      - Empty entries (body is blank or whitespace)
      - Cache staleness (embedding cache older than index)
      - Stale entries (not referenced in >90 days)
    """
    issues = []
    entries = read_all_entries()
    entry_names = {e["name"] for e in entries}
    index = load_index()
    index_entries = index.get("entries", {})
    index_names = set(index_entries.keys())

    # Index ↔ files integrity
    on_disk_only = entry_names - index_names
    in_index_only = index_names - entry_names
    if on_disk_only:
        issues.append({
            "check": "index_integrity",
            "severity": "warning",
            "message": f"Entries on disk but missing from INDEX.json: {', '.join(sorted(on_disk_only))}",
            "fix": "Run rebuild: knowledge_base.py check --fix",
        })
    if in_index_only:
        issues.append({
            "check": "index_integrity",
            "severity": "warning",
            "message": f"Entries in INDEX.json but missing from disk: {', '.join(sorted(in_index_only))}",
            "fix": "Rebuild index to remove stale references",
        })

    # Empty entries
    empty = [e["name"] for e in entries if not (e.get("body") or "").strip()]
    if empty:
        issues.append({
            "check": "empty_entries",
            "severity": "info",
            "message": f"Entries with no body content: {', '.join(empty)}",
            "fix": "Add content or delete these entries",
        })

    # Stale entries (>90 days without reference)
    from datetime import datetime as dt, timezone as tz, timedelta
    cutoff = (dt.now(tz.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    stale = []
    for e in index_entries.values():
        last_ref = e.get("last_referenced", "") or e.get("created", "")
        if last_ref and last_ref < cutoff:
            # Only flag if older than cutoff AND has no recent implicit activity
            ref_count = e.get("reference_count", 0)
            if ref_count < 2:  # very low engagement
                stale.append(e["name"])
    if stale:
        issues.append({
            "check": "stale_entries",
            "severity": "info",
            "message": f"Entries with no recent references (>90 days, low engagement): {', '.join(stale[:10])}",
            "fix": "Review and consider archiving or updating",
        })

    # Cache staleness
    if EMBEDDING_CACHE.exists() and INDEX_FILE.exists():
        cache_mtime = EMBEDDING_CACHE.stat().st_mtime
        index_mtime = INDEX_FILE.stat().st_mtime
        if index_mtime > cache_mtime + 60:  # index newer than cache by >1 min
            issues.append({
                "check": "cache_staleness",
                "severity": "info",
                "message": "Embedding cache is older than INDEX.json — cache will be rebuilt on next search",
                "fix": "No action needed (auto-heals with next search/add)",
            })

    return {
        "status": "ok" if not any(i["severity"] == "error" for i in issues) else "issues_found",
        "total_entries": len(entries),
        "index_entries": len(index_entries),
        "issues": issues,
    }


def cmd_health(args):
    """Run structural health check."""
    result = check_health()
    issues = result["issues"]

    print(f"=== claude-knowledge Health Check ===")
    print(f"Entries on disk: {result['total_entries']}")
    print(f"Entries in index: {result['index_entries']}")

    if not issues:
        print("✓ All checks passed.")
    else:
        print(f"\n{len(issues)} issue(s) found:\n")
        for issue in issues:
            sev = issue["severity"].upper()
            print(f"  [{sev}] {issue['check']}")
            print(f"  {issue['message']}")
            if issue.get("fix"):
                print(f"  → {issue['fix']}")
            print()


def cmd_list(args):
    """List all entries."""
    index = load_index()
    entries_data = index.get("entries", {})
    if not entries_data:
        print("Knowledge base is empty.")
        return

    items = list(entries_data.values())
    if args.tag:
        tag = args.tag.lower()
        items = [e for e in items if tag in [t.lower() for t in e.get("tags", [])]]

    items.sort(key=lambda e: e.get("created", ""), reverse=True)

    print(f"Total: {len(items)} entries\n")
    for e in items:
        tags = " ".join(f"#{t}" for t in e.get("tags", []))
        print(f"  {e['name']}")
        print(f"  {e.get('description', '')[:120]}")
        if tags:
            print(f"  Tags: {tags}")
        print()


def cmd_stats(args):
    """Show knowledge base statistics."""
    index = load_index()
    entries = index.get("entries", {})
    if not entries:
        print("Knowledge base is empty.")
        return

    types = {}
    all_tags = {}
    ref_counts = []
    for e in entries.values():
        t = e.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
        for tag in e.get("tags", []):
            all_tags[tag] = all_tags.get(tag, 0) + 1
        ref_counts.append(e.get("reference_count", 0))

    created_dates = [e.get("created", "") for e in entries.values() if e.get("created")]
    created_dates.sort()

    print(f"=== claude-knowledge Statistics ===")
    print(f"Total entries: {len(entries)}")
    print(f"Earliest: {created_dates[0][:10] if created_dates else 'N/A'}")
    print(f"Latest: {created_dates[-1][:10] if created_dates else 'N/A'}")
    if ref_counts:
        print(f"Total references: {sum(ref_counts)}")
        print(f"Most referenced entry: {max(ref_counts)} refs")
    print(f"\nBy type:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    top_tags = sorted(all_tags.items(), key=lambda x: -x[1])[:10]
    if top_tags:
        print(f"\nTop tags:")
        for tag, count in top_tags:
            print(f"  #{tag}: {count}")


def cmd_show(args):
    """Show a complete entry."""
    entries = read_all_entries()
    for e in entries:
        if e["name"] == args.name:
            print(f"# {e['name']}\n")
            if e.get("tags"):
                print("Tags:", " ".join(f"#{t}" for t in e["tags"]))
            print(f"Type: {e.get('type', 'reference')}")
            print(f"Source: {e.get('source', 'N/A')}")
            if e.get("description"):
                print(f"Description: {e['description']}")
            print()
            print(e.get("body", ""))
            return
    print(f"Entry not found: {args.name}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(description="claude-knowledge cross-project knowledge base")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a new entry (read content from stdin)")
    p_add.add_argument("name", help="Entry name")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--type", choices=sorted(VALID_TYPES), default="reference", help="Entry type")
    p_add.add_argument("--source", default="conversation", help="Source of knowledge")
    p_add.add_argument("--description", default="", help="Short description")
    p_add.add_argument("--force", action="store_true", help="Skip duplicate/conflict checks")

    p_search = sub.add_parser("search", help="Search knowledge entries")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top", type=int, default=5, help="Number of results")
    p_search.add_argument("--mode", choices=["semantic", "keyword", "hybrid"], default="hybrid")

    p_check = sub.add_parser("check", help="Audit all entries")
    p_check.add_argument("--fix", action="store_true", help="Auto-merge duplicates")

    p_list = sub.add_parser("list", help="List all entries")
    p_list.add_argument("--tag", default="", help="Filter by tag")

    sub.add_parser("stats", help="Show statistics")

    sub.add_parser("health", help="Run structural health check")

    p_show = sub.add_parser("show", help="Show full entry")
    p_show.add_argument("name", help="Entry name")

    args = parser.parse_args()

    commands = {
        "add": cmd_add,
        "search": cmd_search,
        "check": cmd_check,
        "health": cmd_health,
        "list": cmd_list,
        "stats": cmd_stats,
        "show": cmd_show,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
