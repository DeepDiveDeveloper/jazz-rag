#!/usr/bin/env python3
"""Build a corpus manifest of Wikipedia articles under jazz root categories.

Walks the subcategory graph via the MediaWiki API (list=categorymembers)
starting from one or more root categories and writes ns=0 articles to a JSONL
manifest that drives scripts/fetch_articles.py.
"""

import argparse
import json
import os
import re
from collections import deque

import requests

from pacer import Pacer, api_get, log

UA = "jazzbot-ingest/0.1 (contact: local; for research pipeline)"


def category_members(session, title, cmtype, pacer=None):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": title,
        "cmtype": cmtype,
        "cmlimit": "max",
        "format": "json",
        "formatversion": "2",
    }
    while True:
        data = api_get(session, params, pacer=pacer)
        yield from data["query"]["categorymembers"]
        if "continue" not in data:
            return
        params.update(data["continue"])


def normalize_root(root):
    return re.sub(r"^(Category:)?", "Category:", root, count=1)


def write_manifest(path, pages):
    """Atomically dump the records collected so far to the manifest file."""
    records = [{
        "pageid": rec["pageid"],
        "title": rec["title"],
        "categories": sorted(rec["categories"]),
        "roots": sorted(rec["roots"]),
    } for rec in sorted(pages.values(), key=lambda r: r["title"].lower())]
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def build(root_categories, depth, max_pages, pacer, output):
    session = requests.Session()
    session.headers["User-Agent"] = UA

    queue = deque((normalize_root(root), 0, normalize_root(root)) for root in root_categories)
    seen_categories = set()
    pages = {}
    checkpointed = -1

    def checkpoint():
        nonlocal checkpointed
        if not pages or len(pages) == checkpointed:
            return
        write_manifest(output, pages)
        checkpointed = len(pages)
        log(f"[checkpoint] wrote {len(pages)} pages while throttled")

    pacer.on_throttle = checkpoint

    while queue:
        cat, level, root = queue.popleft()
        if cat in seen_categories:
            continue
        seen_categories.add(cat)
        if max_pages is not None and len(pages) >= max_pages:
            log(f"[stop] reached max_pages={max_pages}")
            break

        for member in category_members(session, cat, "subcat|page", pacer=pacer):
            if member["ns"] == 14:  # subcategory
                if depth is None or level + 1 <= depth:
                    queue.append((member["title"], level + 1, root))
            elif member["ns"] == 0:  # article
                rec = pages.setdefault(
                    member["pageid"],
                    {
                        "pageid": member["pageid"],
                        "title": member["title"],
                        "categories": set(),
                        "roots": set(),
                    },
                )
                rec["categories"].add(cat)
                rec["roots"].add(root)
        if len(seen_categories) % 20 == 0:
            log(f"[progress] {len(seen_categories)} categories, {len(pages)} pages, queue {len(queue)}")

    log(f"[done] {len(pages)} pages from {len(seen_categories)} categories")
    return pages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", required=True,
                    help="comma-separated root categories, e.g. 'Jazz,Jazz people'")
    ap.add_argument("--depth", type=int, default=5,
                    help="subcategory recursion depth; use -1 for unlimited")
    ap.add_argument("--max-pages", type=int, default=50000,
                    help="stop after collecting this many pages")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds to sleep between API requests")
    ap.add_argument("-o", "--output", required=True, help="output JSONL path")
    args = ap.parse_args()

    depth = None if args.depth < 0 else args.depth
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    if not roots:
        ap.error("--roots requires at least one category")

    pages = build(roots, depth, args.max_pages, Pacer(args.sleep), args.output)

    write_manifest(args.output, pages)
    log(f"[write] {args.output}")


if __name__ == "__main__":
    main()