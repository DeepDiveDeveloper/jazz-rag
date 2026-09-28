#!/usr/bin/env python3
"""Fetch and clean Wikipedia articles listed in a manifest (JSONL).

Reads the manifest from scripts/build_manifest.py, downloads each article's
current wikitext via the MediaWiki API (action=query&prop=revisions), strips
markup into readable prose, and appends one JSONL record per article to
--output. Pageids already present in the output file are skipped, so reruns
resume cleanly.
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import mwparserfromhell
import requests
from tqdm import tqdm

from pacer import Pacer, api_get

UA = "jazzbot-ingest/0.1 (contact: local; for research pipeline)"
BATCH = 50
DROPPED_TAGS = {"ref", "references", "gallery", "imagemap", "timeline",
                "chem", "math", "score", "noinclude", "includeonly", "onlyinclude"}
SKIP_LINK_PREFIXES = ("File:", "Image:", "Media:", "Category:", "Template:",
                      "Wikipedia:", "Portal:", "Help:", "Module:", "Draft:",
                      "Special:", "Talk:", "User:", "Book:")


def node_to_text(node):
    if isinstance(node, mwparserfromhell.wikicode.Wikicode):
        return "".join(node_to_text(c) for c in node.nodes)
    if isinstance(node, mwparserfromhell.nodes.Template):
        return ""
    if isinstance(node, mwparserfromhell.nodes.Argument):
        return ""
    if isinstance(node, mwparserfromhell.nodes.Comment):
        return ""
    if isinstance(node, mwparserfromhell.nodes.Tag):
        if node.tag.lower() in DROPPED_TAGS:
            return ""
        if not node.contents:
            return ""
        if isinstance(node.contents, str):
            return node.contents
        return node_to_text(node.contents)
    if isinstance(node, mwparserfromhell.nodes.Wikilink):
        target = str(node.title)
        if target.lower().startswith(SKIP_LINK_PREFIXES):
            return ""
        if node.text:
            return node_to_text(node.text)
        return node_to_text(node.title)
    if isinstance(node, mwparserfromhell.nodes.ExternalLink):
        return node_to_text(node.title) if node.title else ""
    return str(node)


def clean_wikitext(wikitext):
    text = node_to_text(mwparserfromhell.parse(wikitext))
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^;\s*", "", text, flags=re.M)
    text = re.sub(r"^={2,}\s?(.+?)\s?={2,}$", r"## \1", text, flags=re.M)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def fetch_batch(session, titles, pacer=None):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "redirects": "1",
        "titles": "|".join(titles),
        "format": "json",
        "formatversion": "2",
    }
    data = api_get(session, params, pacer=pacer)
    out = {}
    for page in data["query"]["pages"]:
        revs = page.get("revisions") or []
        if not revs:
            continue
        slots = revs[0].get("slots", {})
        content = slots.get("main", {}).get("content")
        if content is not None:
            out[page["title"]] = {"pageid": page["pageid"], "wikitext": content}
    return out


def load_manifest(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_seen(output_path):
    seen = set()
    try:
        with open(output_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    seen.add(json.loads(line)["pageid"])
    except FileNotFoundError:
        pass
    return seen


def write_record(lock, fh, record):
    with lock:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def sync_output(lock, fh):
    with lock:
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="manifest JSONL from build_manifest.py")
    ap.add_argument("-o", "--output", required=True, help="output JSONL path")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="base seconds between API requests (global pacing)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch the first N articles (for smoke tests)")
    args = ap.parse_args()

    records = load_manifest(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]

    out_fh = open(args.output, "a", encoding="utf-8")
    lock = threading.Lock()
    skip = load_seen(args.output)

    pending = [rec for rec in records if rec["pageid"] not in skip]
    batches = [pending[i:i + BATCH] for i in range(0, len(pending), BATCH)]

    topic = f"fetch {len(pending)}/{len(records)} articles ({len(batches)} batch)"
    pbar = tqdm(total=len(batches), desc=topic, unit="batch", file=sys.stdout)

    def on_throttle():
        sync_output(lock, out_fh)

    pacer = Pacer(args.sleep, on_throttle=on_throttle)

    def worker(batch):
        session = requests.Session()
        session.headers["User-Agent"] = UA
        written = 0
        fetched = fetch_batch(session, [r["title"] for r in batch], pacer=pacer)
        for rec in batch:
            hit = fetched.get(rec["title"])
            if hit is None:
                continue
            text = clean_wikitext(hit["wikitext"])
            if not text:
                continue
            write_record(lock, out_fh, {
                "pageid": hit["pageid"],
                "title": rec["title"],
                "url": f"https://en.wikipedia.org/wiki/{quote(rec['title'].replace(' ', '_'))}",
                "categories": rec["categories"],
                "roots": rec["roots"],
                "text": text,
            })
            written += 1
        return written

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, b) for b in batches]
        for fut in as_completed(futures):
            fut.result()
            pbar.update(1)

    pbar.close()
    out_fh.close()


if __name__ == "__main__":
    main()