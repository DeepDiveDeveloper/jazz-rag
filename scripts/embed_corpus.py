#!/usr/bin/env python3
"""Chunk and embed the fetched corpus into a Chroma vector store.

Reads the cleaned articles from scripts/fetch_articles.py (JSONL), splits
each into section-aware chunks, embeds them with a local
sentence-transformers model on CPU, and stores them in a persistent Chroma
collection. Re-runs skip articles whose first chunk is already indexed, so
it is resumable and can be pointed at a growing corpus.jsonl.
"""

import argparse
import json
import re
import sys
import time

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "jazz-corpus-v1"
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
DEFAULT_CHUNK_CHARS = 1500


def log(message):
    sys.stderr.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def split_sections(text):
    """Split cleaned prose into (heading, body_lines) sections."""
    sections = []
    heading = None
    body = []
    for line in text.split("\n"):
        m = HEADING_RE.match(line)
        if m:
            if heading is not None or body:
                sections.append((heading, body))
            heading = m.group(1)
            body = []
        else:
            body.append(line)
    if heading is not None or body:
        sections.append((heading, body))
    return sections


def section_text(heading, body):
    s = "\n".join(body).strip()
    if not s:
        return None
    if heading:
        return f"## {heading}\n{s}"
    return s


def char_split(text, target):
    return [text[i:i + target] for i in range(0, len(text), target)]


def word_split(text, target):
    words = text.split()
    pieces = []
    cur = []
    cur_len = 0
    for w in words:
        if cur and cur_len + len(w) + 1 > target:
            pieces.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_long(section, target):
    """Split a single section that exceeds the chunk budget."""
    prefix = ""
    if section.startswith("## "):
        nl = section.find("\n")
        prefix, section = section[: nl + 1], section[nl + 1:]
    pieces = []
    cur = prefix
    for para in section.split("\n\n"):
        if cur != prefix and len(cur) + len(para) + 2 > target:
            pieces.append(cur.strip())
            cur = prefix
        cur += ("" if cur.endswith("\n") else "\n\n") + para
    if cur.strip():
        pieces.append(cur.strip())
    if len(pieces) == 1 and len(pieces[0]) > target * 1.5:
        return [prefix + p.strip() for p in word_split(section, target)]
    return [p for p in pieces if p.strip()]


def chunk_article(text, target=DEFAULT_CHUNK_CHARS):
    """Return a deterministic list of chunk strings, each self-contained."""
    chunks = []
    buf = []
    buf_len = 0
    for heading, body in split_sections(text):
        section = section_text(heading, body)
        if section is None:
            continue
        if len(section) > target:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_len = [], 0
            chunks.extend(split_long(section, target))
        else:
            if buf and buf_len + len(section) > target:
                chunks.append("\n\n".join(buf))
                buf, buf_len = [], 0
            buf.append(section)
            buf_len += len(section)
    if buf:
        chunks.append("\n\n".join(buf))
    return [c for c in chunks if c.strip()]


def load_records(path, limit):
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if limit is not None:
        records = records[:limit]
    return records


def done_article(collection, pageid):
    hits = collection.get(ids=[f"{pageid}:0"], include=[])
    return len(hits["ids"]) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus.jsonl",
                    help="cleaned articles JSONL from fetch_articles.py")
    ap.add_argument("-o", "--db", default="data/chroma",
                    help="Chroma persistent directory (default: data/chroma)")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="sentence-transformers model id")
    ap.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS,
                    help="target chunk size in characters")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="chunks per embed/add round trip")
    ap.add_argument("--limit", type=int, default=None,
                    help="only index the first N articles (smoke tests)")
    args = ap.parse_args()

    records = load_records(args.corpus, args.limit)
    if not records:
        sys.exit("no records in corpus")

    client = chromadb.PersistentClient(path=args.db)
    collection = client.get_or_create_collection(args.collection)

    pending_records = []
    for rec in records:
        if done_article(collection, rec["pageid"]):
            continue
        pending_records.append(rec)
    log(f"{len(pending_records)}/{len(records)} articles to index "
        f"({len(records) - len(pending_records)} already in {args.collection})")

    log(f"loading model {args.model}…")
    model = SentenceTransformer(args.model)

    pending = []
    indexed = 0
    pbar = tqdm(total=len(pending_records), unit="article", file=sys.stdout)
    for rec in pending_records:
        chunks = chunk_article(rec["text"], args.chunk_chars)
        base_meta = {
            "pageid": rec["pageid"],
            "title": rec["title"],
            "url": rec["url"],
            "categories": rec["categories"],
            "roots": rec["roots"],
        }
        for seq, chunk in enumerate(chunks):
            meta = dict(base_meta)
            meta["seq"] = seq
            meta["chunks"] = len(chunks)
            pending.append((f"{rec['pageid']}:{seq}", chunk, meta))
        if len(pending) >= args.batch_size:
            indexed += flush(collection, model, pending, args.batch_size)
        pbar.update(1)
    if pending:
        indexed += flush(collection, model, pending, args.batch_size)
    pbar.close()

    log(f"indexed {indexed} chunks into {args.collection} at {args.db}")


def flush(collection, model, pending, embed_batch):
    ids = [p[0] for p in pending]
    docs = [p[1] for p in pending]
    metas = [p[2] for p in pending]
    embeddings = model.encode(
        docs, batch_size=embed_batch, normalize_embeddings=True, show_progress_bar=False
    ).tolist()
    collection.add(ids=ids, embeddings=embeddings, metadatas=metas, documents=docs)
    pending.clear()
    return len(ids)


if __name__ == "__main__":
    main()