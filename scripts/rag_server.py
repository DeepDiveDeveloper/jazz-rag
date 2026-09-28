#!/usr/bin/env python3
"""RAG server: retrieve jazz context and answer with a local LLM.

Loads the Chroma vector store built by embed_corpus.py, embeds an incoming
question with the same sentence-transformers model, retrieves the nearest
chunks, and asks an OpenAI-compatible chat endpoint (Ollama, llama.cpp,
LM Studio, vLLM, …) to answer from that context. Serves a tiny JSON HTTP API
so it runs with nothing but the stdlib + the deps already in requirements.txt.

Endpoints:
  GET  /health                 store/model status
  POST /query                  {"question": "...", "top_k": 6, ...}
  POST /retrieve               same body, but skips the LLM and returns chunks
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chromadb
import requests
from sentence_transformers import SentenceTransformer

DEFAULT_COLLECTION = "jazz-corpus-v1"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "tinyllama")
DEFAULT_TOP_K = 6

SYSTEM_PROMPT = (
    "You are Jazzbot, an assistant that answers questions about jazz music: "
    "artists, songs, albums, and venues. Answer using ONLY the provided "
    "context below. When you rely on a source, mention its title. If the "
    "context does not contain the answer, say so plainly instead of "
    "guessing."
)


def log(message):
    sys.stderr.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def _normalize_model(name):
    name = (name or "").strip().lower()
    if name.endswith(":latest"):
        name = name[: -len(":latest")]
    return name


def build_context(hits):
    lines = []
    for i, (doc, meta) in enumerate(zip(hits["documents"], hits["metadatas"]), 1):
        title = meta.get("title", "?")
        seq = meta.get("seq", 0)
        url = meta.get("url", "")
        head = f"[{i}] {title} (chunk {seq + 1})"
        if url:
            head += f" — {url}"
        lines.append(f"{head}\n{doc}")
    return "\n\n".join(lines)


def format_sources(hits):
    sources = []
    for doc, meta, dist in zip(
        hits["documents"], hits["metadatas"], hits["distances"]
    ):
        sources.append({
            "pageid": meta.get("pageid"),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "seq": meta.get("seq"),
            "distance": round(dist, 4),
            "snippet": doc[:400],
        })
    return sources


class RagServer:
    def __init__(
        self,
        db,
        collection_name,
        embed_model,
        base_url,
        api_key,
        llm_model,
        top_k,
        timeout,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.llm_model = llm_model
        self.top_k = top_k
        self.timeout = timeout
        self.collection_name = collection_name
        self._models = None

        log(f"loading embedding model {embed_model}…")
        self.model = SentenceTransformer(embed_model)
        log(f"opening Chroma store at {db}…")
        client = chromadb.PersistentClient(path=db)
        self.collection = client.get_collection(collection_name)
        self.check_llm()

    def count(self):
        return self.collection.count()

    def available_models(self):
        if self._models is None:
            self._models = []
            try:
                resp = requests.get(
                    f"{self.base_url}/models", timeout=min(self.timeout, 10)
                )
                resp.raise_for_status()
                data = resp.json()
                entries = data.get("data") or data.get("models") or []
                self._models = [
                    m.get("id") or m.get("name") or m.get("model")
                    for m in entries
                ]
                self._models = [m for m in self._models if m]
            except Exception as exc:
                log(f"could not list models at {self.base_url}: {exc!r}")
        return self._models

    def llm_available(self):
        models = self.available_models()
        if not models:
            return None
        return _normalize_model(self.llm_model) in {
            _normalize_model(m) for m in models
        }

    def check_llm(self):
        status = self.llm_available()
        if status is None:
            log(f"model list unavailable; cannot verify {self.llm_model!r}")
        elif status:
            log(f"llm model {self.llm_model!r} available at {self.base_url}")
        else:
            log(
                f"WARNING: llm model {self.llm_model!r} is not installed at "
                f"{self.base_url} — queries will fail with 404. Available: "
                f"{', '.join(self._models) or 'none'}. "
                f"Pull it (ollama pull {self.llm_model}) or set --llm-model."
            )

    def retrieve(self, question, top_k):
        emb = self.model.encode(
            [question], normalize_embeddings=True
        ).tolist()
        hits = self.collection.query(
            query_embeddings=emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return {
            "documents": hits["documents"][0],
            "metadatas": hits["metadatas"][0],
            "distances": hits["distances"][0],
        }

    def answer(self, question, top_k, llm_model, temperature, max_tokens):
        hits = self.retrieve(question, top_k)
        context = build_context(hits)
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        body = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        log(f"asking {llm_model} at {self.base_url}/chat/completions…")
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            self._models = None
            available = ", ".join(self.available_models()) or "none reported"
            detail = resp.text.strip()[:300]
            raise RuntimeError(
                f"LLM endpoint {self.base_url}/chat/completions returned 404 — "
                f"model {llm_model!r} is not available. "
                f"Available models: {available}. "
                f"Endpoint said: {detail}"
            )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer, format_sources(hits)


def make_handler(server, args):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *vals):
            log(f"{self.client_address[0]} {fmt % vals}")

        def _send_json(self, obj, status=200):
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                raise ValueError("empty request body")
            raw = self.rfile.read(length)
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def do_GET(self):
            if self.path == "/health":
                available = server.llm_available()
                self._send_json({
                    "status": "ok" if available is not False else "degraded",
                    "collection": server.collection_name,
                    "chunks": server.count(),
                    "llm": server.llm_model,
                    "llm_available": available,
                    "llm_models": server.available_models(),
                })
            elif self.path == "/":
                self._send_json({"service": "jazzbot", "endpoints": ["/health", "/query", "/retrieve"]})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                data = self._read_body()
                question = data.get("question") or data.get("query")
                if not question:
                    raise ValueError("'question' is required")
                top_k = int(data.get("top_k", server.top_k))
                llm_model = data.get("model", server.llm_model)
                if self.path == "/retrieve":
                    hits = server.retrieve(question, top_k)
                    self._send_json({
                        "question": question,
                        "sources": format_sources(hits),
                    })
                elif self.path == "/query":
                    temperature = float(data.get("temperature", 0.3))
                    max_tokens = int(data.get("max_tokens", 600))
                    answer, sources = server.answer(
                        question, top_k, llm_model, temperature, max_tokens
                    )
                    self._send_json({
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                        "model": llm_model,
                    })
                else:
                    self._send_json({"error": "not found"}, 404)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
            except Exception as exc:
                log(f"error handling {self.path}: {exc!r}")
                self._send_json({"error": str(exc)}, 500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/chroma",
                    help="Chroma persistent directory (default: data/chroma)")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="embedding model id (default: all-MiniLM-L6-v2)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="OpenAI-compatible base URL incl. /v1 "
                         "(default: http://localhost:11434/v1)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", None),
                    help="bearer token for the LLM endpoint (default: env OPENAI_API_KEY)")
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                    help="model name at the endpoint (default: env LLM_MODEL or tinyllama)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help="chunks to retrieve per query (default: 6)")
    ap.add_argument("--timeout", type=int, default=120,
                    help="seconds to wait on the LLM call (default: 120)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()

    server = RagServer(
        db=args.db,
        collection_name=args.collection,
        embed_model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        llm_model=args.llm_model,
        top_k=args.top_k,
        timeout=args.timeout,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server, args))
    log(f"jazzbot RAG listening on http://{args.host}:{args.port} "
        f"(collection={args.collection}, chunks={server.count()}, "
        f"llm={args.llm_model} at {args.base_url})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()