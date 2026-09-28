# DISCLAIMER
The below RAG implementation was fully coded by OpenCode Big Pickle free model based on my instructions.
I didn't add the `.data/` folder, as I plan to change datacollection later. (you need to create one before running the data collection scripts)
You can use the scripts to build the vector database as described below.
They take in account the heavy throttling happening on wikipedia and back off as needed.

# Jazzbot — RAG ingestion for a local jazz model

RAG corpus + server to give a local LLM accurate information about jazz
music: artists, songs, albums, and venues.

## Pipeline

1. **Manifest** — `scripts/build_manifest.py` walks the Wikipedia category
   tree from jazz root categories and writes the article list to
   `data/manifest.jsonl`.
2. **Fetch** — `scripts/fetch_articles.py` downloads each listed article's
   current wikitext, strips markup into clean prose, and writes
   `data/corpus.jsonl` (resumable).
3. **Embed** — `scripts/embed_corpus.py` chunks each article, embeds the
   chunks with a local sentence-transformers model, and stores them in a
   persistent Chroma vector DB under `data/chroma/` (resumable).
4. **Serve** — `scripts/rag_server.py` embeds an incoming question with the
   same local model, retrieves the nearest chunks from Chroma, and asks an
   OpenAI-compatible chat endpoint (Ollama, llama.cpp, LM Studio, vLLM, …) to
   answer from that context.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Build the manifest

Walk the category tree from several jazz roots. `--depth` is the number of
subcategory levels to follow (larger = more articles, looser focus); `-1`
means unlimited.

```bash
.venv/bin/python scripts/build_manifest.py \
  --roots "Jazz,Jazz people,Jazz albums,Jazz compositions,Jazz clubs,Jazz record labels" \
  --depth 5 \
  --max-pages 200000 \
  --output data/manifest.jsonl
```

> **`--max-pages` caps every run.** It defaults to `50000`, so a crawl is
> silently truncated at 50k pages unless you raise it — that is why the
> example above sets it explicitly. It is checked between categories: once the
> cap is reached the whole walk stops and logs `[stop] reached max_pages=…`.
> (The current manifest is *not* capped — its run was interrupted by throttling
> before finishing.)

> **Rate limits.** From a shared/masked IP the API can return `429` for tens
> of seconds at a time. Both scripts share a single global `Pacer` for the
> whole run, shared across every request (including all worker threads), so
> throttling state is cumulative rather than per-request. Each `429` grows the
> pacing interval up to 60s and honors `Retry-After`; the interval only
> relaxes back toward `--sleep` after a sustained clean stretch (~30s one
> halving step), so a throttled crawl truly self-paces instead of bursting
> every time a single request slips through. Multiple workers are fine — the
> pacer throttles them globally to ~1 request per interval. On every `429` the
> scripts also checkpoint what they have already retrieved to disk before
> backing off, so a long wait never leaves data only in memory. `--sleep`
> already defaults to `2`; raise it to `3`–`5` if you see frequent
> `[rate-limit]` lines. Fetch packs 50 titles into each API request, so a
> throttled network still moves ~1 batch per interval.

## Fetch the articles

Fetch is resumable — pageids already in `--output` are skipped. Use a prefix
subsample `--limit` to smoke-test on a handful of articles first.

```bash
.venv/bin/python scripts/fetch_articles.py \
  --manifest data/manifest.jsonl \
  --output data/corpus.jsonl \
  --workers 2 \
  --sleep 2 \
  --limit 5   # drop in production
```

## Output format

`data/manifest.jsonl` — one line per Wikipedia article. Both `categories`
and `roots` are full `Category:`-prefixed titles (`--roots` entries are
normalized to the prefix too):

```json
{"pageid": 716318, "title": "...", "categories": ["Category:Jazz albums"], "roots": ["Category:Jazz albums"]}
```

`data/corpus.jsonl` — one line per cleaned article:

```json
{"pageid": 716318, "title": "...", "url": "...", "categories": ["..."], "roots": ["..."], "text": "..."}
```

`data/chroma/` — persistent Chroma store, collection `jazz-corpus-v1`. One
document per chunk, id `{pageid}:{seq}`; metadata carries `pageid`, `title`,
`url`, `seq`, and `chunks`. The list-valued `categories`/`roots` fields do
reach Chroma at write time but are silently dropped by its persistent backend,
so they exist only in the JSONL files. Nothing downstream reads them — the RAG
server's sources are built from `title`, `url`, `seq`, and `pageid`.

## Build the vector store

Embed is resumable — articles whose chunks are already indexed are skipped,
so it can be re-run whenever `corpus.jsonl` grows (or pointed at another
corpus entirely).

```bash
.venv/bin/python scripts/embed_corpus.py \
  --corpus data/corpus.jsonl \
  --db data/chroma \
  --model all-MiniLM-L6-v2 \   # swap any sentence-transformers id
  --chunk-chars 1500 \
  --batch-size 64
```

The first run downloads the model (~90 MB) into the Hugging Face cache
(`~/.cache/huggingface`); everything runs on CPU. `--limit` smoke-tests on the
first N articles.

> `data/models/all-MiniLM-L6-v2/` also exists, but it is **not** usable as-is:
> only `config.json` and `pytorch_model.bin` have content, while `modules.json`,
> `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
> `sentence_bert_config.json`, `vocab.txt`, and
> `config_sentence_transformers.json` are all 0 bytes. Passing
> `--model data/models/all-MiniLM-L6-v2` fails with a JSON decode error. Redownload
> the folder (or point `--model` at a complete copy) before relying on it — no
> script references it by default.

## Run the RAG server

Start with an OpenAI-compatible chat endpoint. The default is Ollama on
`http://localhost:11434/v1` with model `tinyllama`, so make sure a model is
pulled first (`ollama list`, then `ollama pull tinyllama`, or point
`LLM_MODEL` at one you already have):

```bash
.venv/bin/python scripts/rag_server.py
```

The model name has to match something the endpoint actually has: Ollama
answers an unknown model with `404`, the same status as a wrong URL path. The
server checks the model at startup, and re-lists the endpoint's models on every
`404` from `/chat/completions` (logging the available ids); other failures go
straight through as errors. `/health` reports `llm_available` and flips `status`
to `degraded` when the configured model is missing — `llm_available` is `null`
when the endpoint's model list can't be read at all.

The current `data/` state already indexes ~170k chunks in the
`jazz-corpus-v1` collection, so it is ready to serve immediately. It loads
the same embedding model as the embed step (which matches the vectors it
stores), opens the Chroma store, and listens on `http://127.0.0.1:8008`.
`/retrieve` skips the LLM and returns only the retrieved chunks, so the
server is testable on its own even before a local model is up.

```bash
curl http://127.0.0.1:8008/health
curl -X POST http://127.0.0.1:8008/retrieve -H 'Content-Type: application/json' \
  -d '{"question": "Who was Chet Baker?", "top_k": 3}'
curl -X POST http://127.0.0.1:8008/query -H 'Content-Type: application/json' \
  -d '{"question": "Who was Chet Baker?"}'
```

Per-query body knobs: `question` (or `query`), `top_k`, `model`, `temperature`,
`max_tokens`. Server flags: `--db`, `--collection`, `--model` (the *embedding*
model — it must be the one `embed_corpus.py` indexed with, or distances are
meaningless), `--base-url` (default `http://localhost:11434/v1`), `--api-key`
(or env `OPENAI_API_KEY`), `--llm-model` (or env `LLM_MODEL`), `--top-k`
(default 6), `--timeout` (default 120s), `--host` (default `127.0.0.1`),
`--port` (default 8008).

## API + web UI

The RAG server is the LLM-facing backend. On top of it sit two more layers:

```
React web UI (:5173) ──▶ jazzbot API (:3000) ──▶ RAG server (:8008) ──▶ Ollama (:11434)
                              │
                              └─ each request & response is stored in data/api/jazzbot.db
```

### Start everything at once

`scripts/start_dev.sh` brings the three tiers up in order, waiting for each one
to actually answer before starting the next, with prefixed/interleaved logs:

```bash
./scripts/start_dev.sh
```

```
RAG (:8008) ──healthy──▶ API (:3000) ──healthy + RAG reachable──▶ UI (:5173)
```

Ctrl-C stops all three (it kills each process group, so no orphans are left).
It installs `server/`+`web/` deps on first run if `node_modules` is missing, and
exits non-zero if any tier fails to come up.

Overridable: `RAG_PORT`, `API_PORT`, `UI_PORT`, `TIMEOUT` (default 180s), plus
`LLM_BASE_URL`, `LLM_MODEL`, `API_KEY` for the RAG server's chat endpoint, and
`RAG_BASE_URL` to point the API at a different RAG server. Each port moves its
tier's process *and* the readiness probe, and the UI's `/api` proxy follows
`API_PORT` — so `API_PORT=4000 UI_PORT=4001 ./scripts/start_dev.sh` shifts the
whole stack. (`API_PORT` is exported as `PORT`, which is the variable the API
itself reads; set `PORT` instead only when running `server/` by hand.)

### 1. Start the API (`server/`)

Node.js + TypeScript (Express). Requires Node ≥ 22 (uses the built-in
`node:sqlite`). It stores conversations and every request/response in SQLite,
and forwards chat messages to the RAG server.

```bash
cd server
npm install
npm run dev          # http://127.0.0.1:3000
```

Env vars: `PORT` (default `3000`), `RAG_BASE_URL` (default
`http://127.0.0.1:8008`), `DB_PATH` (default
`data/api/jazzbot.db`). Endpoints are listed at `GET /` and include:

```
GET   /api/health                          API + RAG reachability
POST  /api/conversations                   create a conversation (optional {"title": "..."})
GET   /api/conversations                   list conversations
GET   /api/conversations/:id               conversation with messages
PATCH /api/conversations/:id               rename ({"title": "..."})
DELETE /api/conversations/:id              204, cascades to the messages
POST  /api/conversations/:id/messages      send "content", forwards to RAG
GET   /api/conversations/:id/messages      stored messages for a conversation
GET   /api/messages                        audit log of every stored request/response
```

`POST /api/conversations/:id/messages` takes `content` plus the RAG knobs it
forwards verbatim — `top_k`, `temperature`, `max_tokens`, `model` (defaulting to
`6` / `0.3` / `600` / the server's own). It answers `201` with
`{userMessage, assistantMessage}` — the user row is persisted *before* the RAG
call, so a failure still returns `502` with both rows plus `error`, and the
assistant row is stored with `status: "error"`.

`GET /api/health` always reports its own `status: "ok"`; RAG state is under
`rag.reachable` / `rag.health` / `rag.error`.

Each stored message keeps the full incoming `request` and the raw `response`,
plus `model`, `sources`, `status`, `error`, and `durationMs`, so the API
doubles as an audit/history layer in front of the model.

### 2. Start the UI (`web/`)

React + TypeScript (Vite). A chat interface with a conversation sidebar and a
status line showing whether the API and RAG server are reachable.

```bash
cd web
npm install
npm run dev          # http://127.0.0.1:5173 (proxies /api to :3000)
```

`npm run build` produces a static bundle under `web/dist`. The bundle calls the
API at the relative path `/api/...`, so whatever serves `dist/` in production
must proxy `/api` to the API — Vite's dev proxy handles that locally.


