import express from "express";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { openDb, Store } from "./store.js";
import { RagClient } from "./rag.js";
import { buildRouter } from "./routes.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.PORT ?? 3000);
const RAG_BASE_URL = process.env.RAG_BASE_URL ?? "http://127.0.0.1:8008";
const DB_PATH = process.env.DB_PATH ?? join(__dirname, "..", "..", "data", "api", "jazzbot.db");

const store = new Store(openDb(DB_PATH));
const rag = new RagClient(RAG_BASE_URL);

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use("/api", buildRouter(store, rag));

app.get("/", (_req, res) => {
  res.json({
    service: "jazzbot-api",
    endpoints: [
      "GET  /api/health",
      "POST /api/conversations",
      "GET  /api/conversations",
      "GET  /api/conversations/:id",
      "PATCH /api/conversations/:id",
      "DELETE /api/conversations/:id",
      "POST /api/conversations/:id/messages",
      "GET  /api/conversations/:id/messages",
      "GET  /api/messages",
    ],
  });
});

app.use(
  (
    err: unknown,
    _req: express.Request,
    res: express.Response,
    _next: express.NextFunction,
  ) => {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ error: message });
  },
);

app.listen(PORT, () => {
  console.log(`jazzbot API listening on http://127.0.0.1:${PORT}`);
  console.log(`forwarding to RAG server at ${RAG_BASE_URL}`);
  console.log(`storing requests/responses in ${DB_PATH}`);
});