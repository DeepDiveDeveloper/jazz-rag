import { Router, type NextFunction, type Request, type Response } from "express";
import { Store } from "./store.js";
import { RagClient, type QueryOptions } from "./rag.js";

type AsyncHandler = (
  req: Request,
  res: Response,
  next: NextFunction,
) => Promise<unknown>;

const wrap =
  (fn: AsyncHandler) =>
  (req: Request, res: Response, next: NextFunction): void => {
    fn(req, res, next).catch(next);
  };

export function buildRouter(store: Store, rag: RagClient): Router {
  const router = Router();

  router.get(
    "/health",
    wrap(async (_req, res) => {
      let ragHealth: unknown = null;
      let ragError: string | null = null;
      try {
        ragHealth = await rag.health();
      } catch (err) {
        ragError = err instanceof Error ? err.message : String(err);
      }
      res.json({
        status: "ok",
        rag: { reachable: ragError === null, health: ragHealth, error: ragError },
      });
    }),
  );

  router.post(
    "/conversations",
    wrap(async (req, res) => {
      const title =
        typeof req.body?.title === "string" && req.body.title.trim()
          ? req.body.title.trim()
          : "New conversation";
      res.status(201).json(store.createConversation(title));
    }),
  );

  router.get(
    "/conversations",
    wrap(async (_req, res) => {
      res.json(store.listConversations());
    }),
  );

  router.get(
    "/conversations/:id",
    wrap(async (req, res) => {
      const conv = store.getConversation(req.params.id);
      if (!conv) {
        res.status(404).json({ error: "conversation not found" });
        return;
      }
      res.json({ ...conv, messages: store.listMessages(conv.id) });
    }),
  );

  router.patch(
    "/conversations/:id",
    wrap(async (req, res) => {
      const title =
        typeof req.body?.title === "string" && req.body.title.trim()
          ? req.body.title.trim()
          : null;
      if (!title) {
        res.status(400).json({ error: "'title' is required" });
        return;
      }
      if (!store.renameConversation(req.params.id, title)) {
        res.status(404).json({ error: "conversation not found" });
        return;
      }
      res.json(store.getConversation(req.params.id));
    }),
  );

  router.delete(
    "/conversations/:id",
    wrap(async (req, res) => {
      if (!store.deleteConversation(req.params.id)) {
        res.status(404).json({ error: "conversation not found" });
        return;
      }
      res.status(204).end();
    }),
  );

  router.get(
    "/conversations/:id/messages",
    wrap(async (req, res) => {
      if (!store.getConversation(req.params.id)) {
        res.status(404).json({ error: "conversation not found" });
        return;
      }
      res.json(store.listMessages(req.params.id));
    }),
  );

  router.post(
    "/conversations/:id/messages",
    wrap(async (req, res) => {
      const content =
        typeof req.body?.content === "string" ? req.body.content.trim() : "";
      if (!content) {
        res.status(400).json({ error: "'content' is required" });
        return;
      }
      const conv = store.getConversation(req.params.id);
      if (!conv) {
        res.status(404).json({ error: "conversation not found" });
        return;
      }

      const num = (v: unknown, dflt: number): number | undefined => {
        const n = Number(v);
        return Number.isFinite(n) && v !== undefined && v !== null ? n : dflt;
      };
      const options: QueryOptions = {
        top_k: num(req.body.top_k, 6),
        temperature: num(req.body.temperature, 0.3),
        max_tokens: num(req.body.max_tokens, 600),
      };
      if (typeof req.body.model === "string" && req.body.model.trim()) {
        options.model = req.body.model.trim();
      }

      const userMsg = store.insertMessage(conv.id, {
        role: "user",
        content,
        model: null,
        sources: null,
        request: { question: content, ...options },
        response: null,
        status: "ok",
        error: null,
        durationMs: null,
      });

      const started = Date.now();
      try {
        const result = await rag.query(content, options);
        const durationMs = Date.now() - started;
        const assistantMsg = store.insertMessage(conv.id, {
          role: "assistant",
          content: result.answer,
          model: result.model,
          sources: result.sources,
          request: { question: content, ...options },
          response: result,
          status: "ok",
          error: null,
          durationMs,
        });
        res
          .status(201)
          .json({ userMessage: userMsg, assistantMessage: assistantMsg });
      } catch (err) {
        const durationMs = Date.now() - started;
        const error = err instanceof Error ? err.message : String(err);
        const assistantMsg = store.insertMessage(conv.id, {
          role: "assistant",
          content: "",
          model: null,
          sources: null,
          request: { question: content, ...options },
          response: null,
          status: "error",
          error,
          durationMs,
        });
        res.status(502).json({ userMessage: userMsg, assistantMessage: assistantMsg, error });
      }
    }),
  );

  router.get(
    "/messages",
    wrap(async (_req, res) => {
      res.json(store.listAllMessages());
    }),
  );

  return router;
}