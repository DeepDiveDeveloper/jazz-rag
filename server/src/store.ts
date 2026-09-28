import { DatabaseSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface Message {
  id: string;
  conversationId: string;
  role: "user" | "assistant";
  content: string;
  model: string | null;
  sources: unknown;
  request: unknown;
  response: unknown;
  status: "ok" | "error";
  error: string | null;
  durationMs: number | null;
  createdAt: string;
}

function now(): string {
  return new Date().toISOString();
}

export function openDb(dbPath: string): DatabaseSync {
  mkdirSync(dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  db.exec(`
    CREATE TABLE IF NOT EXISTS conversations (
      id         TEXT PRIMARY KEY,
      title      TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
      id              TEXT PRIMARY KEY,
      conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role            TEXT NOT NULL,
      content         TEXT NOT NULL,
      model           TEXT,
      sources         TEXT,
      request         TEXT,
      response        TEXT,
      status          TEXT NOT NULL DEFAULT 'ok',
      error           TEXT,
      duration_ms     INTEGER,
      created_at      TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
  `);
  db.exec("PRAGMA foreign_keys = ON;");
  return db;
}

export class Store {
  constructor(private db: DatabaseSync) {}

  createConversation(title = "New conversation"): Conversation {
    const id = randomUUID();
    const ts = now();
    this.db
      .prepare(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)"
      )
      .run(id, title, ts, ts);
    return { id, title, createdAt: ts, updatedAt: ts, messageCount: 0 };
  }

  listConversations(): Conversation[] {
    return this.db
      .prepare(
        `SELECT c.id, c.title, c.created_at AS createdAt, c.updated_at AS updatedAt,
                (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS messageCount
         FROM conversations c ORDER BY c.updated_at DESC`
      )
      .all() as unknown as Conversation[];
  }

  getConversation(id: string): Conversation | null {
    const row = this.db
      .prepare(
        `SELECT c.id, c.title, c.created_at AS createdAt, c.updated_at AS updatedAt,
                (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS messageCount
         FROM conversations c WHERE c.id = ?`
      )
      .get(id) as unknown as Conversation | undefined;
    return row ?? null;
  }

  renameConversation(id: string, title: string): boolean {
    const res = this.db
      .prepare("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?")
      .run(title, now(), id);
    return res.changes > 0;
  }

  deleteConversation(id: string): boolean {
    const res = this.db.prepare("DELETE FROM conversations WHERE id = ?").run(id);
    return res.changes > 0;
  }

  listMessages(conversationId: string): Message[] {
    const rows = this.db
      .prepare(
        `SELECT id, conversation_id AS conversationId, role, content, model,
                sources, request, response, status, error,
                duration_ms AS durationMs, created_at AS createdAt
         FROM messages WHERE conversation_id = ? ORDER BY created_at ASC`
      )
      .all(conversationId) as unknown as Array<Record<string, unknown>>;
    return rows.map(parseMessage);
  }

  listAllMessages(): Array<Message & { conversationTitle: string }> {
    const rows = this.db
      .prepare(
        `SELECT m.id, m.conversation_id AS conversationId, m.role, m.content, m.model,
                m.sources, m.request, m.response, m.status, m.error,
                m.duration_ms AS durationMs, m.created_at AS createdAt,
                c.title AS conversationTitle
         FROM messages m JOIN conversations c ON c.id = m.conversation_id
         ORDER BY m.created_at DESC`
      )
      .all() as unknown as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      ...parseMessage(r),
      conversationTitle: r.conversationTitle as string,
    }));
  }

  getMessage(id: string): Message | null {
    const row = this.db
      .prepare(
        `SELECT id, conversation_id AS conversationId, role, content, model,
                sources, request, response, status, error,
                duration_ms AS durationMs, created_at AS createdAt
         FROM messages WHERE id = ?`
      )
      .get(id) as unknown as Record<string, unknown> | undefined;
    return row ? parseMessage(row) : null;
  }

  insertMessage(
    conversationId: string,
    msg: Omit<Message, "id" | "conversationId" | "createdAt">
  ): Message {
    const id = randomUUID();
    const ts = now();
    this.db
      .prepare(
        `INSERT INTO messages
           (id, conversation_id, role, content, model, sources, request, response, status, error, duration_ms, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        id,
        conversationId,
        msg.role,
        msg.content,
        msg.model,
        jsonOrNull(msg.sources),
        jsonOrNull(msg.request),
        jsonOrNull(msg.response),
        msg.status,
        msg.error,
        msg.durationMs,
        ts
      );
    this.db
      .prepare("UPDATE conversations SET updated_at = ? WHERE id = ?")
      .run(ts, conversationId);
    return {
      id,
      conversationId,
      role: msg.role,
      content: msg.content,
      model: msg.model,
      sources: msg.sources,
      request: msg.request,
      response: msg.response,
      status: msg.status,
      error: msg.error,
      durationMs: msg.durationMs,
      createdAt: ts,
    };
  }
}

function jsonOrNull(v: unknown): string | null {
  return v === undefined || v === null ? null : JSON.stringify(v);
}

function parseMessage(row: Record<string, unknown>): Message {
  const parse = (v: unknown): unknown => {
    if (typeof v !== "string" || v === "") return null;
    try {
      return JSON.parse(v);
    } catch {
      return v;
    }
  };
  return {
    id: row.id as string,
    conversationId: row.conversationId as string,
    role: row.role as "user" | "assistant",
    content: row.content as string,
    model: (row.model as string | null) ?? null,
    sources: parse(row.sources),
    request: parse(row.request),
    response: parse(row.response),
    status: row.status as "ok" | "error",
    error: row.error as string | null,
    durationMs: row.durationMs as number | null,
    createdAt: row.createdAt as string,
  };
}