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
  sources: Array<{
    pageid?: number;
    title?: string;
    url?: string;
    seq?: number;
    distance?: number;
    snippet?: string;
  }> | null;
  request: unknown;
  response: unknown;
  status: "ok" | "error";
  error: string | null;
  durationMs: number | null;
  createdAt: string;
}

interface SendResult {
  userMessage: Message;
  assistantMessage: Message;
  error?: string;
}

export interface RagStatus {
  status: string;
  rag: {
    reachable: boolean;
    health: { status?: string; chunks?: number; llm?: string } | null;
    error?: string | null;
  };
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    throw new Error(data?.error ?? `request failed: ${res.status}`);
  }
  return data as T;
}

export const api = {
  health: () => req<RagStatus>("/api/health"),

  listConversations: () => req<Conversation[]>("/api/conversations"),

  createConversation: (title?: string) =>
    req<Conversation>("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),

  getConversation: (id: string) =>
    req<Conversation & { messages: Message[] }>(`/api/conversations/${id}`),

  renameConversation: (id: string, title: string) =>
    req<Conversation>(`/api/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),

  deleteConversation: (id: string) =>
    req<void>(`/api/conversations/${id}`, { method: "DELETE" }),

  sendMessage: (conversationId: string, content: string) =>
    req<SendResult>(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
};