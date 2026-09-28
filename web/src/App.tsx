import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Conversation, type Message, type RagStatus } from "./api";

function Sources({ message }: { message: Message }) {
  const [open, setOpen] = useState(false);
  if (!message.sources || message.sources.length === 0) return null;
  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "hide" : "show"} {message.sources.length} source
        {message.sources.length === 1 ? "" : "s"}
      </button>
      {open && (
        <ul>
          {message.sources.map((s, i) => (
            <li key={i}>
              {s.title ?? `chunk ${s.seq ?? "?"}`}
              {s.url ? (
                <>
                  {" "}
                  <a href={s.url} target="_blank" rel="noreferrer">
                    link
                  </a>
                </>
              ) : null}
              {s.distance !== undefined ? (
                <span className="muted"> · d={s.distance}</span>
              ) : null}
              {s.snippet ? <p className="snippet">{s.snippet}</p> : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Bubble({ message }: { message: Message }) {
  return (
    <div className={`bubble ${message.role} ${message.status}`}>
      {message.content ||
        (message.status === "error" ? (
          <span className="muted">⚠ {message.error}</span>
        ) : null)}
      {message.durationMs ? (
        <span className="meta">
          {message.model ?? ""}
          {message.model ? " · " : ""}
          {(message.durationMs / 1000).toFixed(1)}s
        </span>
      ) : null}
      <Sources message={message} />
    </div>
  );
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[] | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [rag, setRag] = useState<RagStatus | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshConversations = useCallback(async () => {
    setConversations(await api.listConversations());
  }, []);

  const loadConversation = useCallback(async (id: string) => {
    const conv = await api.getConversation(id);
    setActiveId(conv.id);
    setMessages(conv.messages);
  }, []);

  useEffect(() => {
    api.health().then(setRag).catch(() => setRag(null));
    refreshConversations().catch(() => setConversations([]));
  }, [refreshConversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const newConversation = useCallback(async () => {
    const conv = await api.createConversation();
    setActiveId(conv.id);
    setMessages([]);
    await refreshConversations();
  }, [refreshConversations]);

  const selectConversation = useCallback(
    async (id: string) => {
      try {
        await loadConversation(id);
      } catch {
        // conversation was deleted elsewhere
        await refreshConversations();
      }
    },
    [loadConversation, refreshConversations],
  );

  const deleteConversation = useCallback(
    async (id: string) => {
      await api.deleteConversation(id);
      if (activeId === id) {
        setActiveId(null);
        setMessages([]);
      }
      await refreshConversations();
    },
    [activeId, refreshConversations],
  );

  const send = useCallback(async () => {
    const content = input.trim();
    if (!content || sending) return;
    let convId = activeId;
    if (!convId) {
      const conv = await api.createConversation();
      convId = conv.id;
      setActiveId(convId);
      await refreshConversations();
    }
    setInput("");
    setSending(true);
    setMessages((prev) => [...prev, placeholderUser(content)]);
    try {
      const result = await api.sendMessage(convId, content);
      setMessages((prev) => {
        const withoutPlaceholder = [...prev];
        withoutPlaceholder.pop();
        return [
          ...withoutPlaceholder,
          result.userMessage,
          result.assistantMessage,
        ];
      });
    } catch (err) {
      const text = err instanceof Error ? err.message : String(err);
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          conversationId: convId,
          role: "assistant",
          content: "",
          model: null,
          sources: null,
          request: null,
          response: null,
          status: "error",
          error: text,
          durationMs: null,
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setSending(false);
      await refreshConversations();
    }
  }, [activeId, input, refreshConversations, sending]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-head">
          <h1>🎷 Jazzbot</h1>
          <button className="new" onClick={newConversation}>
            + New chat
          </button>
        </div>
        {rag ? (
          <div className={`status ${rag.rag.reachable ? "ok" : "bad"}`}>
            UI → API: ok · RAG:{" "}
            {rag.rag.reachable
              ? `ok (${rag.rag.health?.chunks ?? "?"} chunks, ${rag.rag.health?.llm ?? "?"})`
              : `down (${rag.rag.error ?? "unknown"})`}
          </div>
        ) : (
          <div className="status bad">API offline</div>
        )}
        <nav className="conversations">
          {conversations === null ? (
            <p className="muted">loading…</p>
          ) : conversations.length === 0 ? (
            <p className="muted">No conversations yet.</p>
          ) : (
            conversations.map((c) => (
              <div
                key={c.id}
                className={`conv ${c.id === activeId ? "active" : ""}`}
                onClick={() => selectConversation(c.id)}
              >
                <span className="conv-title">
                  {c.title || "Untitled"}
                  <span className="muted">
                    {" "}
                    · {c.messageCount} msg
                  </span>
                </span>
                <button
                  className="delete"
                  title="Delete conversation"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation(c.id);
                  }}
                >
                  ✕
                </button>
              </div>
            ))
          )}
        </nav>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty">
              <h2>Ask about jazz</h2>
              <p>
                Questions are answered by the local Ollama model using the RAG
                corpus. Every request and response is stored by the API.
              </p>
            </div>
          ) : (
            messages.map((m) => <Bubble key={m.id} message={m} />)
          )}
          <div ref={bottomRef} />
        </div>
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
        >
          <textarea
            value={input}
            placeholder="Who was Chet Baker?"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button type="submit" disabled={sending || !input.trim()}>
            {sending ? "…" : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}

function placeholderUser(content: string): Message {
  return {
    id: `tmp-${Date.now()}`,
    conversationId: "",
    role: "user",
    content,
    model: null,
    sources: null,
    request: null,
    response: null,
    status: "ok",
    error: null,
    durationMs: null,
    createdAt: new Date().toISOString(),
  };
}