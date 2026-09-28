export interface RagHealth {
  status: string;
  collection?: string;
  chunks?: number;
  llm?: string;
}

export interface RagQueryResult {
  question: string;
  answer: string;
  sources: Array<Record<string, unknown>>;
  model: string;
}

export interface QueryOptions {
  top_k?: number;
  model?: string;
  temperature?: number;
  max_tokens?: number;
}

export class RagClient {
  constructor(
    private baseUrl: string,
    private timeoutMs = 120_000,
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  async health(): Promise<RagHealth> {
    return (await this.get("/health")) as RagHealth;
  }

  async query(question: string, opts: QueryOptions = {}): Promise<RagQueryResult> {
    const body: Record<string, unknown> = { question };
    if (opts.top_k !== undefined) body.top_k = opts.top_k;
    if (opts.model !== undefined) body.model = opts.model;
    if (opts.temperature !== undefined) body.temperature = opts.temperature;
    if (opts.max_tokens !== undefined) body.max_tokens = opts.max_tokens;
    return (await this.post("/query", body)) as RagQueryResult;
  }

  private async get(path: string): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new Error(`RAG server ${path} failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }

  private async post(path: string, body: Record<string, unknown>): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new Error(`RAG server ${path} failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }
}