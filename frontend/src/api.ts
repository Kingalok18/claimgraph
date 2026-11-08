// All calls to the local API, with the types of what comes back.

export type Relation = "supports" | "contradicts" | "different_setting" | "unclear";

export const RELATIONS: { key: Relation; label: string; colour: string; ink: string; line: string }[] = [
  { key: "supports", label: "Supports", colour: "var(--supports)", ink: "var(--supports)", line: "solid" },
  { key: "contradicts", label: "Contradicts", colour: "var(--contradicts)", ink: "var(--contradicts)", line: "solid" },
  { key: "different_setting", label: "Different setting", colour: "var(--setting)", ink: "var(--setting-ink)", line: "dashed" },
  { key: "unclear", label: "Unclear", colour: "var(--unclear)", ink: "var(--unclear-ink)", line: "dotted" },
];

export function relationInfo(relation: string | null | undefined) {
  return RELATIONS.find((item) => item.key === relation) ?? RELATIONS[3];
}

export interface Claim {
  claim_id: string;
  paper_id: string;
  text: string;
  intervention: string;
  outcome: string;
  direction: string;
  conditions: Record<string, string | null>;
  hedged: boolean;
  section: string;
  page: number | null;
  quote: string;
  quote_verified: boolean;
  paper_title?: string;
  year?: number | null;
}

export interface PaperSummary {
  id: string;
  title: string | null;
  authors: string | null;
  year: number | null;
  venue: string | null;
  doi: string | null;
  source: string;
  parse_status: string;
  parse_error: string | null;
  extract_status: string;
  chunks_done: number;
  chunks_total: number;
  claims_dropped: number;
  missing: boolean;
  n_pages: number | null;
  claims: number;
}

export interface PaperDetail extends PaperSummary {
  abstract: string | null;
  has_pdf: boolean;
  sections: { name: string; heading: string | null; page_start: number | null; char_start: number }[];
  claims_list: Claim[];
}

export interface Library {
  papers: PaperSummary[];
  folder: string;
  pending: number;
}

export interface Job {
  id: number;
  kind: string;
  status: string;
  progress: { step?: string; label?: string; done?: number; total?: number; current?: string | null; summary?: Record<string, unknown> | null };
  error: string | null;
}

export interface TopicSummary {
  id: number;
  label: string;
  source: string;
  question_text: string | null;
  claims: number;
  pairs: number;
  supports: number;
  contradicts: number;
  different_setting: number;
  unclear: number;
}

export interface GraphData {
  topic: { id: number; label: string; source: string; question_text: string | null };
  elements: {
    nodes: { data: { id: string; label: string; text: string; paper_id: string; paper_title: string; year: number | null; colour: string } }[];
    edges: { data: { id: string; pair_id: number; source: string; target: string; relation: Relation } }[];
  };
  papers: { paper_id: string; colour: string; title: string }[];
}

export interface PairSide {
  claim_id: string;
  text: string;
  quote: string;
  page: number | null;
  direction: string;
  conditions: Record<string, string | null>;
  hedged: boolean;
  paper_id: string;
  paper_title: string;
  year: number | null;
}

export interface Pair {
  id: number;
  topic_id: number;
  pair_key: string;
  a: PairSide;
  b: PairSide;
  nli_label: string | null;
  nli_prob: number | null;
  conditions_match: string | null;
  differing_fields: string[];
  field_status: Record<string, string>;
  reason: string | null;
  decided_by: string | null;
  relation: Relation | null;
  user_label: Relation | null;
}

export interface Report {
  id: number;
  topic_id: number;
  topic_label: string;
  text: string;
  cited_claim_ids: string[];
  removed_count: number;
  model: string;
  created_at: string;
  claims: Record<string, Claim>;
}

export interface Health {
  status: string;
  app_name: string;
  default_theme: string;
  services: {
    database: { ok: boolean; kind: string; detail: string };
    llm: { ok: boolean; provider: string; model: string; detail: string };
    embeddings: { ok: boolean; model: string };
    nli: { ok: boolean; model: string; finetuned: boolean };
    vector_store: { ok: boolean; mode: string; path: string };
  };
  setup_problems: { service: string; blocking: boolean; detail: string }[];
  settings: { papers_dir: string; parsing_backend: string; online_search: boolean };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `The request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* the plain message above is enough */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export const api = {
  health: () => request<Health>("/health"),
  library: () => request<Library>("/library"),
  scan: () => post<{ job_id: number }>("/library/scan?background=true"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ saved_as: string; new: number; failed: number }>("/library/upload", { method: "POST", body: form });
  },
  paper: (id: string) => request<PaperDetail>(`/papers/${id}`),
  analyze: (limit?: number, paper_ids?: string[]) =>
    post<{ job_id: number }>("/analyze", { ...(limit ? { limit } : {}), ...(paper_ids?.length ? { paper_ids } : {}) }),
  activeJob: () => request<{ job: Job | null }>("/jobs/active"),
  job: (id: number) => request<Job>(`/jobs/${id}`),
  cancelJob: (id: number) => post<{ cancelling: boolean }>(`/jobs/${id}/cancel`),
  topics: () => request<{ topics: TopicSummary[] }>("/topics"),
  addQuestion: (question: string) => post<{ job_id: number }>("/topics/question", { question }),
  graph: (id: number) => request<GraphData>(`/topics/${id}/graph`),
  pair: (id: number) => request<Pair>(`/pairs/${id}`),
  labelPair: (id: number, relation: Relation) => post<Pair>(`/pairs/${id}/label`, { relation }),
  report: (id: number, regenerate = false) => request<Report>(`/topics/${id}/report${regenerate ? "?regenerate=true" : ""}`),
  search: (q: string) => request<{ claims: Claim[]; papers: PaperSummary[] }>(`/search?q=${encodeURIComponent(q)}`),
  labelStats: () => request<{ pair_labels: number; claim_labels: number; pairs_total: number; claims_total: number }>("/labels/stats"),
  pairQueue: () => request<{ items: { id: number }[] }>("/labels/queue?kind=pair"),
  claimQueue: () => request<{ items: Claim[] }>("/labels/queue?kind=extraction"),
  labelClaim: (claim_id: string, verdict: string) => post<{ ok: boolean }>("/labels/claim", { claim_id, verdict }),
};
