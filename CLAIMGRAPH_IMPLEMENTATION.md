# ClaimGraph — Research Claim & Contradiction Mapper
## Implementation spec for Claude Code

> **How to use this file**
> 1. Put this file inside the folder **`research-project`** on your Desktop.
> 2. Open Claude Code in that folder.
> 3. Say: *"Read CLAIMGRAPH_IMPLEMENTATION.md. First ask me all the questions in Section 0.2 in one message. After I answer, build the whole project on your own, test it yourself, fix what fails, and only come back when the Definition of Done in Section 13 passes."*

---

## 0. Instructions for the coding agent

### 0.1 Workspace boundary (strict)

- The **only** project folder is `research-project` on the user's Desktop:
  - Windows: `%USERPROFILE%\Desktop\research-project` (if Desktop is synced by OneDrive: `%USERPROFILE%\OneDrive\Desktop\research-project`)
  - macOS / Linux: `~/Desktop/research-project`
- At the start, print the current working directory and confirm it is that folder. If it is not, stop and ask.
- **Create, edit and delete files only inside this folder.** The Python virtual environment (`.venv/`), data, models, logs, results and the frontend all live inside it.
- Allowed outside the folder, and only for these purposes: package caches (pip, npm), the Hugging Face model cache (`~/.cache/huggingface`), Ollama's model store, and Docker images and named volumes. Never modify any other file on the computer.
- Never run global installs that change the system (`sudo`, `npm -g`, editing PATH or shell profiles) without asking.

### 0.2 Ask first: one message, then no more questions

Before writing any code:

1. **Detect** what you can yourself and do not ask about it: OS, Python version, Node version, whether `docker`, `ollama` and `git` are installed, GPU (`nvidia-smi`), total RAM, free disk space, and whether ports 8000, 5173, 5432, 6333 and 8070 are free.
2. Show the user what you detected in a short list.
3. Ask **all** of the questions below in **one** message, numbered, each with your recommended default in brackets, so the user can reply "defaults" or answer only some:

| # | Question | Default |
|---|---|---|
| 1 | Where are your papers? (They should be PDFs inside `research-project/papers/`. How many, and mainly which field?) | `research-project/papers/` |
| 2 | Name 1–3 research questions you want to explore. Or should the app discover the main topics automatically from your papers? | Discover automatically |
| 3 | May the app also search online (OpenAlex and arXiv) to add more papers? If yes, give an email address for OpenAlex's polite pool. | No (only your papers) |
| 4 | Main language model. With less than 16 GB RAM, the 3B model is used. | `qwen2.5:7b-instruct` via Ollama |
| 5 | Database: PostgreSQL in Docker, or SQLite (simpler, no Docker needed)? | SQLite if Docker is missing, else PostgreSQL |
| 6 | PDF parsing: GROBID in Docker (best quality, needs Docker), or PyMuPDF (no Docker)? | GROBID if Docker is available |
| 7 | The NLI model works best when fine-tuned on a GPU. Will you run the provided Kaggle/Colab notebook later, or should the app use the base model without fine-tuning? | Base model now; notebook provided for later |
| 8 | Evaluation datasets to download: SciFact (small), SciFact-Open (large, needs a GPU to embed), BioDivergence (check its licence). Which ones? | SciFact only |
| 9 | Will you hand-label claims or claim pairs later for evaluation? | Yes, later; build the labelling tool |
| 10 | Hugging Face token (only needed for gated models; leave blank otherwise). | Blank |
| 11 | App name shown in the interface, and light or dark theme by default. | "ClaimGraph", light |

4. Save the answers to `configs/user_answers.yaml`, then generate `.env` and the config files from them.
5. **Do not ask further questions** unless something is truly blocked (Section 0.3).

### 0.3 Autonomy rules

- Build **all** milestones in Section 10 in order **without stopping to report**. After each milestone, run its checks yourself, fix failures, and continue.
- Keep a short log in `BUILD_LOG.md`: one entry per milestone with what was built, test results, and decisions made.
- If a step needs the user (a GPU notebook run, hand labelling, a licence decision), use the **fallback** written in that milestone, note it in `BUILD_LOG.md` under "Waiting for the user", and keep going. Never stop the whole build for an optional step.
- Ask the user only if: an action would delete user files (including anything in `papers/`), a download is larger than 5 GB, or something outside Section 0.1's boundary is needed.
- When everything in Section 13 passes, send **one** final message: how to start the app (one command), what was tested and the results, and the list of optional steps waiting for the user.

### 0.4 General rules

- Every module in `src/` has unit tests in `tests/`. Do not skip or weaken a test to make it pass; fix the code.
- **Never invent numbers.** Metrics come only from scripts in `src/claimgraph/eval/`, saved to `results/*.json`.
- **No paid APIs.** All models are free and open from Hugging Face, run locally (Ollama or `transformers`).
- Secrets go in `.env` (git-ignored). Provide `.env.example`.
- Never modify or delete the user's PDFs in `papers/`. Read them only.
- Cross-platform: the app must start on Windows, macOS and Linux. Do not depend on `make`; provide `python scripts/run.py <command>` (Section 12).

---

## 1. Goal

The user provides research papers (PDFs). The system:

1. imports and parses the papers,
2. extracts **claims** with their **study conditions** and an **exact quote** from the paper,
3. verifies every quote really exists in the paper,
4. groups claims into **topics** (discovered automatically) or under the user's **research questions**,
5. labels each pair of related claims as **supports / contradicts / different setting / unclear**, and
6. shows a **claim graph**, a side-by-side **pair view** with the PDF page, and an **evidence-map report** where every sentence cites existing claim IDs.

Optional: if the user allows it (question 3), an online search can add more papers from OpenAlex and arXiv.

**Core research idea:** a pair is *contradicts* only if an NLI model sees a conflict **and** the study conditions (dataset, model or population, metric, setting) are comparable. Otherwise it is *different setting*.

---

## 2. Tech stack (all free)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 (inside `research-project/.venv`) | |
| **Orchestration** | **LangChain** (`langchain`, `langchain-core`, `langchain-community`) | LCEL chains, structured output, document loaders, text splitters |
| Agent | **LangGraph** (`langgraph`) | Only for the optional online paper search |
| LLM integration | `langchain-ollama` (`ChatOllama`); `langchain-huggingface` (`HuggingFacePipeline`, `ChatHuggingFace`) | Chosen in config |
| Embeddings | `langchain-huggingface` → `HuggingFaceEmbeddings` | |
| Vector store | Qdrant (Docker) via `langchain-qdrant`; **fallback without Docker:** Qdrant local mode (`QdrantClient(path=...)`) inside the project folder | |
| PDF parsing | GROBID (Docker) via LangChain `GrobidParser` + `GenericLoader`; **fallback:** `PyMuPDFLoader` with heading-based section detection | |
| NLI | Hugging Face `transformers` cross-encoder | |
| Clustering | `sklearn.cluster.HDBSCAN` | |
| Quote matching | `rapidfuzz` | |
| DB | PostgreSQL (Docker) or SQLite file in `data/` + SQLAlchemy 2.x | Question 5 |
| API | FastAPI + Pydantic v2 | |
| Frontend | React + Vite + TypeScript (Section 9) | |
| Tests | pytest; Playwright (Python) for UI tests and screenshots | |

Use the latest stable versions; pin exact versions in `requirements.txt` and `frontend/package-lock.json` after the first successful install.

---

## 3. Models (free, from Hugging Face)

| Purpose | Hugging Face model ID | How it runs | Licence (verify on the card) |
|---|---|---|---|
| Main LLM (claim extraction, condition comparison, report) | `Qwen/Qwen2.5-7B-Instruct` | **Default:** Ollama `qwen2.5:7b-instruct` (GGUF build of the same model). **Alternative:** `transformers` 4-bit on a GPU | Apache-2.0 |
| Small LLM (low RAM, topic labels) | `Qwen/Qwen2.5-3B-Instruct` | Ollama `qwen2.5:3b-instruct` | Check the card |
| Embeddings | `BAAI/bge-m3` | `HuggingFaceEmbeddings`, CPU | MIT |
| NLI | `cross-encoder/nli-deberta-v3-base` (base); optionally fine-tuned on SciFact | `transformers`, CPU | Apache-2.0 |

All model IDs live in `configs/models.yaml`. If Ollama is not installed, tell the user in the final message how to install it, and make the app show a clear setup screen instead of failing.

---

## 4. Inputs and datasets

| Input | Where | Notes |
|---|---|---|
| **User papers (main input)** | `research-project/papers/*.pdf` and drag-and-drop upload in the UI | Uploads are copied into `papers/uploads/`. Never modify originals. |
| User research questions | Question 2, or typed in the UI | Optional; topics are discovered automatically otherwise |
| Online papers (optional) | OpenAlex + arXiv | Only if question 3 = yes; cached in `data/cache/` |
| SciFact | Hugging Face `allenai/scifact` (`claims`, `corpus`) | Evaluation and optional NLI fine-tuning |
| SciFact-Open | GitHub `dwadden/scifact-open` | Only if chosen (question 8) |
| BioDivergence | arXiv 2606.11208 data link | Only if chosen and the licence allows |
| Hand labels (later) | `data/labels/` via the built-in labelling screen | Extraction gold and claim-pair labels |

---

## 5. Configuration

`configs/models.yaml`
```yaml
llm:
  provider: ollama            # ollama | hf_local | hf_endpoint
  model: qwen2.5:7b-instruct
  hf_model_id: Qwen/Qwen2.5-7B-Instruct
  temperature: 0.1
  max_retries: 3
llm_small:
  provider: ollama
  model: qwen2.5:3b-instruct
embeddings:
  model_id: BAAI/bge-m3
  device: cpu
nli:
  base_model_id: cross-encoder/nli-deberta-v3-base
  finetuned_path: models/nli-scifact   # used if present, else base model
```

`configs/pipeline.yaml`
```yaml
papers_dir: papers
online_search:
  enabled: false
  max_papers: 50
parsing:
  backend: grobid             # grobid | pymupdf
extraction:
  sections: [abstract, results, discussion, conclusion]
  chunk_chars: 4000
  chunk_overlap: 300
quote_verify:
  fuzzy_threshold: 95
pairs:
  neighbours_per_claim: 10
relations:
  nli_contradict_threshold: 0.6
  nli_support_threshold: 0.6
```

`.env.example`
```
DATABASE_URL=sqlite:///data/claimgraph.db
QDRANT_URL=                      # blank = local mode in data/qdrant
GROBID_URL=http://localhost:8070
OLLAMA_BASE_URL=http://localhost:11434
OPENALEX_MAILTO=
HF_TOKEN=
```

---

## 6. Folder layout (inside `research-project/`)

```text
research-project/
  CLAIMGRAPH_IMPLEMENTATION.md
  README.md
  BUILD_LOG.md
  docker-compose.yml          # postgres, qdrant, grobid (only services the user chose)
  requirements.txt
  .env  .env.example  .gitignore
  configs/                    # models.yaml, pipeline.yaml, user_answers.yaml
  prompts/                    # extract_claims.v1.txt, compare_conditions.v1.txt, label_topic.v1.txt, write_report.v1.txt
  papers/                     # user's PDFs (read-only for the app) + uploads/
  scripts/
    run.py                    # cross-platform entry point (Section 12)
    download_data.py
    make_fixture_papers.py    # builds test PDFs with known claims
    make_results_table.py
  notebooks/
    kaggle_train_nli.ipynb    # optional, for the user
  src/claimgraph/
    config.py  llm.py  db.py  models_db.py
    ingest/       library.py  grobid_loader.py  pymupdf_loader.py  sections.py
    search/       openalex.py  arxiv_client.py  cache.py  search_agent.py      # optional
    claims/       schema.py  extract_chain.py  verify_quote.py
    embed/        embeddings.py  vectorstore.py
    topics/       group.py  label_chain.py  question_filter.py
    relations/    nli_infer.py  compare_chain.py  decide.py
    graph/        build.py  export.py
    report/       report_chain.py  citation_check.py
    pipeline.py
    eval/         scifact_nli_eval.py  scifact_open_eval.py  biodivergence_eval.py  extraction_eval.py  pairs_eval.py  report_eval.py  fixture_eval.py
    api/          main.py  routes_library.py  routes_topics.py  routes_pairs.py  routes_report.py  routes_labels.py  jobs.py  schemas.py
  frontend/
  tests/
    fixtures/                 # generated PDFs + expected claims and relations
    ui/                       # Playwright tests
    test_*.py
  results/                    # metrics JSON + RESULTS.md + ui_screenshots/
  data/                       # git-ignored: db, qdrant, cache, labels
  models/                     # git-ignored
```

---

## 7. Data model

### 7.1 Claim (Pydantic, `claims/schema.py`)

```python
class Conditions(BaseModel):
    dataset: str | None = None
    model: str | None = None        # or population, for biomedical papers
    metric: str | None = None
    setting: str | None = None

class Claim(BaseModel):
    claim_id: str                   # f"{paper_id}_c{n}"  (set by code)
    paper_id: str                   # set by code
    text: str
    intervention: str
    outcome: str
    direction: Literal["increase", "decrease", "no_effect", "mixed"]
    conditions: Conditions
    hedged: bool
    section: str                    # set by code
    page: int | None = None         # set by code
    quote: str                      # exact sentence from the paper
    quote_span: tuple[int, int] | None = None   # set by code
    quote_verified: bool = False                # set by code

class ClaimList(BaseModel):
    claims: list[Claim]
```

### 7.2 Database tables

| Table | Columns |
|---|---|
| `papers` | id, file_path, file_hash, title, authors, year, venue, doi, abstract, source (`user`/`online`), parse_status, n_pages |
| `sections` | id, paper_id, name, text, char_start, page_start |
| `claims` | all `Claim` fields (conditions as JSON) |
| `topics` | id, label, source (`auto`/`question`), question_text |
| `claim_topics` | claim_id, topic_id, score |
| `claim_pairs` | id, topic_id, claim_a, claim_b, nli_label, nli_prob, conditions_match, condition_diff JSON, relation |
| `reports` | id, topic_id, text, cited_claim_ids JSON, prompt_version, model, created_at |
| `jobs` | id, kind, status, progress JSON, error, started_at, finished_at |
| `labels` | id, kind (`extraction`/`pair`), target_id, value JSON, created_at |
| `llm_calls` | id, chain_name, prompt_version, model, ms, ok |

Re-importing the same PDF (same `file_hash`) must not duplicate anything.

---

## 8. Module specifications

### 8.1 `llm.py`
- `get_chat_model(kind="main"|"small")` returns a LangChain chat model from `configs/models.yaml`:
  - `ollama` → `ChatOllama(model=..., temperature=..., base_url=OLLAMA_BASE_URL)`
  - `hf_local` → `ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(...))` with 4-bit quantisation (GPU only; check current `langchain-huggingface` docs for argument names)
  - `hf_endpoint` → `ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id=...))` (optional)
- A LangChain callback handler logs every call to `llm_calls`.

### 8.2 Library import (`ingest/`)
- `library.py`: scan `papers/` (recursive) and `papers/uploads/`; hash each file; import new ones; mark removed files as missing (never delete user files).
- `grobid_loader.py`: `GenericLoader` + `GrobidParser(segment_sentences=True)`; keep section names, page numbers and char offsets.
- `pymupdf_loader.py`: fallback; `PyMuPDFLoader` + heading detection (font size / numbering such as "4 Results", "4.1"); same output format.
- Title, authors, year from GROBID header (or PDF metadata / first page in the fallback).

### 8.3 Claim extraction (`claims/extract_chain.py`)
- LCEL: `ChatPromptTemplate` (from `prompts/extract_claims.v1.txt`) `|` `llm.with_structured_output(ClaimList)`.
- Input: one section chunk (`RecursiveCharacterTextSplitter`) + paper title.
- The prompt requires: only findings this paper reports (not related work), one claim per finding, `quote` copied exactly from the input, `null` for unknown conditions.
- Retry on validation errors; log failures; process papers in a background job with progress.

### 8.4 Quote verification (`claims/verify_quote.py`)
- Exact substring search, then RapidFuzz `partial_ratio` ≥ threshold over sentence windows. Set `quote_verified`, `quote_span`, `page`. **Drop claims that fail** and log the drop rate.

### 8.5 Topics (`topics/`)
- `group.py`: embed `f"{intervention} -> {outcome}"` with bge-m3; HDBSCAN (`min_cluster_size=2`); each cluster becomes an `auto` topic.
- `label_chain.py`: small LLM gives each topic a plain label (≤ 8 words, no marketing words).
- `question_filter.py`: for a user question, retrieve claims by similarity to the question (Qdrant, threshold + top-k) and create a `question` topic.
- Pairs: inside each topic, pair each claim with its `neighbours_per_claim` nearest claims from **other** papers.

### 8.6 Relations (`relations/`)
- `nli_infer.py`: load `models/nli-scifact` if present, else the base model; score both directions and average.
- `compare_chain.py`: field rules first (normalised strings); LLM structured output `{conditions_match, differing_fields, explanation}` only when rules are inconclusive.
- `decide.py` (pure function, fully tested):
```
if nli == entailment and prob >= support_th:            "supports"
elif nli == contradiction and prob >= contradict_th:
    "contradicts" if conditions_match == "same"
    else "different_setting" if conditions_match == "different"
    else "unclear"
elif opposite directions and conditions_match == "different": "different_setting"
else: "unclear"
```

### 8.7 Graph and report
- `graph/build.py`: NetworkX; `export.py`: Cytoscape.js JSON (nodes: claims with paper colour; edges: relation type).
- `report/report_chain.py`: LCEL; input = topic claims and relations as JSON; output = evidence map in plain language; every factual sentence ends with `[claim_id]` citations.
- `report/citation_check.py`: remove sentences with unknown IDs; log them.

### 8.8 Optional online search (`search/`)
- Only active when `online_search.enabled`. OpenAlex and arXiv clients with disk cache; LangGraph agent (`search → rank → decide → expand_references → END`, at most 3 loops, budget `max_papers`). Online papers are marked `source=online` and shown with a small "added from search" label.

### 8.9 API (FastAPI)

| Method + path | Purpose |
|---|---|
| `GET /health` | service status + setup problems (e.g. Ollama missing) |
| `POST /library/scan` | import new PDFs from `papers/` |
| `POST /library/upload` | multipart PDF upload → copied to `papers/uploads/` |
| `GET /library` | papers with parse status, claim counts |
| `GET /papers/{id}` | metadata, sections, claims |
| `GET /papers/{id}/pdf` | stream the PDF (for the viewer) |
| `POST /analyze` | run extraction → topics → relations for all new papers (background job) |
| `GET /jobs/{id}` | progress: step, done/total, current paper |
| `GET /topics` | topics with counts of supports / contradicts / different setting |
| `POST /topics/question` | `{question}` → creates a question topic |
| `GET /topics/{id}/graph` | Cytoscape.js elements |
| `GET /pairs/{id}` | both claims, quotes, pages, NLI result, condition diff, relation |
| `POST /pairs/{id}/label` | user label (used for evaluation) |
| `GET /topics/{id}/report` | generate or fetch the report |
| `GET /search?q=` | search claims and papers |

---

## 9. Frontend: smooth, calm, not "AI-looking"

The app should feel like a **well-made research tool** (think Zotero or a good reading app), not an AI demo.

### 9.1 Do not use
- Purple or blue gradients, glows, neon, glassmorphism, animated backgrounds.
- Sparkle, magic-wand, robot or brain icons; emojis anywhere.
- Words like "AI-powered", "magic", "smart", "insights", "unlock", "Ask AI".
- Chat bubbles or a chatbot as the main interface; typing animations; streaming text effects.
- A marketing-style landing page. The app opens straight into the library.
- Default component-library looks left unstyled; oversized rounded "pill" buttons everywhere; heavy drop shadows.

### 9.2 Visual style
- **Colours (light):** background `#FAFAF7`, surface `#FFFFFF`, text `#1F2328`, secondary text `#5B616B`, borders `#E3E3DE`, one accent `#0F5F5F` (deep teal). Relations: supports `#2F7D4F`, contradicts `#B23A3A`, different setting `#8A8F98` (dashed), unclear `#C9A227` (dotted).
- **Dark theme:** same hues adjusted for contrast; respect `prefers-color-scheme` and the user's default (question 11).
- **Typography:** UI in **Inter** (or IBM Plex Sans); quotes and report text in a serif such as **Source Serif 4**; tabular numbers for counts; base size 14–15 px; line height 1.5 for reading text. Self-host the fonts (no external CDN at runtime).
- **Shape:** 1 px borders instead of shadows, 6 px corner radius, 8 px spacing grid, generous white space.
- **Icons:** Lucide outline icons, 16 px, used sparingly.
- **Colour is never the only signal:** relation edges also differ by line style and carry a text label on hover; badges contain words.

### 9.3 Smoothness
- Transitions 150–200 ms ease-out; respect `prefers-reduced-motion`.
- Skeleton placeholders while loading, never blank screens or large spinners; no layout shift when data arrives.
- TanStack Query for caching and background refresh; optimistic updates for labels.
- Virtualised lists (TanStack Virtual) for long claim and paper lists.
- Graph: Cytoscape.js with the `fcose` layout, animated once on load; smooth zoom and pan; selected node centres gently; graphs with more than 300 nodes start collapsed by paper.
- Keyboard: `/` focus search, `j`/`k` move through lists, `Enter` open, `Esc` close panels, `g` then `l`/`t`/`r` go to Library, Topics, Report.
- Background jobs show a slim progress strip at the top with plain wording: "Reading papers 12 of 40", "Finding claims", "Comparing claims". Short, calm toast messages on completion.
- Works well at 1280 px and wider; usable on a tablet.

### 9.4 Screens
1. **Library:** drop zone ("Drop PDFs here or add them to the papers folder"), paper table (title, year, status, claims), "Analyse new papers" button.
2. **Paper:** left: section outline; centre: PDF viewer (`react-pdf`) with verified quotes highlighted; right: the paper's claims.
3. **Topics:** list of topics with a small stacked bar (supports / contradicts / different setting) and claim count; a field to add a research question.
4. **Topic graph:** the claim graph; clicking an edge opens the pair panel.
5. **Pair view:** two quotes side by side in serif type, each with paper title, year and "open page", plus a condition table with differing fields highlighted and the relation with a one-line reason; buttons to confirm or correct the label.
6. **Report:** clean reading layout (max 70 characters per line), citations as small superscript links that open the claim; export to Markdown and PDF.
7. **Labelling:** fast keyboard-driven screen for claim-pair labels (keys 1–4) and claim checks, saving to `labels`.
8. **Settings / setup:** model, theme, folders; if Ollama or a model is missing, show exactly what to install.

### 9.5 Frontend stack
React + Vite + TypeScript, Tailwind CSS with the tokens above as CSS variables, Radix UI primitives (styled to match), TanStack Query, TanStack Virtual, Cytoscape.js + `cytoscape-fcose`, `react-pdf`, Lucide icons, Framer Motion only for small transitions.

---

## 10. Milestones (build all in order; self-check each)

For every milestone: implement → run `python scripts/run.py test` → fix until green → record in `BUILD_LOG.md` → continue.

### M1 — Setup from the user's answers
- Folder check (Section 0.1), environment detection, questions (Section 0.2), `.venv`, `requirements.txt`, configs, `docker-compose.yml` with only the chosen services, `config.py`, `llm.py`, DB tables, `/health`.
- **Self-check:** tests pass; `/health` reports each service correctly; a test call to the LLM returns valid JSON (skip with a clear note if Ollama is absent).

### M2 — Test fixtures with known answers
- `scripts/make_fixture_papers.py` generates 4 small PDFs (ReportLab) that look like papers (title, abstract, method, results) with **planted claims**: two that support each other, one real contradiction under the same conditions, and one apparent contradiction with a different dataset. Expected claims and relations go to `tests/fixtures/expected.json`.
- **Self-check:** the PDFs open and parse with both GROBID and PyMuPDF backends (whichever are installed).

### M3 — Library import and parsing
- `ingest/` package, `/library/*` endpoints, deduplication by hash.
- **Self-check:** importing `papers/` and the fixtures works; re-import creates no duplicates; user PDFs are untouched (compare hashes before and after).

### M4 — Claim extraction and quote verification
- `claims/` package, prompt v1, background job with progress.
- **Self-check:** on the fixtures, every planted claim is found (`eval/fixture_eval.py`), and 100% of stored claims have `quote_verified=true`. Then run on the user's papers and log claims per paper and the drop rate.

### M5 — Topics and relations
- `topics/`, `relations/` (base NLI if no fine-tuned model), question topics.
- **Self-check:** on the fixtures the expected relations come out: supports, contradicts and **different setting** for the planted pairs. `decide.py` has full branch coverage.

### M6 — Graph, report, API
- `graph/`, `report/`, all remaining endpoints, citation check.
- **Self-check:** every report citation resolves; API tests for every endpoint.

### M7 — Frontend
- All screens in Section 9 with the specified style.
- **Self-check with Playwright:** start the app, upload a fixture PDF, run the analysis, open a topic graph, open a pair, open the report, use keyboard shortcuts. Save screenshots of every screen (light and dark) to `results/ui_screenshots/`. Review them against Section 9.1 and 9.2 (no gradients, no AI wording, consistent spacing, nothing cut off or overlapping) and fix any problems found, then re-run.

### M8 — Evaluation and optional extras
- `scripts/download_data.py` for the datasets the user chose; `eval/scifact_nli_eval.py` (+ others if chosen); `notebooks/kaggle_train_nli.ipynb` ready for the user; optional online search if enabled.
- `scripts/make_results_table.py` builds `results/RESULTS.md` from JSON only.
- **Self-check:** `python scripts/run.py eval` completes; `RESULTS.md` lists each metric with the run date; evaluations that need user labels are listed as "waiting for labels", not filled with numbers.

---

## 11. Testing requirements

- Unit tests: library dedup, both PDF loaders on fixtures, schema validation, quote verification, pair generation, `decide`, citation check, config loading.
- Chain tests: each LCEL chain returns valid Pydantic objects on fixture inputs (marked `llm`; skipped with a message if Ollama is absent).
- End-to-end: `tests/test_pipeline_fixtures.py` runs the whole pipeline on the fixtures and compares with `expected.json`.
- UI: Playwright tests in `tests/ui/` (Section 10, M7).
- `python scripts/run.py test` runs everything that can run on this machine and prints a summary.

---

## 12. Commands (`scripts/run.py`, cross-platform)

```
python scripts/run.py setup     # create .venv, install Python and npm packages
python scripts/run.py up        # start Docker services (if chosen)
python scripts/run.py start     # start API + frontend and open the browser  <- the one command for the user
python scripts/run.py test      # all tests
python scripts/run.py eval      # evaluations -> results/
python scripts/run.py fixtures  # rebuild test PDFs
```

`start` must work from a fresh terminal in `research-project` and print the local URL.

---

## 13. Definition of done

- `python scripts/run.py start` opens the app; the user's papers from `papers/` appear in the Library and can be analysed end to end: claims with verified quotes, topics, a graph, pair views and a cited report.
- All tests pass (`run.py test`), including the fixture end-to-end test with the planted contradiction and the planted "different setting" case.
- Playwright screenshots of all screens exist and follow Section 9.
- `results/RESULTS.md` exists and contains only measured numbers.
- Nothing outside `research-project` was changed (except allowed caches, Section 0.1).
- `README.md` explains: starting the app, adding papers, optional steps (fine-tuning notebook, labelling, extra datasets, online search), model and dataset licences, and limitations.
- The final message to the user follows Section 0.3.
