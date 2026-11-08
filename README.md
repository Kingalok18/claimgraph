# ClaimGraph

ClaimGraph reads your research papers and maps what they claim. It shows which findings agree, which truly conflict, and which only look like a conflict because the studies were run under different conditions. Every claim keeps an exact quote from its paper, and every sentence of a report cites the claims behind it.

Everything runs on your own computer with free models. Your papers are not sent anywhere, and no paid API is used.

---

## Contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Tech stack](#tech-stack)
4. [Requirements](#requirements)
5. [Install](#install)
6. [Run the app](#run-the-app)
7. [Using the app](#using-the-app)
8. [Where data is stored](#where-data-is-stored)
9. [Database tables](#database-tables)
10. [Configuration](#configuration)
11. [Commands](#commands)
12. [Tests](#tests)
13. [Evaluation and results](#evaluation-and-results)
14. [API](#api)
15. [Project structure](#project-structure)
16. [Optional steps](#optional-steps)
17. [Troubleshooting](#troubleshooting)
18. [Limitations](#limitations)
19. [Licences](#licences)

---

## What it does

- **Reads PDFs** from a folder and splits them into sections (abstract, results, discussion, conclusion).
- **Extracts claims** with a local language model: what was tested, what changed, in which direction, and under which conditions (dataset, model or population, metric, setting).
- **Checks every quote.** A claim is kept only if its quote is found in the paper's text. Claims that fail the check are dropped.
- **Groups claims into topics** by meaning.
- **Compares claims from different papers** in the same topic and labels each pair **supports**, **contradicts**, **different setting** or **unclear**.
- **Writes a report** for each topic. Every sentence must cite real claim IDs, and sentences the claims don't back are removed.
- **Shows it all in a web interface:** library, PDF viewer with highlighted quotes, topic graph, side-by-side pair view, report, and a labelling screen for measuring accuracy.

## How it works

```
PDF files ─► Parse (PyMuPDF) ─► Sections ─► Chunks ─► LLM extracts claims (JSON, grammar-constrained)
                                                              │
                                                     Quote check (exact, then fuzzy)
                                                              │
                                                   Verified claims ─► MySQL
                                                              │
                                     Embed "intervention -> outcome" (bge-m3) ─► Qdrant index
                                                              │
                                          Cluster into topics (HDBSCAN + merge + attach)
                                                              │
                                  Pairs: 10 nearest claims from other papers in the same topic
                                                              │
                         NLI model scores both directions ─► conditions compared (rules first, LLM if needed)
                                                              │
                                     Decision rule ─► supports / contradicts / different setting / unclear
                                                              │
                                    Report: LLM writes, citations checked, ungrounded sentences removed
```

**The decision rule** (`src/claimgraph/relations/decide.py`):

- Entailment ≥ 0.6 → **supports**.
- Contradiction ≥ 0.6 → **contradicts** if the conditions are comparable, **different setting** if they differ, **unclear** if that can't be told.
- Opposite directions with different conditions → **different setting**.
- Anything else → **unclear**.

**Long runs are resumable.** Each finished passage is committed, so after a stop, crash or restart the next run continues where it left off. Relations already worked out are cached, so regrouping topics doesn't redo model work.

## Tech stack

| Part | Choice |
|---|---|
| Language model | Qwen2.5-3B-Instruct, GGUF `q4_k_m`, run by `llama-cpp-python` with JSON grammar |
| Embeddings | `BAAI/bge-m3` via `sentence-transformers` |
| Comparison (NLI) | `cross-encoder/nli-deberta-v3-base` (or your fine-tuned copy in `models/nli-scifact`) |
| PDF parsing | PyMuPDF (GROBID optional) |
| Orchestration | LangChain runnables; LangGraph for the optional online search |
| Clustering | HDBSCAN |
| Database | MySQL 8 via SQLAlchemy (SQLite also works) |
| Vector index | Qdrant, local mode (server optional) |
| Backend | FastAPI + Uvicorn, one background job worker |
| Frontend | React 18, TypeScript, Vite, Tailwind, TanStack Query, Cytoscape.js, react-pdf |
| Tests | pytest, Playwright |

## Requirements

- **Windows, macOS or Linux.** Built and tested on Windows 11.
- **Python 3.11.**
- **Node.js 20 or newer.** Only needed to build the interface.
- **MySQL 8.0.** Or use SQLite with no server; see [Configuration](#configuration).
- **8 GB of RAM** at minimum (16 GB is more comfortable) and about **10 GB of free disk** for packages and models.
- **Internet once,** to download packages and models. After that the app runs offline.

## Install

### 1. Get the code

```
git clone https://github.com/kingalok18/claimgraph.git
cd claimgraph
```

Repository: **https://github.com/kingalok18/claimgraph**

### 2. Create the MySQL databases

Open the MySQL command line (or MySQL Workbench) as root and run:

```sql
CREATE DATABASE claimgraph CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE claimgraph_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'claimgraph'@'127.0.0.1' IDENTIFIED BY 'choose-a-password';
GRANT ALL PRIVILEGES ON claimgraph.* TO 'claimgraph'@'127.0.0.1';
GRANT ALL PRIVILEGES ON claimgraph_test.* TO 'claimgraph'@'127.0.0.1';
```

The app creates its tables by itself on first start. `claimgraph_test` is used only by the tests.

### 3. Create `.env`

Copy `.env.example` to `.env`, then replace `CHANGE_ME` with the password you chose:

```
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

`.env` is git-ignored. Never commit it.

### 4. Install everything

```
python scripts/run.py setup
```

This step:
- creates the virtual environment `.venv` and installs the Python packages;
- installs the Playwright browser into `.venv/playwright`;
- installs the npm packages and builds the interface into `frontend/dist`;
- creates the test PDFs and the database tables;
- downloads the three models from Hugging Face (about 5 GB, once).

### 5. Add your papers

Create a folder named `paper` in the project and copy your PDFs into it. You can also drop PDFs into the app later.

## Run the app

### 1. Activate the virtual environment

Open a terminal in the project folder (in VS Code: **Terminal → New Terminal**, or `` Ctrl+` ``), then run the line that matches your terminal:

```
.venv\Scripts\activate.bat        # Windows Command Prompt (prompt looks like C:\...>)
.venv\Scripts\Activate.ps1        # Windows PowerShell (prompt starts with PS)
source .venv/bin/activate         # macOS / Linux
```

`(.venv)` now shows at the start of the prompt.
- If PowerShell says *running scripts is disabled*, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. It only affects that terminal window.
- If `Activate.ps1` seems to do nothing, the terminal is Command Prompt, so use `activate.bat`.
- To leave the environment later, type `deactivate`.

### 2. Start

```
python scripts/run.py start
```

- The models load for 30–60 seconds, then the app opens **http://127.0.0.1:8000** in your browser.
- Keep the terminal open while you use the app. Press **Ctrl+C** in it to stop.
- MySQL must be running. On Windows that's the **MySQL80** service.

### VS Code tip

Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick `.venv\Scripts\python.exe`. This gives you code completion, the Testing panel, and new terminals that activate `.venv` automatically.

## Using the app

### Add papers

- **Drop PDFs** onto the Library screen. They are copied into `paper/uploads`.
- Or **copy PDFs** into the `paper` folder and press **Check folder**. The folder is also checked every time the app starts.

The Library always matches the `paper` folder:
- **Your PDFs are only ever read,** never changed.
- **If you delete or move a PDF out of the folder,** its paper leaves the Library on the next check, together with its claims. Topics are regrouped on the next analysis, and your labels are kept.
- **A file put back** is added again as a new paper.
- **The same file is never added twice,** even under another name, because files are matched by SHA-256 hash.
- **Scanned PDFs with no text layer can't be read.** Run OCR on them first.

### Analyse

| Button | What it does |
|---|---|
| **Analyse** (on a paper's row) | Analyses only that paper. Useful right after adding one. |
| **Analyse 5** | Analyses the 5 oldest new papers. Shown when more than 5 are waiting. |
| **Analyse new papers** | Analyses every paper not analysed yet. |
| **Stop** (in the progress strip) | Stops safely. The next run continues where it left off. |

On a laptop CPU one paper takes about 10–15 minutes, so a whole library is best left overnight. Close other programs, keep the computer plugged in, and stop it from sleeping, because sleep pauses the work. Only one analysis runs at a time.

### The screens

| Screen | What it does |
|---|---|
| Library | Your papers, with status, claim counts and Analyse buttons |
| Paper | The PDF with verified quotes highlighted, its outline, and its claims |
| Topics | Groups of related claims, with a bar showing how they relate; add your own research question here |
| Topic graph | Claims as dots and relations as lines; select a line to compare its two claims |
| Pair view | Two quotes side by side with their conditions compared; confirm or correct the relation here |
| Report | An evidence map with citations, exported as Markdown or PDF |
| Labelling | Fast keyboard checking of claims and pairs, which is what accuracy is measured against |
| Settings | Theme, folders, and the status of each model and service, with what to do if something is missing |

**Keys:**

| Key | Action |
|---|---|
| `/` | Search |
| `j` / `k` | Move through a list |
| `Enter` | Open |
| `Esc` | Close |
| `g` then `l`, `t` or `r` | Go to Library, Topics or Report |
| `1` to `4` | Label a pair |

## Where data is stored

| What | Where | In git? |
|---|---|---|
| Your PDFs | `paper/` (uploads in `paper/uploads/`) | No |
| Papers set aside for later | `papers_not_analysed/` (not read by the app) | No |
| Papers, sections, claims, topics, pairs, reports, jobs, labels, model-call log | MySQL database `claimgraph` (the source of truth) | No |
| Claim vectors (search index) | `data/qdrant/` (rebuilt from MySQL if deleted) | No |
| Labels, also as plain files | `data/labels/extraction_labels.jsonl`, `data/labels/pair_labels.jsonl` | No |
| Extraction log (claims kept and dropped per paper) | `data/logs/extraction.jsonl` | No |
| Evaluation datasets (SciFact and others) | `data/datasets/` | No |
| Cache | `data/cache/` | No |
| Downloaded models | Hugging Face cache: `C:\Users\<you>\.cache\huggingface\` (`~/.cache/huggingface/` on macOS/Linux) | No |
| Fine-tuned comparison model | `models/nli-scifact/` (optional) | No |
| Database password and keys | `.env` | No |
| Python environment and Playwright browser | `.venv/` | No |
| Built interface | `frontend/dist/` (rebuilt by `setup`) | No |
| Measured results | `results/*.json`, `results/RESULTS.md` | Yes |
| Screenshots | `results/ui_screenshots/` | Yes |
| Settings | `configs/*.yaml` | Yes |
| Prompts | `prompts/*.txt` (versioned, `.v1`) | Yes |

To start over with an empty library, stop the app, drop and recreate the `claimgraph` database, and delete `data/qdrant`. Your PDFs aren't affected.

## Database tables

The app creates ten tables in MySQL:

| Table | Holds |
|---|---|
| `papers` | One row per PDF: file path, SHA-256 hash, title, authors, year, parse and analysis status, resume position |
| `sections` | Each paper's sections with canonical names, headings, pages and character offsets |
| `claims` | Extracted claims: intervention, outcome, direction, conditions (JSON), section, page, exact quote, quote position, verified flag |
| `topics` | Topic groups: key (from member claims), label, source (`auto` or your question) |
| `claim_topics` | Which claims belong to which topic, with a similarity score |
| `claim_pairs` | Compared pairs: NLI label and probability, condition comparison, final relation |
| `reports` | Generated reports with cited claim IDs and removed sentences |
| `jobs` | Background jobs (scan, analysis) with progress and errors |
| `labels` | Your labels for claims and pairs |
| `llm_calls` | Every model call: step, prompt version, model, time taken, success |

Claim IDs look like `p1a2b3c4d5e_c3`: the paper ID (`p` plus the first 10 characters of the file hash) and the claim number. Qdrant point IDs are UUID5s of the claim ID, so re-indexing gives the same IDs.

## Configuration

### `.env`

| Variable | Use |
|---|---|
| `DATABASE_URL` | Main database. MySQL: `mysql+pymysql://user:password@127.0.0.1:3306/claimgraph?charset=utf8mb4`. No server: `sqlite:///data/claimgraph.db` |
| `TEST_MYSQL_URL` | The `claimgraph_test` database for the MySQL tests. If blank, those tests are skipped. |
| `QDRANT_URL` | Blank uses local mode in `data/qdrant`. Set it to use a Qdrant server. |
| `GROBID_URL` | Only used if `parsing.backend` is `grobid` |
| `OLLAMA_BASE_URL` | Only used if the LLM provider is `ollama` |
| `OPENALEX_MAILTO` | Email address for the optional online search |
| `HF_TOKEN` | Optional; the default models don't need it |
| `CLAIMGRAPH_PAPERS_DIR`, `CLAIMGRAPH_DATA_DIR` | Optional; move the papers folder or the data folder |

### `configs/models.yaml`

Which models are used and how: the LLM file and provider (`hf_gguf`, `ollama`, `hf_local`, `hf_endpoint`), temperature, context size, CPU threads (`n_threads`, set it to your physical core count), the embedding model, and the NLI model.

### `configs/pipeline.yaml`

| Setting | Default | Meaning |
|---|---|---|
| `papers_dir` | `paper` | Folder that is scanned |
| `scan_on_start` | `true` | Check the folder when the app starts |
| `parsing.backend` | `pymupdf` | Or `grobid` |
| `extraction.max_chunks_per_paper` | 12 | Passages read per paper; more is slower |
| `extraction.chunk_chars` / `chunk_overlap` | 4000 / 300 | Passage size |
| `quote_verify.fuzzy_threshold` | 95 | How close a quote must be to the paper text |
| `topics.min_cluster_size` | 2 | Smallest topic |
| `topics.merge_similarity` | 0.85 | Clusters this similar become one topic |
| `topics.attach_noise_similarity` | 0.60 | Unclustered claims join the nearest topic above this |
| `pairs.neighbours_per_claim` | 10 | Pairs made per claim |
| `relations.nli_support_threshold` / `nli_contradict_threshold` | 0.6 / 0.6 | Decision thresholds |
| `relations.max_llm_compares` | 200 | Cap on slow LLM condition checks per run |
| `online_search.enabled` | `false` | Optional OpenAlex and arXiv search |

Restart the app after changing a setting.

## Commands

Run these with the virtual environment active:

```
python scripts/run.py setup     # install everything and download models
python scripts/run.py start     # start the app
python scripts/run.py test      # run all tests and print a summary
python scripts/run.py eval      # run evaluations and write results/
python scripts/run.py fixtures  # rebuild the test PDFs
python scripts/run.py up        # nothing is needed on this install
```

Re-run the logged sample of your papers and check that the PDFs are unchanged (slow, and not part of `eval`):

```
PYTHONPATH=src .venv/Scripts/python.exe -m claimgraph.eval.user_papers_eval
```

Rebuild only the interface after changing `frontend/src`:

```
cd frontend
npm run build
```

## Tests

```
python scripts/run.py test
```

- **What's covered:** parsing, quote checking, claim schema, clustering, pairing, the decision rule, report grounding, the API, the database (SQLite and MySQL), the full pipeline on four planted test papers with the real models, and browser tests of every screen with Playwright.
- **Data:** tests use temporary folders, temporary SQLite files and the `claimgraph_test` database. They never touch your library or your `claimgraph` database.
- **Time and memory:** the full run takes about 12 minutes on a laptop CPU and needs the memory for the real models. Close other programs first. On a low-memory machine, run it from a normal terminal rather than inside an editor.

Run one file:

```
python -m pytest tests/test_relations.py -q
```

## Evaluation and results

```
python scripts/run.py eval
```

- Each script in `src/claimgraph/eval/` writes one file in `results/`, and `results/RESULTS.md` is the summary table. No number is written by hand.
- **Planted test papers** (`fixture_eval`): checks the whole pipeline on four PDFs with known relations.
- **SciFact NLI** (`scifact_nli_eval`): accuracy of the comparison model on SciFact.
- **Extraction precision and pair accuracy on your papers:** reported once you've labelled at least 20 claims and 20 pairs on the Labelling screen. Until then they're listed as waiting, with no numbers.

See `results/RESULTS.md` for what has been measured, and `BUILD_LOG.md` for how the project was built and what was decided along the way.

## API

The backend is a FastAPI app. While the app runs, interactive API docs are at **http://127.0.0.1:8000/docs**.

| Method | Path | Does |
|---|---|---|
| GET | `/health` | Status of the database, models and services |
| GET | `/library` | All papers with status and claim counts |
| POST | `/library/scan` | Check the papers folder |
| POST | `/library/upload` | Add a PDF |
| GET | `/papers/{id}`, `/papers/{id}/pdf` | One paper with its claims; the PDF file |
| POST | `/analyze` | Start an analysis: `{}` for all new papers, `{"limit": 5}`, or `{"paper_ids": ["p..."]}` |
| GET | `/jobs/active`, `/jobs/{id}` | Progress of a job |
| POST | `/jobs/{id}/cancel` | Stop a job |
| GET | `/topics`, `/topics/{id}/graph`, `/topics/{id}/pairs` | Topics, the graph data, the pairs |
| POST | `/topics/question` | Add your own research question as a topic |
| GET | `/pairs/{id}` | One pair with both claims and the condition comparison |
| POST | `/pairs/{id}/label` | Correct a pair's relation |
| GET | `/topics/{id}/report`, `.md`, `.pdf` | The report, as JSON, Markdown or PDF |
| GET | `/search?q=` | Search claims and papers |
| GET | `/labels/stats`, `/labels/queue` | Labelling progress and the next items |
| POST | `/labels/claim` | Label a claim |
| POST | `/search/online` | Optional online search (off by default) |

## Project structure

```
configs/          models.yaml, pipeline.yaml, user_answers.yaml
prompts/          the prompts, versioned
src/claimgraph/
  ingest/         PDF finding, parsing (PyMuPDF, GROBID), sections, library scan
  claims/         extraction chain, claim schema, quote check
  embed/          embeddings and the Qdrant index
  topics/         clustering, pairing, topic labels, research-question filter
  relations/      NLI, condition comparison, decision rule
  report/         report writing, citation check, grounding check
  graph/          graph data and export
  search/         optional online search (OpenAlex, arXiv, LangGraph agent)
  eval/           evaluation scripts that write results/
  api/            FastAPI routes and the background job worker
  pipeline.py     the analysis steps in order
  llm.py          the local LLM wrapper
  models_db.py    database tables
  config.py       settings from .env and configs/
frontend/src/     the React interface
tests/            pytest tests, test fixtures, browser tests (tests/ui)
scripts/          run.py and helper scripts
notebooks/        Kaggle notebook for fine-tuning the comparison model
results/          measured results, RESULTS.md, screenshots
docker-compose.yml  optional GROBID and Qdrant servers
```

## Optional steps

- **Fine-tune the comparison model on Kaggle.**
  1. Upload `notebooks/kaggle_train_nli.ipynb` to Kaggle.
  2. Turn on the T4 GPU and internet, then choose **Save & Run All**.
  3. Download `nli-scifact.zip` from the Output tab and unzip it so that `models/nli-scifact/config.json` exists.
  4. Restart the app, and it uses the new model. Run `python scripts/run.py eval` to measure it.
- **Label claims and pairs.** Use the Labelling screen. Once 20 of each are labelled, `eval` reports real precision and accuracy for your papers.
- **Extra datasets.** `python scripts/download_data.py --scifact-open` fetches SciFact-Open. For BioDivergence, check its licence, then place claim pairs at `data/datasets/biodivergence/pairs.jsonl` with the fields `claim_a`, `claim_b` and `relation`.
- **Online search.** Set `online_search.enabled: true` in `configs/pipeline.yaml` and add `OPENALEX_MAILTO` in `.env`. Then `POST /search/online` with `{"query": "..."}` adds papers from OpenAlex and arXiv, labelled "added from search". This sends your search words to those services. It has no button in the interface yet.
- **A larger model.** With 16 GB of RAM or more, set `hf_model_id` and `gguf_file` in `configs/models.yaml` to `Qwen/Qwen2.5-7B-Instruct-GGUF` and its file name. It should extract better, and it runs more slowly.
- **Docker services.** `docker compose --profile optional up -d` starts GROBID and a Qdrant server. They aren't needed by default.

## Troubleshooting

| Problem | Fix |
|---|---|
| `The environment is not set up yet` | Run `python scripts/run.py setup` |
| `npm was not found` | Install Node.js 20 or newer, then run `setup` again |
| Database connection error | Start MySQL (Windows: Services → MySQL80 → Start) and check `DATABASE_URL` in `.env` |
| `Activate.ps1` cannot be loaded | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or use `activate.bat` in Command Prompt |
| `Activate.ps1` does nothing in VS Code | The terminal is Command Prompt; use `.venv\Scripts\activate.bat` |
| Port 8000 already in use | The app is already running in another terminal; stop it with Ctrl+C there |
| Analysis is very slow or the PC freezes | Close other programs; lower `max_chunks_per_paper`; keep the PC plugged in and awake |
| A test run is killed partway | Not enough memory; close other programs and run it from a normal terminal |
| `The frontend is not built` | Run `npm ci` and `npm run build` in `frontend`, or run `setup` again |
| Paper shows "Could not read" | It's a scanned PDF with no text layer, or a damaged file; run OCR first |
| A model is missing in Settings | Run `setup` again to download it |

## Limitations

- **A small model makes mistakes.** It misses findings, and it can fill a field such as "method tested" wrongly. The quote check proves that a sentence is in the paper, not that the claim summarises it correctly. Check important claims against the highlighted quote, and use the Labelling screen to measure accuracy.
- **Procedure and table filters are simple rules.** Sentences like "We evaluate on X" and table rows are dropped by pattern, so an occasional one gets through and an occasional real finding is lost.
- **Tables and figures are not read as data.** Results that appear only in a table are usually not extracted.
- **Long papers are not fully read.** At most 12 passages per paper are analysed, abstract and conclusion first.
- **PDF layout varies.** PyMuPDF can misjudge headings in unusual layouts, and text can pick up small errors such as joined words at line breaks. A quote that spans two pages is highlighted only where its pieces can be found.
- **The comparison model is general purpose until it is fine-tuned,** and it is weak on scientific wording. Expect many "unclear" pairs and few "supports" or "contradicts".
- **"Different setting" can join unrelated claims.** When the comparison model sees opposing statements and the conditions differ, the pair is labelled "different setting" even if the claims aren't really about the same thing.
- **Condition extraction is noisy.** A field can be filled with something vague, such as "model", so a condition comparison can be wrong. The pair view shows both values.
- **Runs vary.** The language model is not seeded, so analysing a paper again can find a slightly different set of claims.
- **Reports ignore pair corrections for now.** A relation you correct in the pair view updates the graph and topic counts, but the report is still written from the model's relation.
- **Reports on topics without supports or contradicts pairs are plain.** They are written from stored data without the language model.
- **A topic can have no pairs.** Pairs compare claims from different papers, so a topic whose claims all come from one paper has nothing to compare.
- **Topics change as papers are added or removed.** A topic is rebuilt when its claims change, so its name and report may be regenerated. Your labels are kept.

## Licences

Check each model's card before relying on these, especially for anything beyond personal research.

| Item | Licence |
|---|---|
| `Qwen/Qwen2.5-3B-Instruct-GGUF` | Custom Qwen licence, shown as "other" on Hugging Face; read it before any commercial use |
| `BAAI/bge-m3` | MIT |
| `cross-encoder/nli-deberta-v3-base` | Apache-2.0 |
| SciFact | Check the terms in its release; used here only for evaluation |
| PyMuPDF | AGPL-3.0 or a commercial licence; this matters if you distribute the app |
