# ClaimGraph build log

One entry per milestone. Measured results live in `results/*.json` and `results/RESULTS.md`.
Dates are from this PC's clock.

## Machine

Windows 11, Intel i5-10210U (4 cores, 8 threads), 7.8 GB RAM, MX250 with 2 GB. Python 3.11 in `.venv`.
Memory is tight: 19–24 GB committed on 7.8 GB of RAM. Claude Code's low-memory safety stop ended several
long background runs during the build. None lost any work: extraction commits after every passage, and the evaluations use temporary storage.

## M1: Setup (2026-11-06)

Built `.venv`, `requirements.txt`, configs, `config.py`, `llm.py`, database tables and `/health`.
Self-check: tests pass, `/health` reports each service, and a real call returns valid JSON.

Decisions:
- **Database:** the user's MySQL as `root`; credentials are in the git-ignored `.env`. Tests use only `claimgraph_test`.
- **Language model:** a free Hugging Face GGUF file (Qwen2.5 3B Instruct, Q4_K_M) through `llama-cpp-python`, as the user asked, instead of the spec's Ollama. Provider name is `hf_gguf`. Ollama remains selectable in `configs/models.yaml`.
- **Model choice was measured:** on one paper, the 3B model used about 1.1–1.6 GB at peak and took 48–213 s per passage. It was kept over the 1.5B model, which took 66–116 s per passage and was not faster in this test.
- **Sequential models:** one heavy model is in memory at a time, and each is released between steps.
- **Folder name:** the user's folder stays `paper` rather than the spec's `papers`.

## M2: Test fixtures (2026-11-06)

`scripts/make_fixture_papers.py` writes 4 PDFs with planted claims, plus `tests/fixtures/expected.json`.
Self-check: the PDFs parse with PyMuPDF. GROBID was not used (the user chose PyMuPDF; Docker is not running), so its loader is tested with stand-ins only.

## M3: Library import and parsing (2026-11-06)

`ingest/` and the `/library/*` endpoints, with de-duplication by hash.
Self-check: re-import makes no duplicates, and PDF hashes match before and after.

Decisions:
- PyMuPDF's line-level API is used for the body text, because LangChain's loader does not expose the font sizes that heading detection needs. The loader still supplies metadata.
- Extra columns beyond the spec: page offsets, parse errors, extraction progress, dropped-claim counts, a missing-file flag, section headings, and stable keys for topics and pairs. These allow resuming and repeatable runs.

## M4: Claim extraction and quote verification (2026-11-06 to 2026-11-07)

`claims/`, prompt v1, background job with progress, committed after every chunk so a stopped run resumes.
Self-check on real models: 4 of 4 planted claims found, every stored quote verified (`results/fixture_eval.json`).

Decisions:
- `ExtractedClaim` is what the model may fill. Code owns the ids, spans, page and the verified flag, so the model cannot claim its own quote was checked.
- `quote` is the first schema field, so the model has chosen its evidence before it writes the claim.
- At most 12 chunks per paper (`max_chunks_per_paper`), most informative sections first, because of CPU speed.

Bugs found by real-model runs and fixed:
- The model returned empty claim lists. A grammar change alone did not fix this; rewriting the prompt to define a result did.
- Table rows passed as "exact" quotes. They are now rejected by `looks_like_prose`.
- Method sentences were stored as findings. They are now rejected by `is_procedure`.
- A fuzzy match could change a number. Quotes with different numbers are now rejected.
- Section headings were glued to quotes ("3 Results Adding dropout ..."). This also carried non-findings past the procedure filter. Line breaks now end a sentence, and copied headings are stripped.

Bugs found on the user's papers and fixed:
- Captions, "In this letter, we propose ..." and "The dataset used ... is provided" were stored as findings. `is_procedure` now covers them.
- Quotes were cut mid-sentence or began with a bullet. `whole_sentence` now completes a quote from the whole paper, not just its passage, and strips bullets. A quote that still starts in lower case is dropped by `is_fragment`.

The sample of 5 papers is hand-picked, not random; the run is logged in `results/user_papers.json`.

## M5: Topics and relations (2026-11-06 to 2026-11-07)

`topics/`, `relations/` with the base NLI model, and question topics.
Self-check on real models: 5 of 5 planted relations correct (`results/fixture_eval.json`). `decide.py` is tested on every branch.

Decisions:
- `topics.merge_similarity: 0.85`. With `min_cluster_size=2`, HDBSCAN split near-identical claims into arbitrary pairs, so the planted pairs were never compared. The first real run scored 2 of 5 for this reason. Close clusters are now merged.
- The language-model conditions check is skipped when its answer cannot change the relation, and it is capped by `max_llm_compares`.

## M6: Graph, report, API (2026-11-06)

`graph/`, `report/` and all endpoints. The report checks every citation and falls back to a template built from stored relations when the model's text does not pass.
Self-check: citation tests and API tests for every endpoint pass.

Fix: citations separated from their sentence let invented ids survive the check. `_sentences` now keeps them together.

Addition beyond the spec's citation check: on the user's papers the model wrote conflicts and agreements that no stored pair supported, using valid citations, so the citation check could not catch them. It also repeated lines and added details. `report/grounding.py` now keeps a sentence only if a stored pair with that section's relation backs it, drops numbers not in the cited claims, and writes the settings sentences from stored condition values. Stored relations the model skipped are added. When a topic has no supports or contradicts pairs, the model is skipped, so those reports are plain stored-data wording.

Fix: the pair view's reason could disagree with its condition table, because it showed the model's free text. `reason_for` now builds that line from the stored comparison, and the model's text is kept only for the record.

## M7: Frontend (2026-11-07)

All screens from Section 9, served by FastAPI on one port with a hash router, so that `start` is a single process.
Self-check with Playwright: the full journey (upload, analyse, graph, pair, report, keyboard), both themes, and tablet width. Screenshots are in `results/ui_screenshots/`, and I reviewed them against Sections 9.1 and 9.2.

The browser tests use stand-in models so that they take seconds rather than hours. The real-model path is covered by `tests/test_pipeline_fixtures.py` and by the run on the user's papers.

Problems found in the review and fixed:
- The search icon overlapped the text. `.input` was overriding Tailwind utilities, so it moved into the components layer.
- Graph labels were huge on small graphs. The first fit now stops near actual size, and the reader can still zoom in.
- The empty table box in the Library now shrinks to its rows.
- The report's back link had no arrow, and nothing explained the node colours.

The PDF highlight worked line by line, so it marked only part of a multi-line quote, or a whole line that held the end of another sentence. It now matches the quote across the page and marks only those characters. Checked on a real paper, where the entire quote was marked. Screenshots of the user's papers are `results/ui_screenshots/real_*.png`.

One intermittent failure: the journey test failed once in its first 10 runs, when the pair panel showed no differing row. I could not reproduce it. The likely cause is the click landing where edges cross, since the layout is random, so the test now clicks the clearest point on the edge. This cause is not proven.

## M8: Evaluation and optional extras (2026-11-07)

`scripts/download_data.py`, `eval/` scripts, `notebooks/kaggle_train_nli.ipynb`, online search (built, switched off), and `make_results_table.py`.
Added `eval/user_papers_eval.py`, which logs the sample run required by M4 and compares PDF hashes.
`scifact_nli_eval.py` reports two numbers: both directions averaged, as the pipeline scores pairs, and one way, evidence to claim, which is the question SciFact asks. `fixture_eval.py` now also writes a real-model report and checks that each planted relation sits under the right heading.
Results are in `results/RESULTS.md`.

Finding: on SciFact, averaging both directions scored lower than scoring one way (see `results/RESULTS.md`). The pipeline still averages both directions, as the spec asks, because a claim pair has no natural direction. Whether scoring one way would find more relations on the user's papers has not been measured.

Bug in the test setup, found at the end of the build: the browser test's helper was named `serve_for_test.py`, which matches pytest's `*_test.py` pattern. A full `pytest tests` imported it during collection, and importing it replaces the real models with stand-ins. So every real-model pipeline test in a full run silently used stand-ins, while running those tests alone used the real models. Noticed because a full run took 4.5 minutes when the real-model tests alone took over 10, then confirmed by counting model calls. The helper is now `stand_in_server.py`. The real-model tests now assert that the real models were called, and a fast test checks that collection leaves them in place. Evaluations run outside pytest, so `results/fixture_eval.json` was not affected.

Final check after that fix: `python scripts/run.py test` gave 236 passed, 0 failed, 0 skipped in 719.72 s (output in `results/final_test_output.txt`). The user ran it in their own PowerShell window, because Claude Code's low-memory stop ended the run started from the session. `python scripts/run.py start` was checked on the user's papers before the rename; the rename touched only test files.

Cleanup at the user's request: removed Python caches, `frontend/node_modules` (`start` only needs `frontend/dist`), the unused 1.5B model from the Hugging Face cache, and the Ollama copy of the 3B model. Kept the optional GROBID and Docker files, because the spec asks for them.

## Waiting for the user

- **Fine-tuning:** run `notebooks/kaggle_train_nli.ipynb` on Kaggle. It was written but has not been executed.
- **Hand labels:** the Labelling screen collects them. The extraction and pair accuracy evaluations stay "waiting for labels" until then.
- **Extra datasets:** SciFact-Open and BioDivergence were not chosen at setup, so those evaluations are waiting for data.
- **The full library:** the first run covers a sample of 5 of the 46 papers. The rest can be started from the Library screen.
- **Online search:** built and switched off, as chosen at setup.

## 2026-11-08: Analyse one paper from the Library

- Asked for: analyse a newly added paper from the main page. Before this, "Analyse 5" took the 5 oldest unanalysed papers, so a new paper could only be analysed alone through `POST /analyze` with `paper_ids`.
- Added an **Analyse** button on each Library row whose paper is read and not yet analysed. It sends `{"paper_ids": [id]}`. It is disabled while an analysis job runs, because `jobs.submit` returns the running job instead of queueing a second one.
- Frontend packages were reinstalled (`npm ci`) to rebuild; `frontend/node_modules` is present again.
- Checked: the build (with the TypeScript check) passed; the live Library showed the button on unanalysed rows only, with no browser errors; `tests/ui` 3 passed in 107 s.

## 2026-11-08: The Library mirrors the papers folder

- Asked for: a deleted PDF should leave the Library completely, not show "File missing"; keep only analysed papers plus the newly added one.
- `scan_library` now calls `remove_paper` for a user paper whose file is gone. It deletes the paper, its sections and claims, pairs and topic links involving those claims, reports of the affected topics (stale), auto topics left with no claims, and the claims' Qdrant points (`vectorstore.remove_claims`). Labels are kept. If the folder does not exist, nothing is removed. `ScanResult.missing` became `removed`, and the scan endpoints return `removed`.
- Tests: the old "marked missing and returns" test now checks the new behaviour (removed, file untouched, same id when it returns). Added tests for a missing folder and for the cleanup around an analysed paper. `tests/test_ingest.py` 34 passed; `test_api.py`, `test_pipeline_plumbing.py`, `test_config_db.py` 35 passed.
- With the user's choice, the 39 unanalysed PDFs in `paper/` were moved (not deleted) to `papers_not_analysed/`, each checked by SHA-256 after the move. Kept: the 6 analysed papers and the new upload `paper/uploads/Facial_Action_Unit-...pdf`.
- A scan of the real library then removed 40 papers (39 moved, 1 already deleted by the user). Papers 47 -> 7; topics 6, pairs 29, reports 3 unchanged. Claims 34 -> 32: one moved paper had 2 claims from an interrupted run (still "pending"); they went with it. No topic link points at a missing claim.
