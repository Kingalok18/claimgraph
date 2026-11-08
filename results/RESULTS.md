# ClaimGraph results

Every number below comes from a JSON file in this folder, written by a script in `src/claimgraph/eval/`.

## Planted fixtures: claims and relations recovered

Run date: 2026-11-07 02:19 UTC

- Planted claims found: 4 of 4
- Planted relations correct: 5 of 5
- All stored quotes verified: True
- Model: qwen2.5-3b-instruct-q4_k_m.gguf; database: mysql; time: 571.0 s
  - A1 / B1: expected supports, got supports
  - A1 / C1: expected contradicts, got contradicts
  - B1 / C1: expected contradicts, got contradicts
  - A1 / D1: expected different_setting, got different_setting
  - B1 / D1: expected different_setting, got different_setting
- Report: 5 of 5 planted relations under the right heading
- Report written by: qwen2.5-3b-instruct-q4_k_m.gguf; model sentences removed by the checks: 4; time: 147.4 s

## NLI model on SciFact (dev split)

Run date: 2026-11-07 02:29 UTC

- Both directions averaged, as the pipeline scores claim pairs:
- n = 450, accuracy 0.4111, macro F1 0.4016
  - entailment: P 1.0, R 0.0463, F1 0.0885 (n=216)
  - contradiction: P 0.7931, R 0.5656, F1 0.6603 (n=122)
  - neutral: P 0.3003, R 0.9464, F1 0.4559 (n=112)
- Model: cross-encoder/nli-deberta-v3-base
- One way only, evidence to claim, which is the question SciFact asks:
  - n = 450, accuracy 0.5622, macro F1 0.5771
  - entailment: P 0.9481, R 0.338, F1 0.4983 (n=216)
  - contradiction: P 0.7642, R 0.6639, F1 0.7105 (n=122)
  - neutral: P 0.3708, R 0.8839, F1 0.5224 (n=112)

## Report citations

Run date: 2026-11-07 02:29 UTC

- reports: 3
- citations: 16
- unresolved citations: 0
- model sentences removed: 0
- reports using template: 3
- sentences failing check after cleanup: 0

## Run on the user's papers

Run date: 2026-11-06 20:46 UTC

- sample: hand-picked to share a subject (document forgery), not a random sample
- papers in folder: 46
- pdfs unchanged: True
- seconds this run: 316.1
- papers in sample: 5
- claims kept: 22
- claims dropped: 11
- drop rate: 0.3333
- claims with unverified quote: 0
- topics: 6
- pairs: 29
- pairs supports: 0
- pairs contradicts: 0
- pairs different setting: 8
- pairs unclear: 21

## Waiting

These have no accuracy result yet. Any counts shown are stored data, not accuracy.

- **NLI model on SciFact-Open** (not selected): Dataset not selected at setup. Optional step: see the README.
- **BioDivergence** (not selected): Dataset not selected at setup; check its licence first. See the README.
- **Claim extraction, checked against hand labels** (waiting for labels): 0 of at least 20 claims labelled. Use the Labelling screen.
  - claims stored: 22
  - claims dropped by checks: 11
  - drop rate: 0.3333
- **Claim-pair relations, checked against hand labels** (waiting for labels): 0 of at least 20 pairs labelled. Use the Labelling screen.
