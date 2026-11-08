"""Build results/RESULTS.md from results/*.json only. Nothing here computes or invents a number."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TITLES = {
    "fixture_eval": "Planted fixtures: claims and relations recovered",
    "scifact_nli": "NLI model on SciFact (dev split)",
    "scifact_open": "NLI model on SciFact-Open",
    "biodivergence": "BioDivergence",
    "extraction": "Claim extraction, checked against hand labels",
    "pairs": "Claim-pair relations, checked against hand labels",
    "report": "Report citations",
    "user_papers": "Run on the user's papers",
}


def metric_lines(data: dict) -> list[str]:
    name = data["name"]
    lines: list[str] = []
    if name == "fixture_eval":
        lines.append(f"- Planted claims found: {data['claims_found']} of {data['claims_planted']}")
        lines.append(f"- Planted relations correct: {data['relations_correct']} of {data['relations_expected']}")
        lines.append(f"- All stored quotes verified: {data['all_quotes_verified']}")
        lines.append(f"- Model: {data.get('llm')}; database: {data.get('database')}; time: {data.get('seconds')} s")
        for item in data.get("relations", []):
            lines.append(f"  - {item['a']} / {item['b']}: expected {item['expected']}, got {item['got']}")
        report = data.get("report")
        if report:
            lines.append(f"- Report: {report['relations_in_right_section']} of {report['relations_checked']} planted relations under the right heading")
            lines.append(f"- Report written by: {report.get('model')}; model sentences removed by the checks: {report['model_sentences_removed']}; time: {report['seconds']} s")
        return lines
    if "accuracy" in data:
        lines.append(f"- n = {data['n']}, accuracy {data['accuracy']}, macro F1 {data['macro_f1']}")
        for label, values in data.get("per_class", {}).items():
            lines.append(f"  - {label}: P {values['precision']}, R {values['recall']}, F1 {values['f1']} (n={values['support']})")
        if data.get("model"):
            lines.append(f"- Model: {data['model']}")
        if data.get("both_directions"):
            lines.insert(0, "- Both directions averaged, as the pipeline scores claim pairs:")
        if data.get("one_way"):
            other = data["one_way"]
            lines.append("- One way only, evidence to claim, which is the question SciFact asks:")
            lines.append(f"  - n = {other['n']}, accuracy {other['accuracy']}, macro F1 {other['macro_f1']}")
            for label, values in other.get("per_class", {}).items():
                lines.append(f"  - {label}: P {values['precision']}, R {values['recall']}, F1 {values['f1']} (n={values['support']})")
        return lines
    skip = {"name", "run_date", "status", "reason"}
    for key, value in data.items():
        if key not in skip and not isinstance(value, (list, dict)):
            lines.append(f"- {key.replace('_', ' ')}: {value}")
        elif key == "quote_check":
            lines.extend(f"- {k.replace('_', ' ')}: {v}" for k, v in value.items())
    return lines


def main() -> Path:
    files = sorted(RESULTS.glob("*.json"), key=lambda path: list(TITLES).index(path.stem) if path.stem in TITLES else 99)
    output = ["# ClaimGraph results", "", "Every number below comes from a JSON file in this folder, written by a script in `src/claimgraph/eval/`.", ""]
    waiting: list[str] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        title = TITLES.get(data.get("name"), data.get("name", path.stem))
        if data.get("status") != "measured":
            waiting.append(f"- **{title}** ({data.get('status', 'unknown').replace('_', ' ')}): {data.get('reason', '')}")
            lines = metric_lines(data)
            if lines:
                waiting.extend("  " + line for line in lines)
            continue
        output += [f"## {title}", "", f"Run date: {data.get('run_date')}", ""] + metric_lines(data) + [""]
    if waiting:
        output += ["## Waiting", "", "These have no accuracy result yet. Any counts shown are stored data, not accuracy.", ""] + waiting + [""]
    target = RESULTS / "RESULTS.md"
    target.write_text("\n".join(output), encoding="utf-8")
    print(f"Wrote {target}")
    return target


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
