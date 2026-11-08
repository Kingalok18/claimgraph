"""Build small test PDFs with planted claims, and the answers they should produce.

Papers A and B agree. Paper C truly contradicts them under the same conditions. Paper D looks like a
contradiction, but on a different dataset, so the correct relation is "different setting".
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

METHOD = (
    "We trained each network for 200 epochs with stochastic gradient descent and a batch size of 128. "
    "Each configuration was repeated five times with different random seeds, and the mean is reported. "
    "The learning rate started at 0.1 and was divided by ten at epochs 100 and 150."
)
RELATED = (
    "Earlier work by Harlow and Chen reported that weight decay improved the generalisation of small networks. "
    "Okafor and Lind described several regularisation methods for convolutional networks."
)

PAPERS = [
    {
        "key": "A",
        "file": "fixture_a_dropout_cifar_gain.pdf",
        "title": "Measuring Dropout in Residual Networks for Image Classification",
        "authors": "Mira Ansel, Tomas Reuter",
        "dataset": "CIFAR-10",
        "claim_key": "A1",
        "quote": "Adding dropout increased the test accuracy of ResNet-50 on CIFAR-10 from 92.1 percent to 93.4 percent.",
    },
    {
        "key": "B",
        "file": "fixture_b_dropout_cifar_gain.pdf",
        "title": "A Replication Study of Dropout Regularisation in Deep Residual Networks",
        "authors": "Lena Fischer, Omar Haddad",
        "dataset": "CIFAR-10",
        "claim_key": "B1",
        "quote": "Adding dropout increased the test accuracy of ResNet-50 on CIFAR-10 from 92.4 percent to 93.5 percent.",
    },
    {
        "key": "C",
        "file": "fixture_c_dropout_cifar_loss.pdf",
        "title": "When Dropout Fails: Regularisation Costs in Residual Networks",
        "authors": "Priya Nair, Jonas Berg",
        "dataset": "CIFAR-10",
        "claim_key": "C1",
        "quote": "Adding dropout decreased the test accuracy of ResNet-50 on CIFAR-10 from 93.0 percent to 91.8 percent.",
    },
    {
        "key": "D",
        "file": "fixture_d_dropout_imagenet_loss.pdf",
        "title": "Dropout at Scale: Regularisation on a Large Image Benchmark",
        "authors": "Sara Lindqvist, Wei Tan",
        "dataset": "ImageNet",
        "claim_key": "D1",
        "quote": "Adding dropout decreased the test accuracy of ResNet-50 on ImageNet from 76.2 percent to 75.1 percent.",
    },
]

EXPECTED_RELATIONS = [
    {"a": "A1", "b": "B1", "relation": "supports"},
    {"a": "A1", "b": "C1", "relation": "contradicts"},
    {"a": "B1", "b": "C1", "relation": "contradicts"},
    {"a": "A1", "b": "D1", "relation": "different_setting"},
    {"a": "B1", "b": "D1", "relation": "different_setting"},
]


def build_pdf(paper: dict, folder: Path) -> Path:
    title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=8)
    authors = ParagraphStyle("authors", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=14)
    heading = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10, leading=14, spaceAfter=6)

    def section(name: str, text: str) -> list:
        return [Paragraph(name, heading), Paragraph(text, body)]

    story = [Paragraph(paper["title"], title), Paragraph(paper["authors"], authors)]
    story += section(
        "Abstract",
        f"This paper studies the effect of dropout regularisation on ResNet-50 for image classification on {paper['dataset']}. "
        "We describe the training procedure and report the measured test accuracy.",
    )
    story += section("1 Related Work", RELATED)
    story += section("2 Method", METHOD)
    story += section(
        "3 Results",
        f"{paper['quote']} The difference was consistent across all five random seeds. "
        "Training time per epoch stayed within two percent of the baseline.",
    )
    story += section("4 Conclusion", "The measurements above summarise our study of dropout in residual networks.")
    story.append(Spacer(1, 6 * mm))

    path = folder / paper["file"]
    SimpleDocTemplate(str(path), pagesize=A4, title=paper["title"], author=paper["authors"]).build(story)
    return path


def main(folder: Path = OUT) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for paper in PAPERS:
        build_pdf(paper, folder)
    expected = {
        "papers": [
            {"key": p["key"], "file": p["file"], "title": p["title"], "claims": [{"key": p["claim_key"], "quote": p["quote"]}]}
            for p in PAPERS
        ],
        "relations": EXPECTED_RELATIONS,
    }
    path = folder / "expected.json"
    path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"Wrote {len(PAPERS)} PDFs and expected.json to {folder}")
    return path


if __name__ == "__main__":
    main()
