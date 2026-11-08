"""Browser test of every screen, in light and dark, with screenshots in results/ui_screenshots."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "results" / "ui_screenshots"
FIXTURES = ROOT / "tests" / "fixtures"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".venv" / "playwright"))

pytestmark = pytest.mark.ui
playwright_api = pytest.importorskip("playwright.sync_api")

# Words the interface must not use (spec 9.1).
BANNED = re.compile(r"\b(ai-powered|magic|smart|insights?|unlock|ask ai|powered by ai)\b", re.I)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        pytest.skip("The frontend is not built. Run: python scripts/run.py setup")
    temp = tmp_path_factory.mktemp("ui")
    (temp / "papers").mkdir()
    env = {
        **os.environ,
        "CLAIMGRAPH_DATA_DIR": str(temp / "data"),
        "CLAIMGRAPH_PAPERS_DIR": str(temp / "papers"),
        "DATABASE_URL": f"sqlite:///{(temp / 'ui.db').as_posix()}",
        "PYTHONPATH": str(ROOT / "src"),
    }
    log = (temp / "server.log").open("w")
    process = subprocess.Popen([sys.executable, str(ROOT / "tests" / "ui" / "stand_in_server.py"), str(PORT)], env=env, stdout=log, stderr=log, cwd=ROOT)
    try:
        for _ in range(600):
            if process.poll() is not None:
                pytest.fail("The test server stopped: " + (temp / "server.log").read_text()[-1500:])
            try:
                with urllib.request.urlopen(f"{BASE}/health", timeout=2):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("The test server did not start.")
        yield BASE
    finally:
        process.terminate()
        process.wait(timeout=20)
        log.close()


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Chromium is not installed ({exc.__class__.__name__}). Run: python scripts/run.py setup")
        yield chromium
        chromium.close()


def api(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as response:
        return json.loads(response.read())


def audit(page, name: str) -> None:
    """The checks of spec 9.1 and 9.2 that a program can make, on whatever is on screen."""
    text = page.locator("body").inner_text()
    found = BANNED.search(text)
    assert not found, f"{name}: banned word {found.group(0)!r}"
    assert not re.search("[\U0001F300-\U0001FAFF✨]", text), f"{name}: emoji"
    style = page.evaluate(
        """() => {
          const out = {gradients: [], shadows: [], blur: []};
          for (const el of document.querySelectorAll('body *')) {
            const s = getComputedStyle(el);
            if (s.backgroundImage.includes('gradient')) out.gradients.push(el.tagName + '.' + el.className);
            if (s.boxShadow !== 'none' && !el.closest('.textLayer')) out.shadows.push(el.tagName + '.' + el.className);
            if (s.backdropFilter && s.backdropFilter !== 'none') out.blur.push(el.tagName);
          }
          return out;
        }"""
    )
    assert not style["gradients"], f"{name}: gradients {style['gradients'][:3]}"
    assert not style["blur"], f"{name}: glass effect"
    assert not style["shadows"], f"{name}: shadows {style['shadows'][:3]}"
    overflow = page.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth")
    assert not overflow, f"{name}: the page scrolls sideways"


def shot(page, name: str, theme: str) -> None:
    audit(page, name)
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / f"{name}_{theme}.png"))


def go(page, route: str) -> None:
    page.goto(f"{BASE}/#{route}")


def tour(page, theme: str, topic_id: int, pair_id: int, paper_id: str) -> None:
    """Visit every screen and take its screenshot."""
    go(page, "/library")
    page.get_by_test_id("paper-row").first.wait_for()
    shot(page, "1_library", theme)

    go(page, f"/paper/{paper_id}?page=1")
    page.locator(".textLayer mark").first.wait_for(timeout=30_000)
    marked = "".join(page.locator(".textLayer mark").all_inner_texts())
    assert marked.startswith("Adding dropout") and marked.rstrip().endswith("percent."), marked  # the quote and nothing else
    shot(page, "2_paper", theme)

    go(page, "/topics")
    page.get_by_test_id("topic-row").first.wait_for()
    shot(page, "3_topics", theme)

    go(page, f"/topics/{topic_id}")
    page.locator("[data-testid=graph] canvas").first.wait_for()
    page.wait_for_timeout(900)  # the layout animates once
    shot(page, "4_topic_graph", theme)
    edge_tap(page)
    page.get_by_test_id("pair-detail").wait_for()
    page.wait_for_timeout(300)  # the panel slides in
    assert page.evaluate("getComputedStyle(document.querySelector('[data-testid=pair-panel]')).opacity") == "1"
    shot(page, "4b_graph_pair_panel", theme)

    go(page, f"/pair/{pair_id}")
    page.get_by_test_id("pair-detail").wait_for()
    shot(page, "5_pair", theme)

    go(page, f"/report/{topic_id}")
    page.get_by_test_id("report-body").wait_for()
    shot(page, "6_report", theme)

    go(page, "/labelling")
    page.get_by_test_id("label-stats").wait_for()
    page.wait_for_timeout(300)
    shot(page, "7_labelling", theme)

    go(page, "/settings")
    page.get_by_test_id("setting-row").first.wait_for()
    shot(page, "8_settings", theme)


def edge_tap(page, relation: str = "contradicts") -> None:
    """Click the middle of an edge with the real mouse, as a person would."""
    point = page.evaluate(
        """(relation) => {
          const box = document.querySelector('[data-testid=graph]');
          const cy = box._cyreg.cy;
          const edge = cy.edges(`[relation = "${relation}"]`)[0];
          const ends = (e) => [e.renderedSourceEndpoint(), e.renderedTargetEndpoint()];
          const along = ([p, q], t) => ({x: p.x + (q.x - p.x) * t, y: p.y + (q.y - p.y) * t});
          const gap = (point, [p, q]) => {
            const dx = q.x - p.x, dy = q.y - p.y;
            const t = Math.max(0, Math.min(1, ((point.x - p.x) * dx + (point.y - p.y) * dy) / (dx * dx + dy * dy || 1)));
            return Math.hypot(point.x - p.x - t * dx, point.y - p.y - t * dy);
          };
          // The layout is random, so the midpoint can sit on a crossing edge. Use the clearest point.
          const others = cy.edges().not(edge).map(ends);
          let best = null, clearance = -1;
          for (let t = 0.2; t <= 0.8; t += 0.05) {
            const point = along(ends(edge), t);
            const nearest = Math.min(Infinity, ...others.map((segment) => gap(point, segment)));
            if (nearest > clearance) { best = point; clearance = nearest; }
          }
          const rect = box.getBoundingClientRect();
          return {x: rect.left + best.x, y: rect.top + best.y};
        }""",
        relation,
    )
    page.mouse.click(point["x"], point["y"])


def test_full_journey(server, browser):
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)

    # The app opens straight into the library, with no landing page.
    page.goto(BASE)
    page.wait_for_url(re.compile(r"/#/$|/$"))
    assert page.get_by_text("Drop PDFs here or add them to the papers folder").is_visible()
    page.get_by_text("No papers yet").wait_for()

    # Upload the fixtures through the real file input.
    page.get_by_test_id("upload").set_input_files([str(path) for path in sorted(FIXTURES.glob("*.pdf"))])
    page.get_by_test_id("paper-row").nth(3).wait_for(timeout=60_000)
    assert "4 papers, 4 new" in page.get_by_test_id("library-count").inner_text()

    # Analyse, with plain wording in the progress strip.
    page.get_by_test_id("analyse-all").click()
    page.get_by_text("Analysis finished. 4 new claims found.").wait_for(timeout=120_000)
    page.get_by_text("Analysed").first.wait_for()
    assert "0 new" in page.get_by_test_id("library-count").inner_text()

    # Keyboard: g then t goes to Topics; j and Enter open the topic.
    page.keyboard.press("g")
    page.keyboard.press("t")
    page.wait_for_url(re.compile(r"#/topics$"))
    row = page.get_by_test_id("topic-row").first
    row.wait_for()
    assert "Dropout and test accuracy" in row.inner_text()
    assert row.locator("[role=img]").get_attribute("aria-label").count("contradicts") == 1
    page.keyboard.press("j")
    page.keyboard.press("Enter")
    page.wait_for_url(re.compile(r"#/topics/\d+$"))
    page.locator("[data-testid=graph] canvas").first.wait_for()
    page.wait_for_timeout(900)

    # Clicking an edge opens the pair panel; Esc closes it.
    edge_tap(page)
    panel = page.get_by_test_id("pair-panel")
    panel.wait_for()
    panel.get_by_test_id("pair-detail").wait_for()
    assert "Contradicts" in panel.inner_text()
    page.keyboard.press("Escape")
    panel.wait_for(state="detached")

    # The planted different-setting case: the differing condition is highlighted and named.
    edge_tap(page, "different_setting")
    panel.get_by_test_id("pair-detail").wait_for()
    table = panel.get_by_test_id("condition-table")
    differing = table.locator("tr[data-differs]")
    assert differing.count() == 1 and "Dataset" in differing.inner_text(), panel.inner_text()
    assert "ImageNet" in differing.inner_text() and "CIFAR-10" in differing.inner_text()
    assert "The dataset differs." in panel.get_by_test_id("pair-reason").inner_text()

    # Correcting a label shows at once and survives a reload.
    pair_url = panel.get_by_text("Open full view")
    pair_url.click()
    page.wait_for_url(re.compile(r"#/pair/\d+$"))
    pair_id = int(page.url.rsplit("/", 1)[1])
    page.get_by_test_id("pair-detail").wait_for()
    page.get_by_role("button", name="Unclear").click()
    page.get_by_text("Corrected by you").wait_for()
    page.reload()
    page.get_by_text("Corrected by you").wait_for()
    page.get_by_role("button", name="Different setting").click()  # put it back
    page.get_by_text("Confirmed by you").wait_for()

    # Report: citations open the claim, and every citation is a real one.
    page.get_by_role("link", name="Back to the graph").click()
    page.get_by_test_id("open-report").click()
    page.get_by_test_id("report-body").wait_for(timeout=60_000)
    citations = page.get_by_test_id("citation")
    assert citations.count() > 0
    citations.first.click()
    dialog = page.get_by_test_id("claim-dialog")
    dialog.wait_for()
    assert "Adding dropout" in dialog.inner_text()
    page.keyboard.press("Escape")
    dialog.wait_for(state="detached")
    with page.expect_download() as download:
        page.get_by_test_id("export-md").click()
    assert download.value.suggested_filename.endswith(".md")
    with page.expect_download() as download:
        page.get_by_test_id("export-pdf").click()
    assert download.value.suggested_filename.endswith(".pdf")

    # Keyboard: g then l returns to the Library; / focuses search.
    page.keyboard.press("g")
    page.keyboard.press("l")
    page.wait_for_url(re.compile(r"#/library$"))
    page.keyboard.press("/")
    assert page.evaluate("document.activeElement.dataset.testid") == "search"
    page.keyboard.type("dropout")
    page.get_by_text("Claims", exact=True).first.wait_for()
    page.keyboard.press("Escape")
    assert page.evaluate("document.activeElement.dataset.testid") != "search"

    # Labelling by keyboard: key 1 labels the pair and moves on.
    go(page, "/labelling")
    stats = page.get_by_test_id("label-stats")
    stats.wait_for()
    page.get_by_test_id("pair-detail").wait_for()
    page.wait_for_timeout(300)
    before = int(stats.inner_text().split()[0])
    page.keyboard.press("1")
    page.wait_for_function("(n) => parseInt(document.querySelector('[data-testid=label-stats]').innerText) === n", arg=before + 1)
    page.get_by_role("tab", name="Claim checks").click()
    page.get_by_test_id("claim-check").wait_for()
    checked = page.get_by_test_id("claim-check").inner_text()
    page.keyboard.press("1")
    page.wait_for_function("(old) => document.querySelector('[data-testid=claim-check]').innerText !== old", arg=checked)

    # Screenshots of every screen in both themes.
    topics = api("/topics")["topics"]
    papers = api("/library")["papers"]
    tour(page, "light", topics[0]["id"], pair_id, papers[0]["id"])

    go(page, "/settings")
    page.get_by_test_id("theme-dark").click()
    assert page.evaluate("document.documentElement.dataset.theme") == "dark"
    background = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert background == "rgb(22, 24, 27)"
    tour(page, "dark", topics[0]["id"], pair_id, papers[0]["id"])

    page.reload()  # the choice is remembered
    assert page.evaluate("document.documentElement.dataset.theme") == "dark"
    page.get_by_test_id("theme-light").click()

    assert not errors, errors[:5]
    context.close()


def test_light_colours_match_the_spec(server, browser):
    page = browser.new_page()
    page.goto(BASE)
    tokens = page.evaluate(
        """() => {
          const s = getComputedStyle(document.documentElement);
          return ['--bg','--surface','--text','--text-2','--border','--accent','--supports','--contradicts','--setting','--unclear']
            .map(name => s.getPropertyValue(name).trim().toLowerCase());
        }"""
    )
    assert tokens == ["#fafaf7", "#ffffff", "#1f2328", "#5b616b", "#e3e3de", "#0f5f5f", "#2f7d4f", "#b23a3a", "#8a8f98", "#c9a227"]
    page.close()


def test_tablet_width_stays_usable(server, browser):
    context = browser.new_context(viewport={"width": 900, "height": 1100})
    page = context.new_page()
    page.goto(f"{BASE}/#/library")
    page.get_by_test_id("paper-row").first.wait_for()
    audit(page, "tablet library")
    go(page, "/topics")
    page.get_by_test_id("topic-row").first.wait_for()
    audit(page, "tablet topics")
    context.close()
