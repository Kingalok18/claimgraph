"""Cross-platform commands for ClaimGraph. Use: python scripts/run.py <command>

  setup     create .venv, install Python and npm packages, download models, build the frontend
  up        start Docker services (none are needed on this install)
  start     start the app and open the browser
  eval      run the evaluations and write results/
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
FRONTEND = ROOT / "frontend"
HOST, PORT = "127.0.0.1", 8000
URL = f"http://{HOST}:{PORT}"


def env() -> dict:
    values = dict(os.environ)
    values["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + values.get("PYTHONPATH", "")
    values["PYTHONIOENCODING"] = "utf-8"
    values.setdefault("TOKENIZERS_PARALLELISM", "false")
    return values


def run(*command, check: bool = True, cwd: Path = ROOT) -> int:
    print("> " + " ".join(str(part) for part in command), flush=True)
    return subprocess.run([str(part) for part in command], cwd=cwd, env=env(), check=check).returncode


def npm() -> str:
    found = shutil.which("npm.cmd" if os.name == "nt" else "npm") or shutil.which("npm")
    if not found:
        sys.exit("npm was not found. Install Node.js 20 or newer, then run this again.")
    return found


def need_venv() -> None:
    if not PY.exists():
        sys.exit("The environment is not set up yet. Run: python scripts/run.py setup")


def build_frontend() -> None:
    if not (FRONTEND / "node_modules").exists():
        run(npm(), "ci", cwd=FRONTEND)
    run(npm(), "run", "build", cwd=FRONTEND)


def cmd_setup() -> None:
    if not PY.exists():
        run(sys.executable, "-m", "venv", VENV)
    run(PY, "-m", "pip", "install", "--upgrade", "pip", "--quiet")
    run(PY, "-m", "pip", "install", "-r", ROOT / "requirements.txt")
    build_frontend()
    run(PY, "-c", "from claimgraph.db import init_db; init_db()")
    print("Downloading models from Hugging Face. This happens once.")
    run(
        PY,
        "-c",
        "from claimgraph.config import get_settings as g; s=g().models;"
        "from huggingface_hub import hf_hub_download as h;"
        "h(s['llm']['hf_model_id'], s['llm']['gguf_file']);"
        "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T;"
        "T.from_pretrained(s['nli']['base_model_id']); M.from_pretrained(s['nli']['base_model_id']);"
        "from sentence_transformers import SentenceTransformer as E; E(s['embeddings']['model_id'])",
    )
    print("Setup finished. Start the app with: python scripts/run.py start")


def cmd_up() -> None:
    print("Nothing to start: this install uses MySQL on this PC, PyMuPDF, and local Qdrant.")
    print("Optional services are in docker-compose.yml: docker compose --profile optional up -d")


def _open_when_ready() -> None:
    import urllib.request

    for _ in range(240):
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=2):
                break
        except OSError:
            time.sleep(0.5)
    print(f"\nClaimGraph is running at {URL}\nPress Ctrl+C to stop.\n", flush=True)
    webbrowser.open(URL)


def cmd_start() -> None:
    need_venv()
    if not (FRONTEND / "dist" / "index.html").exists():
        build_frontend()
    threading.Thread(target=_open_when_ready, daemon=True).start()
    try:
        run(PY, "-m", "uvicorn", "claimgraph.api.main:app", "--host", HOST, "--port", PORT, "--log-level", "warning")
    except KeyboardInterrupt:
        print("Stopped.")


def cmd_eval() -> None:
    need_venv()
    run(PY, ROOT / "scripts" / "download_data.py", check=False)
    for module in ("scifact_nli_eval", "scifact_open_eval", "biodivergence_eval", "extraction_eval", "pairs_eval", "report_eval"):
        run(PY, "-m", f"claimgraph.eval.{module}", check=False)
    run(PY, ROOT / "scripts" / "make_results_table.py")


COMMANDS = {"setup": cmd_setup, "up": cmd_up, "start": cmd_start, "eval": cmd_eval}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]]()
