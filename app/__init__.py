"""Flask application factory for the SRT/TopoSRT GUI."""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask

from .jobs import JobManager
from .routes import bp


def _runs_dir() -> Path:
    env = os.environ.get("SRT_RUNS_DIR")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SRT_GUI"
        return base / "runs"
    if os.environ.get("SRT_DEMO", "0") == "1":
        return Path(tempfile.gettempdir()) / "srt_workbench_runs"
    return Path(__file__).resolve().parents[1] / "runs"


def _prune_runs(root: Path, max_age_hours: float = 12.0) -> None:
    """Best-effort cleanup of old job directories (public demo hygiene)."""
    now = time.time()
    try:
        for child in root.iterdir():
            if child.is_dir() and now - child.stat().st_mtime > max_age_hours * 3600:
                shutil.rmtree(child, ignore_errors=True)
    except Exception:
        pass


def create_app(test_config=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_AS_ASCII"] = False
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["RUNS_DIR"] = _runs_dir()
    app.config["RUNS_DIR"].mkdir(parents=True, exist_ok=True)
    if test_config:
        app.config.update(test_config)
    _prune_runs(app.config["RUNS_DIR"])

    demo = os.environ.get("SRT_DEMO", "0") == "1"
    max_workers = int(os.environ.get("SRT_MAX_WORKERS", "1" if demo else "2"))
    app.config["DEMO"] = demo
    app.jobs = JobManager(app.config["RUNS_DIR"], max_workers=max_workers)
    app.register_blueprint(bp)
    return app
