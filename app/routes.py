"""HTTP routes for the SRT/TopoSRT GUI."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory

from .engines import fortran, jax_engine
from .params import SRTParams

bp = Blueprint("main", __name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_RATE_LOCK = threading.Lock()
_RATE: dict[str, list[float]] = {}


def _rate_ok() -> bool:
    """Very small in-memory limiter; disabled unless SRT_RATE_LIMIT is set."""
    limit = _env_int("SRT_RATE_LIMIT", 0)
    if limit <= 0:
        return True
    ip = request.remote_addr or "unknown"
    now = time.time()
    window = _env_int("SRT_RATE_WINDOW", 600)
    with _RATE_LOCK:
        arr = [t for t in _RATE.get(ip, []) if now - t < window]
        if len(arr) >= limit:
            return False
        arr.append(now)
        _RATE[ip] = arr
    return True


def _check_limits(params: SRTParams | None = None):
    if not _rate_ok():
        return jsonify(error="请求过于频繁，请稍后再试。"), 429
    max_nz = _env_int("SRT_MAX_NZ", 0)
    if params is not None and max_nz and params.nz > max_nz:
        return jsonify(error=f"在线演示限制 nz ≤ {max_nz}，请调小离散层数。"), 400
    return None


def _forward_job(job, manager):
    params = SRTParams.from_dict(job.params)
    return fortran.run_forward(params, job, manager)


def _jax_forward_job(job, manager):
    params = SRTParams.from_dict(job.params)
    return jax_engine.run_jax_forward(params, job, manager)


def _inversion_job(job, manager):
    params = SRTParams.from_dict(job.params["params"])
    return jax_engine.run_inversion_job(params, job.params["options"], job, manager)


def _sensitivity_job(job, manager):
    params = SRTParams.from_dict(job.params["params"])
    return jax_engine.run_sensitivity_job(params, job.params["options"], job, manager)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/api/health")
def health():
    if not fortran.fortran_enabled():
        return jsonify(ok=True, engine="jax", fortran=False, exe_hash="jax", mode="demo")
    try:
        exe = fortran.find_exe()
        return jsonify(
            ok=True,
            fortran=True,
            exe=str(exe),
            exe_hash=fortran.exe_hash(exe),
            engine="SRT_topo" if "SRT_topo" in str(exe) else "SRT",
            mode="full",
        )
    except Exception as exc:  # noqa: BLE001
        # The app still works with the JAX engine when the Fortran binary is absent.
        return jsonify(ok=True, engine="jax", fortran=False, exe_hash="jax",
                       mode="jax-only", error=str(exc))


@bp.post("/api/forward")
def forward():
    data = request.get_json(force=True, silent=True) or {}
    try:
        params = SRTParams.from_dict(data)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 400
    if not fortran.fortran_enabled():
        return jsonify(error="Fortran 引擎未启用；请使用 JAX 正演。"), 503
    guard = _check_limits(params)
    if guard:
        return guard
    job = current_app.jobs.submit("forward", params.to_dict(), _forward_job)
    return jsonify(job_id=job.id, status=job.status)


@bp.post("/api/jax/forward")
def jax_forward():
    data = request.get_json(force=True, silent=True) or {}
    try:
        params = SRTParams.from_dict(data)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 400
    guard = _check_limits(params)
    if guard:
        return guard
    job = current_app.jobs.submit("jax_forward", params.to_dict(), _jax_forward_job)
    return jsonify(job_id=job.id, status=job.status)


@bp.get("/api/jobs")
def list_jobs():
    return jsonify(jobs=current_app.jobs.list_jobs())


@bp.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    snap = current_app.jobs.snapshot(job_id)
    if snap is None:
        return jsonify(error="job not found"), 404
    return jsonify(snap)


@bp.get("/api/jobs/<job_id>/result")
def job_result(job_id: str):
    snap = current_app.jobs.snapshot(job_id)
    if snap is None:
        return jsonify(error="job not found"), 404
    if snap["status"] != "done":
        return jsonify(error="job not finished", status=snap["status"]), 409
    result = current_app.jobs.result(job_id)
    return jsonify(result or {})


@bp.post("/api/jobs/<job_id>/cancel")
def cancel(job_id: str):
    ok = current_app.jobs.cancel(job_id)
    return jsonify(cancelled=ok)


@bp.get("/api/download/<job_id>/<path:name>")
def download(job_id: str, name: str):
    directory = current_app.jobs.result_dir(job_id)
    if directory is None or not directory.exists():
        return jsonify(error="job not found"), 404
    safe = Path(name).name
    if safe not in {"brf.csv", "jax_brf.csv", "result.json"}:
        return jsonify(error="file not available"), 400
    if not (directory / safe).exists():
        return jsonify(error="file not generated yet"), 404
    return send_from_directory(directory, safe, as_attachment=True)


@bp.post("/api/inversion")
def inversion():
    data = request.get_json(force=True, silent=True) or {}
    try:
        params = SRTParams.from_dict(data.get("params", {}))
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 400
    guard = _check_limits(params)
    if guard:
        return guard
    payload = {k: v for k, v in data.items() if k != "params"}
    max_iter = _env_int("SRT_MAX_ITER", 0)
    if max_iter and payload.get("n_iter"):
        payload["n_iter"] = min(int(payload["n_iter"]), max_iter)
    job = current_app.jobs.submit(
        "inversion", {"params": params.to_dict(), "options": payload}, _inversion_job
    )
    return jsonify(job_id=job.id, status=job.status)


@bp.post("/api/sensitivity")
def sensitivity():
    data = request.get_json(force=True, silent=True) or {}
    try:
        params = SRTParams.from_dict(data.get("params", {}))
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 400
    guard = _check_limits(params)
    if guard:
        return guard
    payload = {k: v for k, v in data.items() if k != "params"}
    job = current_app.jobs.submit(
        "sensitivity", {"params": params.to_dict(), "options": payload}, _sensitivity_job
    )
    return jsonify(job_id=job.id, status=job.status)
