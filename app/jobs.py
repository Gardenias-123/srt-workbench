"""Threaded background job manager for forward/inversion/sensitivity runs."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"          # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    process: object | None = None

    @property
    def result_dir(self) -> Path:
        return Path(self.params.get("_result_dir", ""))


class JobManager:
    def __init__(self, runs_dir: Path, max_workers: int = 2):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()

    # ---------- lifecycle ----------
    def submit(self, kind: str, params: dict, fn: Callable[[Job, "JobManager"], dict]) -> Job:
        job_id = uuid.uuid4().hex[:12]
        result_dir = self.runs_dir / job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        data = dict(params)
        data["_result_dir"] = str(result_dir)
        job = Job(id=job_id, kind=kind, params=data)
        with self.lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn):
        with self.lock:
            job.status = "running"
            job.started_at = time.time()
            job.message = "started"
        try:
            result = fn(job, self)
            with self.lock:
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.message = "cancelled"
                else:
                    job.status = "done"
                    job.progress = 100.0
                    job.message = "finished"
                    job.result = result
                job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = job.error
                job.finished_at = time.time()
                job.log.append(traceback.format_exc())

    # ---------- updates ----------
    def log(self, job: Job, line: str):
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            job.log.append(line)
            if len(job.log) > 800:
                del job.log[:200]

    def progress(self, job: Job, value: float, message: str | None = None):
        with self.lock:
            job.progress = max(0.0, min(100.0, float(value)))
            if message:
                job.message = message

    def set_process(self, job: Job, process):
        with self.lock:
            job.process = process

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job.status in ("done", "error", "cancelled"):
                return False
            job.cancel_event.set()
            job.message = "cancelling"
            process = job.process
        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass
        return True

    # ---------- snapshots ----------
    def snapshot(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            return {
                "id": job.id,
                "kind": job.kind,
                "status": job.status,
                "progress": round(job.progress, 2),
                "message": job.message,
                "error": job.error,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "log_tail": job.log[-80:],
                "has_result": job.result is not None,
            }

    def result(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.result if job else None

    def result_dir(self, job_id: str) -> Path | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.result_dir if job else None

    def list_jobs(self, limit: int = 30) -> list[dict]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [self.snapshot(j.id) for j in jobs[:limit]]
