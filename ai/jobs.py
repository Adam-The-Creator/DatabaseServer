"""
Background jobs of the heavy AI functions (requirements AI-02, TEC-05): they run on a small thread pool, report their
progress, can be cancelled, and only a limited number run at once, so the game's requests on the same server are not
starved. A job is cancelled between its steps: a request to a model that is already running is not interrupted, it ends
at the model's own timeout.

The state lives in memory, which is enough for one server process. What a job produces is stored by its function
(ai/store.py), so a restart loses the progress of running jobs, not the finished results.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from ai import settings

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
ACTIVE = (QUEUED, RUNNING)


class JobCancelled(Exception):
    """Raised inside a job at the next check after it was cancelled."""


class Job:
    def __init__(self, kind: str, target_id: str) -> None:
        self.id = uuid4().hex
        self.kind = kind  # "labels", "description", "summary" or "download"
        self.target_id = target_id
        self.status = QUEUED
        self.progress = 0.0
        self.message = ""
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self._cancel = threading.Event()

    def update(self, progress: float, message: str = "") -> None:
        self.check_cancelled()
        self.progress = max(0.0, min(1.0, progress))
        if message:
            self.message = message

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "targetId": self.target_id, "status": self.status,
                "progress": round(self.progress, 3), "message": self.message, "error": self.error,
                "createdAt": self.created_at, "finishedAt": self.finished_at}


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None

    def submit(self, kind: str, target_id: str, run: Callable[[Job], None]) -> tuple[Job, bool]:
        """
        Queues the work. While a job of the same kind and target is queued or running, that one is returned instead
        of a second one (a double click must not start the work twice). Returns (job, newly started).
        """

        with self._lock:
            for job in self._jobs.values():
                if job.kind == kind and job.target_id == target_id and job.status in ACTIVE:
                    return job, False
            job = Job(kind, target_id)
            self._jobs[job.id] = job
            self._trim()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=max(1, int(settings.number("AI_MAX_CONCURRENT_JOBS", 1))), thread_name_prefix="ai-job")
            self._executor.submit(self._run, job, run)
            return job, True

    def _run(self, job: Job, run: Callable[[Job], None]) -> None:
        try:
            job.check_cancelled()  # cancelled while it was waiting in the queue
            job.status = RUNNING
            run(job)
            job.progress = 1.0
            job.status = DONE
        except JobCancelled:
            job.status = CANCELLED
        except Exception as error:  # a failed job must not take the worker down; the message is shown to the user
            job.status = FAILED
            job.error = str(error) or error.__class__.__name__
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def _trim(self) -> None:
        limit = int(settings.number("AI_JOB_HISTORY", 100))
        finished = [job for job in self._jobs.values() if job.status not in ACTIVE]
        for job in sorted(finished, key=lambda j: j.created_at)[:max(0, len(self._jobs) - limit)]:
            del self._jobs[job.id]

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def active(self, kind: str, target_id: str) -> Optional[Job]:
        with self._lock:
            return next((j for j in self._jobs.values()
                         if j.kind == kind and j.target_id == target_id and j.status in ACTIVE), None)

    def latest(self, kind: str, target_id: str) -> Optional[Job]:
        """The newest job of this kind for this target, finished or not (so a failure can still be shown)."""

        with self._lock:
            jobs = [j for j in self._jobs.values() if j.kind == kind and j.target_id == target_id]
        return max(jobs, key=lambda j: j.created_at, default=None)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if job is not None and job.status in ACTIVE:
            job._cancel.set()
        return job

    def shutdown(self) -> None:
        """Called when the server stops: running jobs are asked to stop, queued ones are dropped."""

        with self._lock:
            for job in self._jobs.values():
                job._cancel.set()
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


manager = JobManager()
