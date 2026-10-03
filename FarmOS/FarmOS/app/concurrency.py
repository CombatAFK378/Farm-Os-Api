"""
FarmOS – Blocking-work offloading
==================================
Leaf image maths, satellite tile downloads and Groq calls are all blocking.
Running them directly inside `async def` endpoints froze the event loop, so
one slow request stalled every other request (even /health).

`run_blocking` moves that work onto a worker thread, and a shared semaphore
caps how many heavy jobs run at once so a burst of requests can't push a
small instance past its RAM (a single job peaks at roughly 130–280 MB).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

from fastapi.concurrency import run_in_threadpool

from app.config import settings

T = TypeVar("T")

_job_slots = asyncio.Semaphore(settings.max_concurrent_jobs)


async def run_blocking(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking function in a worker thread once a job slot is free."""
    async with _job_slots:
        return await run_in_threadpool(func, *args, **kwargs)
