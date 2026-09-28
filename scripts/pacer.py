#!/usr/bin/env python3
"""Shared adaptive pacing + API access for the jazzbot ingestion scripts.

Both scripts create a single Pacer for the whole run and share it across
every request (including all worker threads), so throttle state is global
and cumulative instead of per-request.

On a 429 the interval grows (honoring Retry-After) up to a ceiling, and only
relaxes back toward the base after a *sustained* stretch of clean requests.
That stops the previous behavior of bursting every time one request slipped
through a slow, heavily throttled connection.
"""

import sys
import threading
import time

import requests

API = "https://en.wikipedia.org/w/api.php"


def log(message):
    sys.stderr.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    sys.stderr.flush()


class Pacer:
    """Thread-safe, time-based inter-request pacing for a throttled IP.

    ``wait()`` spaces request starts globally by the current interval, so any
    number of workers still issue ~1 request per interval. On a 429 the
    interval grows (at least ``Retry-After + 1s``) up to ``ceiling``; it only
    decays back toward ``base`` after ``relax_after`` seconds with no throttle,
    one halving per window -- a sustained clean streak, not a mere count of
    requests, is required before backing off the pacing.
    """

    def __init__(self, base, floor=0.5, ceiling=60.0, factor=2.0,
                 relax_after=30.0, on_throttle=None):
        self.base = max(float(base), 0.0)
        self.floor = floor
        self.ceiling = ceiling
        self.factor = factor
        self.relax_after = float(relax_after)
        self.interval = max(self.base, self.floor)
        self.on_throttle = on_throttle
        self._lock = threading.Lock()
        self._next = 0.0
        self._last_throttle = 0.0

    def wait(self):
        """Block until this request may be issued (globally rate-limited)."""
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            if delay > 0:
                time.sleep(delay)
            self._next = max(now, self._next) + self.interval

    def success(self):
        """Called after a clean request; slowly relaxes the interval."""
        with self._lock:
            now = time.monotonic()
            if self.interval > self.base and now - self._last_throttle >= self.relax_after:
                self.interval = max(self.base, self.interval / self.factor)
                self._last_throttle = now
                log(f"[pacer] {self.relax_after:.0f}s clean, relaxed to {self.interval:.1f}s")

    def ratelimited(self, retry_after=None):
        """Called on a 429; grows the interval, honoring Retry-After."""
        with self._lock:
            self._last_throttle = time.monotonic()
            grown = self.interval * self.factor
            if retry_after is not None:
                try:
                    grown = max(grown, float(retry_after) + 1.0)
                except (TypeError, ValueError):
                    pass
            self.interval = min(self.ceiling, grown)
            hint = f", retry-after {retry_after}s" if retry_after else ""
            log(f"[pacer] throttle detected{hint}, pacing at {self.interval:.1f}s")
        if self.on_throttle is not None:
            self.on_throttle()


def api_get(session, params, retries=10, pacer=None, timeout=90):
    """GET the API with pacing, Retry-After-aware 429 backoff, and retries."""
    if pacer is not None:
        pacer.wait()
    wait = 1.0
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=timeout)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(min(wait * 2, 30))
            continue
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if pacer is not None:
                pacer.ratelimited(retry_after)
            try:
                pause = float(retry_after) if retry_after else wait
            except (TypeError, ValueError):
                pause = wait
            log(f"[rate-limit] waiting {pause:.0f}s")
            time.sleep(min(max(pause, 1.0), 60))
            wait = min(wait * 2, 30)
            continue
        if r.status_code == 503:
            time.sleep(min(wait, 60))
            wait *= 2
            continue
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            if attempt == retries - 1:
                raise
            time.sleep(min(wait * 2, 30))
            continue
        if pacer is not None:
            pacer.success()
        return data
    raise RuntimeError("API request failed after retries")
