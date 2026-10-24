"""
Protection of the API against floods and guessing (requirements SEC-02, SEC-03, SEC-04, SEC-09, SEC-11):

- token bucket limits per client address and per account, with stricter ones for login and sign-up,
- blocking of repeated failed logins (per username and address together, so nobody can lock another user out),
- a size cap for request bodies, and a few security headers on every answer.

Everything is in memory, which is enough for the one server process on the central PC. The limits of the feedback
submissions (feedback/store.py) live in the database instead, so they survive a restart.
"""

import ipaddress
import json
import logging
import math
import threading
import time
from typing import Callable, Optional

import config

log = logging.getLogger("protection")


def human_wait(seconds: float) -> str:
    """'45 seconds', '3 minutes', '1 hour 5 minutes': how long a refused caller has to wait, rounded up."""

    seconds = max(1, math.ceil(seconds))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = math.ceil(seconds / 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, rest = divmod(minutes, 60)
    text = f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{text} {rest} minute{'s' if rest != 1 else ''}" if rest else text


class TokenBuckets:
    """One token bucket per key: it holds up to `capacity` tokens and gets `refill_per_second` of them back every second."""

    MAX_KEYS = 10_000

    def __init__(self, capacity: float, refill_per_second: float, clock: Callable[[], float] = time.monotonic):
        self.capacity = capacity
        self.refill = refill_per_second
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, time of the last change)
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """Takes one token. Returns 0 when there was one, otherwise the seconds until there will be one."""

        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                self._trim(now)
                return 0.0
            self._buckets[key] = (tokens, now)
            return (1 - tokens) / self.refill

    def _trim(self, now: float) -> None:
        """A bucket that has had time to fill up again is the same as no bucket, so it can be forgotten."""

        if len(self._buckets) <= self.MAX_KEYS:
            return
        full_after = self.capacity / self.refill
        self._buckets = {key: value for key, value in self._buckets.items() if now - value[1] < full_after}
        if len(self._buckets) > self.MAX_KEYS:  # a flood from more addresses than that: better to forget than to grow
            self._buckets.clear()


class LoginGuard:
    """
    Slows down password guessing: after `max_failures` failed logins for the same username from the same address the
    pair is blocked for `block_seconds`. Unknown usernames count like known ones, so the block does not reveal which
    usernames exist, and it is per pair, so somebody guessing from elsewhere cannot lock the real user out.
    """

    MAX_KEYS = 10_000

    def __init__(self, max_failures: int, block_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_failures = max_failures
        self.block_seconds = block_seconds
        self._clock = clock
        self._failures: dict[tuple[str, str], list[float]] = {}
        self._blocked_until: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(username: str, ip: str) -> tuple[str, str]:
        return username.casefold()[:128], ip

    def blocked_for(self, username: str, ip: str) -> float:
        """Seconds the pair is still blocked, or 0."""

        key, now = self._key(username, ip), self._clock()
        with self._lock:
            until = self._blocked_until.get(key)
            if until is None:
                return 0.0
            if until <= now:
                del self._blocked_until[key]
                return 0.0
            return until - now

    def failure(self, username: str, ip: str) -> None:
        key, now = self._key(username, ip), self._clock()
        with self._lock:
            recent = [t for t in self._failures.get(key, []) if now - t < self.block_seconds]
            recent.append(now)
            if len(recent) >= self.max_failures:
                self._blocked_until[key] = now + self.block_seconds
                self._failures.pop(key, None)
                log.warning("Login blocked for %s seconds after %d failed attempts from %s", int(self.block_seconds),
                            len(recent), ip)
            else:
                self._failures[key] = recent
            if len(self._failures) + len(self._blocked_until) > self.MAX_KEYS:
                self._failures.clear()
                self._blocked_until = {k: v for k, v in self._blocked_until.items() if v > now}

    def success(self, username: str, ip: str) -> None:
        with self._lock:
            self._failures.pop(self._key(username, ip), None)


# The shared instances, configured in config.py
ip_limiter = TokenBuckets(config.THROTTLE_IP_CAPACITY, config.THROTTLE_IP_REFILL_PER_SECOND)
account_limiter = TokenBuckets(config.THROTTLE_ACCOUNT_CAPACITY, config.THROTTLE_ACCOUNT_REFILL_PER_SECOND)
auth_limiter = TokenBuckets(config.THROTTLE_AUTH_CAPACITY, config.THROTTLE_AUTH_REFILL_PER_SECOND)
login_guard = LoginGuard(config.LOGIN_MAX_FAILURES, config.LOGIN_BLOCK_SECONDS)


# --- Client address ---

def _header(scope: dict, name: bytes) -> Optional[str]:
    for key, value in scope.get("headers") or []:
        if key == name:
            return value.decode("latin-1")
    return None


def client_ip(scope: dict) -> str:
    """
    The address of the client. Behind a trusted proxy (ngrok forwards from this PC) it is the address that proxy wrote
    into X-Forwarded-For, taken from the right: the first one that is not a proxy itself. From anybody else the header is
    ignored, so a client cannot invent an address.
    """

    peer = (scope.get("client") or ("unknown", 0))[0]
    if peer not in config.TRUSTED_PROXIES:
        return peer
    forwarded = _header(scope, b"x-forwarded-for")
    if not forwarded:
        return peer
    for hop in reversed([h.strip() for h in forwarded.split(",") if h.strip()]):
        if hop in config.TRUSTED_PROXIES:
            continue
        try:
            return str(ipaddress.ip_address(hop))
        except ValueError:
            return peer  # not an address: the header is not believed at all
    return peer


# --- Middleware ---

_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
]

# Path prefix -> the biggest request body that is accepted (bytes)
_BODY_LIMITS = {
    "/auth": config.AUTH_MAX_BODY_BYTES,
    "/feedback": config.FEEDBACK_MAX_BODY_BYTES,
    "/admin": config.FEEDBACK_MAX_BODY_BYTES,
}
_NO_STORE_PREFIXES = ("/feedback", "/admin")


def _prefix(path: str, prefixes) -> Optional[str]:
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "/"):
            return prefix
    return None


class ProtectionMiddleware:
    """Pure ASGI middleware (no extra dependency): throttling per address, body size cap, security headers."""

    def __init__(self, app):
        self.app = app
        self._logged: dict[str, float] = {}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path, ip = scope["path"], client_ip(scope)
        no_store = _prefix(path, _NO_STORE_PREFIXES) is not None

        async def guarded_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or []) + _SECURITY_HEADERS
                if no_store:
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        wait = ip_limiter.take(ip)
        if not wait and _prefix(path, ("/auth",)):
            wait = auth_limiter.take(ip)
        if wait:
            self._log_throttled(ip, scope["method"], path)
            await self._reject(guarded_send, 429, f"Too many requests. Try again in {human_wait(wait)}.", wait)
            return

        limit_prefix = _prefix(path, _BODY_LIMITS)
        if limit_prefix is not None and scope["method"] in ("POST", "PUT", "PATCH"):
            length = _header(scope, b"content-length")
            if length is None and _header(scope, b"transfer-encoding") is not None:
                # A body of unknown size cannot be checked before it is read
                await self._reject(guarded_send, 411, "A Content-Length header is required.")
                return
            if length is not None:
                try:
                    too_big = int(length) > _BODY_LIMITS[limit_prefix]
                except ValueError:
                    too_big = True
                if too_big:
                    await self._reject(guarded_send, 413, "The request is too large.")
                    return

        await self.app(scope, receive, guarded_send)

    def _log_throttled(self, ip: str, method: str, path: str) -> None:
        """One log line per address and minute, otherwise a flood would flood the log too."""

        now = time.monotonic()
        if now - self._logged.get(ip, -60.0) >= 60:
            if len(self._logged) > 1000:
                self._logged.clear()
            self._logged[ip] = now
            log.warning("Throttled %s %s from %s", method, path, ip)

    @staticmethod
    async def _reject(send, status: int, detail: str, retry_after: Optional[float] = None) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
        if retry_after:
            headers.append((b"retry-after", str(math.ceil(retry_after)).encode()))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
