"""
A small client of a local Ollama server (https://ollama.com), which serves the open-weight language and vision models
(requirements AI-13, AI-16). Standard library only. All inference stays on this machine (AI-01): an address that is
not a loopback address is refused, and the system's proxy settings are ignored, so a request to Ollama never leaves the
machine.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator, Optional
from urllib.parse import urlparse

from ai import settings

LOOPBACK = ("127.0.0.1", "localhost", "::1")


class OllamaError(Exception):
    """reason is one of the ids in ai.envelope.REASONS."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def base_url() -> str:
    url = str(settings.number("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in LOOPBACK:
        raise OllamaError("failed", "OLLAMA_URL must be a local address: all AI runs on the server PC itself.")
    return url


def timeout_seconds() -> float:
    return float(settings.number("AI_OLLAMA_TIMEOUT_SECONDS", 180))


def probe_seconds() -> float:
    return float(settings.number("AI_OLLAMA_PROBE_SECONDS", 1.0))


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy, whatever the environment says


def _open(path: str, payload: Optional[dict[str, Any]], timeout: float):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base_url() + path, data=data, headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data is not None else "GET")
    try:
        return _opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:300]
        if error.code == 404:
            raise OllamaError("missingModel", body) from error
        raise OllamaError("failed", f"Ollama answered {error.code}: {body}") from error
    except (socket.timeout, TimeoutError) as error:
        raise OllamaError("timeout", "Ollama did not answer in time.") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise OllamaError("timeout", "Ollama did not answer in time.") from error
        raise OllamaError("serviceUnreachable", str(error.reason)) from error
    except OSError as error:
        raise OllamaError("serviceUnreachable", str(error)) from error


def _json(path: str, payload: Optional[dict[str, Any]] = None, timeout: Optional[float] = None) -> dict[str, Any]:
    with _open(path, payload, timeout or timeout_seconds()) as response:
        try:
            return json.loads(response.read().decode("utf-8"))
        except (socket.timeout, TimeoutError) as error:
            raise OllamaError("timeout", "Ollama did not answer in time.") from error
        except ValueError as error:
            raise OllamaError("invalidAnswer", "Ollama did not send JSON.") from error


def installed_models() -> list[dict[str, Any]]:
    """The models Ollama has downloaded: [{name, size}]. Raises OllamaError("serviceUnreachable") if it is not running."""

    try:
        data = _json("/api/tags", timeout=probe_seconds())
    except OllamaError as error:
        if error.reason == "timeout":  # a quick probe that gets no answer means nobody is listening
            raise OllamaError("serviceUnreachable", "Ollama does not answer.") from error
        raise
    return [{"name": m.get("name", ""), "size": m.get("size")} for m in data.get("models") or []]


def server_version() -> Optional[str]:
    try:
        return _json("/api/version", timeout=probe_seconds()).get("version")
    except OllamaError:
        return None


def has_model(name: str, installed: Optional[list[dict[str, Any]]] = None) -> bool:
    models = installed if installed is not None else installed_models()
    wanted = name if ":" in name else name + ":latest"
    return any(m["name"] in (name, wanted) for m in models)


def chat(model: str, messages: list[dict[str, Any]], schema: Optional[dict[str, Any]] = None,
         options: Optional[dict[str, Any]] = None, timeout: Optional[float] = None) -> str:
    """
    One answer of a model. A schema asks Ollama for structured output; the answer is still validated by the caller,
    because a model may not follow it. Temperature 0 and a fixed seed make an answer repeatable on the same model.
    """

    payload: dict[str, Any] = {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_predict": 700, **(options or {})},
    }
    if schema is not None:
        payload["format"] = schema
    data = _json("/api/chat", payload, timeout)
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise OllamaError("invalidAnswer", "The model sent an empty answer.")
    return content


def _lines(response) -> Iterator[dict[str, Any]]:
    for raw in response:
        raw = raw.strip()
        if raw:
            try:
                yield json.loads(raw.decode("utf-8"))
            except ValueError:
                continue


def pull(model: str, on_progress: Callable[[float, str], None]) -> None:
    """Downloads a model (an explicit action of the user, requirement AI-06), reporting the progress as it goes."""

    layers: dict[str, tuple[float, float]] = {}
    with _open("/api/pull", {"model": model, "stream": True}, 60) as response:
        for line in _lines(response):
            if line.get("error"):
                raise OllamaError("failed", str(line["error"])[:300])
            if line.get("digest") and line.get("total"):
                layers[line["digest"]] = (float(line.get("completed") or 0), float(line["total"]))
            total = sum(t for _, t in layers.values())
            fraction = sum(c for c, _ in layers.values()) / total if total else 0.0
            on_progress(min(0.99, fraction), str(line.get("status", "")))
