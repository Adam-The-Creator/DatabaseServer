"""
The built web application (DataAnalysis), served by this server (requirements TEC-08, SEC-09).

Why here: the game reaches the server through one public address (the ngrok URL). The web app served from the same address
needs no installation on the psychologists' PCs, and it is the same origin as the API, so CORS is not involved.

It lives under /app. The routes of the app (/sessions, /drawings/<id>, /admin, ...) have the same names as the routes of the
API, so at the root a reload of such a page would reach the API. DataAnalysis is built for this prefix (vite.config.ts, "base").

What is served: the files of the folder named by WEBAPP_PATH (config.py, or the environment variable of the same name), and
index.html for every address that is a route of the app (it has no file extension). A file that does not exist is a 404.
Without the setting the server does not serve the app, and nothing else changes.
"""

import logging
import mimetypes
import os
import posixpath
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

import config
from analysis import reports

log = logging.getLogger("webapp")

# The address the app is served at. DataAnalysis has to be built for the same prefix (vite.config.ts, "base").
PREFIX = "/app"

_root: Optional[Path] = None
_problem: Optional[str] = None

# Python takes the file types from the Windows registry, where a program may have set .js to "text/plain" - which a browser
# refuses for a script when the answer says "nosniff" (it does, see throttle.py). The files of a web app get fixed types.
_TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".html": "text/html",
          ".json": "application/json", ".map": "application/json", ".svg": "image/svg+xml", ".wasm": "application/wasm",
          ".woff": "font/woff", ".woff2": "font/woff2", ".png": "image/png", ".ico": "image/x-icon", ".txt": "text/plain"}


def served_from() -> Optional[Path]:
    """The folder that is served, or None when the server does not serve the web app."""

    return _root


def problem() -> Optional[str]:
    """What is wrong with the folder that is served (it is served anyway), or None."""

    return _problem


def build_problem(root: Path) -> Optional[str]:
    """
    A build made for another address than PREFIX (an old `pnpm build`, or the files of `pnpm dev`) asks the browser for its
    files at /assets/... instead of /app/assets/..., which this server answers with 404: the page stays blank. Says so.
    """

    try:
        page = (root / "index.html").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if PREFIX + "/" in page:
        return None
    return (f"The web app in {root} was not built for the address {PREFIX}/ (its index.html asks for /assets/..., not "
            f"{PREFIX}/assets/...), so the browser will show a blank page. Build it again with `pnpm build` in DataAnalysis.")


def folder() -> Optional[Path]:
    """The folder named by the setting, if it holds a built web app."""

    configured = os.environ.get(config.WEBAPP_PATH_ENV_VAR) or getattr(config, "WEBAPP_PATH", None)
    if not configured:
        return None
    path = Path(configured).resolve()
    if not (path / "index.html").is_file():
        log.warning("The web app folder %s has no index.html, so the web app is not served.", path)
        return None
    return path


def content_security_policy() -> str:
    """
    The policy of the app (SEC-09). Scripts only from the app's own files; the only inline script that is allowed is the
    one of the report pages (by its hash), because a report is opened in a tab made from a blob, which inherits this policy.
    The app talks to its own server only. Styles may be inline for the same reason: the report pages style themselves
    (the app itself does not need it).
    """

    scripts = " ".join(["'self'", *reports.inline_script_hashes()])
    return "; ".join([
        "default-src 'self'",
        f"script-src {scripts}",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "connect-src 'self'",
        "worker-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


class _AppFiles(StaticFiles):
    """The files of the app, with index.html for its routes, cache rules and the policy."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory=directory)
        self._policy = content_security_policy()

    async def get_response(self, path: str, scope: Scope) -> Response:
        # Starlette normalises the path with os.path: the root of the app arrives as "." and, on Windows, with backslashes.
        relative = "" if path == "." else path.replace("\\", "/")
        try:
            response = await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code != 404 or ".." in relative.split("/") or "." in posixpath.basename(relative):
                raise  # a file that does not exist (or an attempt to leave the folder), not a route of the app
            response = await super().get_response("index.html", scope)
        # The files under assets/ carry a hash of their content in their names, so they never change. index.html has to be
        # asked for again, otherwise an update of the app would not be seen.
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if relative.startswith("assets/") else "no-cache"
        response.headers["Content-Security-Policy"] = self._policy
        return response


def mount(app: FastAPI) -> Optional[Path]:
    """Serves the web app at PREFIX when a folder is configured. Returns the folder, or None."""

    global _root, _problem
    root = folder()
    _root = root
    _problem = build_problem(root) if root is not None else None
    if _problem:
        log.warning(_problem)
    if root is not None:
        for extension, media_type in _TYPES.items():
            mimetypes.add_type(media_type, extension)
        # Compression for the app's own files only (about 1.7 MB of scripts): the API's answers are not touched.
        app.mount(PREFIX, GZipMiddleware(_AppFiles(root), minimum_size=1024), name="webapp")
    return root


def is_browser_page_request(request: Request) -> bool:
    """
    A browser that was sent to an address (it asks for text/html), as opposed to the game, a script or a health check
    (UrlConfigurator, the launcher), which accept anything or send no Accept header at all.
    """

    return "text/html" in request.headers.get("accept", "")
