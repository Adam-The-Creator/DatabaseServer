import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import RedirectResponse

import config
import server
import throttle
import webapp
from analysis import reports
from analysis.metrics import analyze_drawing
from tests.fixtures import make_drawing, make_line, straight

INDEX = '<!doctype html><title>app</title><script type="module" src="/app/assets/index-a1b2c3.js"></script><div id=root></div>'
OLD_BUILD_INDEX = '<!doctype html><script type="module" src="/assets/index-Df5H6nh2.js"></script><link rel="icon" href="/favicon.svg">'
SCRIPT = "console.log('x');" * 200  # bigger than the compression threshold


def policy_directive(policy: str, name: str) -> str:
    return next(part for part in policy.split("; ") if part.startswith(name + " "))


class WebAppTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dist"
        (self.root / "assets").mkdir(parents=True)
        (self.root / "index.html").write_text(INDEX, encoding="utf-8")
        (self.root / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        (self.root / "assets" / "index-a1b2c3.js").write_text(SCRIPT, encoding="utf-8")
        (self.root / "assets" / "index-a1b2c3.css").write_text("body{margin:0}" * 200, encoding="utf-8")
        (Path(self.temp.name) / "secret.txt").write_text("outside of the web app", encoding="utf-8")
        self.addCleanup(setattr, webapp, "_root", None)
        self.addCleanup(setattr, webapp, "_problem", None)
        patcher = mock.patch.object(throttle, "ip_limiter", throttle.TokenBuckets(10_000, 10_000))
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_app(self, path=None):
        app = FastAPI()

        @app.get("/sessions")
        def api_sessions():
            return {"api": True}

        env = {config.WEBAPP_PATH_ENV_VAR: str(self.root if path is None else path)}
        with mock.patch.dict(os.environ, env):
            webapp.mount(app)
        return throttle.ProtectionMiddleware(app)  # as in server.py: the security headers are added around it

    @staticmethod
    def call(app, path, method="GET", headers=()):
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
                 "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "headers": [(k.lower().encode(), v.encode()) for k, v in headers], "client": ("203.0.113.9", 1),
                 "server": ("testserver", 80)}
        asyncio.run(app(scope, receive, send))
        start = sent[0]
        return (start["status"], {k.decode(): v.decode() for k, v in start["headers"]},
                b"".join(m.get("body", b"") for m in sent[1:]))

    # --- what is served

    def test_the_app_is_served_under_app_with_its_policy_and_the_security_headers(self):
        status, headers, body = self.call(self.make_app(), "/app/")
        self.assertEqual(status, 200)
        self.assertEqual(body.decode(), INDEX)
        self.assertTrue(headers["content-type"].startswith("text/html"))
        self.assertEqual(headers["cache-control"], "no-cache")  # an update of the app has to be seen
        self.assertEqual(headers["content-security-policy"], webapp.content_security_policy())
        self.assertEqual(headers["x-content-type-options"], "nosniff")  # from the protection middleware
        self.assertEqual(headers["x-frame-options"], "DENY")
        self.assertEqual(headers["referrer-policy"], "no-referrer")

    def test_app_without_a_slash_goes_to_the_slash(self):
        status, headers, _ = self.call(self.make_app(), "/app")
        self.assertEqual(status, 307)
        self.assertEqual(headers["location"].rsplit("/app", 1)[1], "/")

    def test_files_have_their_type_and_the_hashed_ones_are_cached_for_good(self):
        app = self.make_app()
        status, headers, body = self.call(app, "/app/assets/index-a1b2c3.js")
        self.assertEqual((status, headers["content-type"].split(";")[0]), (200, "text/javascript"))
        self.assertEqual(headers["cache-control"], "public, max-age=31536000, immutable")
        self.assertEqual(body.decode(), SCRIPT)
        self.assertEqual(self.call(app, "/app/assets/index-a1b2c3.css")[1]["content-type"].split(";")[0], "text/css")
        status, headers, _ = self.call(app, "/app/favicon.svg")
        self.assertEqual((status, headers["content-type"].split(";")[0], headers["cache-control"]), (200, "image/svg+xml", "no-cache"))

    def test_a_script_is_javascript_even_when_the_registry_says_text_plain(self):
        with mock.patch.dict(mimetypes.types_map, {".js": "text/plain"}):
            status, headers, _ = self.call(self.make_app(), "/app/assets/index-a1b2c3.js")
        self.assertEqual(headers["content-type"].split(";")[0], "text/javascript")  # "nosniff" would refuse text/plain

    def test_a_route_of_the_app_gets_index_html(self):
        app = self.make_app()
        for route in ("/app/sessions", "/app/sessions/6f1b7e52", "/app/drawings/abc-1/", "/app/admin/feedback", "/app/my/profile"):
            status, headers, body = self.call(app, route)
            self.assertEqual((status, body.decode()), (200, INDEX), route)
            self.assertIn("content-security-policy", headers)

    def test_a_missing_file_is_a_404_not_the_app(self):
        app = self.make_app()
        for missing in ("/app/assets/nothing.js", "/app/robots.txt", "/app/assets/index-ffffff.css"):
            self.assertEqual(self.call(app, missing)[0], 404, missing)

    def test_nothing_outside_the_folder_is_served(self):
        app = self.make_app()
        for attack in ("/app/../secret.txt", "/app/assets/../../secret.txt", "/app/..%2fsecret.txt", "/app/%2e%2e/secret.txt",
                       "/app/..\\secret.txt", "/app/assets/..\\..\\secret.txt", "/app/C:/Windows/win.ini"):
            status, _, body = self.call(app, attack)
            self.assertNotIn(b"outside of the web app", body, attack)
            self.assertNotIn(b"[fonts]", body.lower(), attack)
        # an attempt to leave the folder is refused, also when it has no extension (which would make it "a route of the app")
        self.assertEqual(self.call(app, "/app/../secret")[0], 404)
        self.assertEqual(self.call(app, "/app/assets/../../secret")[0], 404)

    def test_only_get_and_head(self):
        app = self.make_app()
        self.assertEqual(self.call(app, "/app/", "HEAD")[0], 200)
        for method in ("POST", "PUT", "DELETE"):
            self.assertEqual(self.call(app, "/app/", method)[0], 405, method)

    def test_conditional_requests_get_304_with_the_policy(self):
        app = self.make_app()
        etag = self.call(app, "/app/")[1]["etag"]
        status, headers, body = self.call(app, "/app/", headers=[("If-None-Match", etag)])
        self.assertEqual((status, body), (304, b""))
        self.assertEqual(headers["content-security-policy"], webapp.content_security_policy())

    def test_only_the_files_of_the_app_are_compressed(self):
        app = self.make_app()
        status, headers, body = self.call(app, "/app/assets/index-a1b2c3.js", headers=[("Accept-Encoding", "gzip")])
        self.assertEqual(headers.get("content-encoding"), "gzip")
        self.assertLess(len(body), len(SCRIPT) // 4)
        self.assertNotIn("content-encoding", self.call(app, "/app/assets/index-a1b2c3.js")[1])  # the client did not ask
        self.assertNotIn("content-encoding", self.call(app, "/sessions", headers=[("Accept-Encoding", "gzip")])[1])  # the API is not touched

    def test_the_api_routes_are_not_shadowed(self):
        status, headers, body = self.call(self.make_app(), "/sessions")
        self.assertEqual((status, json.loads(body)), (200, {"api": True}))
        self.assertNotIn("content-security-policy", headers)  # the policy is for the app, not for the API answers

    # --- when it is not served

    def test_without_a_setting_nothing_is_served(self):
        app = FastAPI()
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(config, "WEBAPP_PATH", None):
            os.environ.pop(config.WEBAPP_PATH_ENV_VAR, None)
            self.assertIsNone(webapp.mount(app))
        self.assertIsNone(webapp.served_from())
        self.assertEqual(self.call(throttle.ProtectionMiddleware(app), "/app/")[0], 404)

    def test_a_folder_without_index_html_is_not_served(self):
        empty = Path(self.temp.name) / "empty"
        empty.mkdir()
        with mock.patch.dict(os.environ, {config.WEBAPP_PATH_ENV_VAR: str(empty)}), self.assertLogs("webapp", level="WARNING"):
            self.assertIsNone(webapp.mount(FastAPI()))
        self.assertIsNone(webapp.served_from())

    def test_a_build_for_another_address_is_served_but_the_server_says_what_is_wrong(self):
        """A build made for "/" (an old `pnpm build`) asks for /assets/..., which the server cannot give: a blank page."""
        (self.root / "index.html").write_text(OLD_BUILD_INDEX, encoding="utf-8")
        with self.assertLogs("webapp", level="WARNING") as logged:
            app = self.make_app()
        self.assertIn("not built for the address /app/", logged.output[0])
        self.assertIn("pnpm build", webapp.problem())
        self.assertEqual(self.call(app, "/app/")[0], 200)  # still served: the message is the help, not a refusal
        self.assertEqual(self.call(app, "/assets/index-Df5H6nh2.js")[0], 404)  # which is what the browser then gets

    def test_a_build_for_the_right_address_has_no_problem(self):
        self.make_app()
        self.assertIsNone(webapp.problem())
        self.assertIsNone(webapp.build_problem(self.root))

    def test_the_config_setting_works_without_the_environment_variable(self):
        with mock.patch.dict(os.environ, {}), mock.patch.object(config, "WEBAPP_PATH", str(self.root)):
            os.environ.pop(config.WEBAPP_PATH_ENV_VAR, None)
            self.assertEqual(webapp.mount(FastAPI()), self.root.resolve())

    # --- the root address

    @staticmethod
    def root_request(accept=None):
        headers = [] if accept is None else [(b"accept", accept.encode())]
        return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})

    def test_a_browser_at_the_root_is_sent_to_the_app_but_scripts_and_health_checks_are_not(self):
        browser = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        online = {"status": "online", "message": "VR Drawing Database Server is running"}
        with mock.patch.object(webapp, "_root", self.root):
            answer = server.health_check(self.root_request(browser))
            self.assertIsInstance(answer, RedirectResponse)
            self.assertEqual((answer.status_code, answer.headers["location"]), (307, "/app/"))
            for accept in (None, "*/*", "application/json"):  # UrlConfigurator, the launcher, the game, curl
                self.assertEqual(server.health_check(self.root_request(accept)), online, accept)
        with mock.patch.object(webapp, "_root", None):  # not served: a browser gets the health answer as before
            self.assertEqual(server.health_check(self.root_request(browser)), online)

    # --- the policy (SEC-09)

    def test_policy_allows_no_script_but_the_apps_own_files_and_the_report_viewer(self):
        policy = webapp.content_security_policy()
        scripts = policy_directive(policy, "script-src").split()[1:]
        self.assertEqual(scripts[0], "'self'")
        self.assertTrue(all(s.startswith("'sha256-") for s in scripts[1:]) and len(scripts) >= 2, scripts)
        for forbidden in ("'unsafe-inline'", "'unsafe-eval'", "http:", "https:", "*", "data:", "blob:"):
            self.assertNotIn(forbidden, scripts)
        self.assertEqual(policy_directive(policy, "default-src"), "default-src 'self'")
        self.assertEqual(policy_directive(policy, "connect-src"), "connect-src 'self'")  # it talks to its own server only
        self.assertEqual(policy_directive(policy, "frame-ancestors"), "frame-ancestors 'none'")
        self.assertEqual(policy_directive(policy, "object-src"), "object-src 'none'")
        self.assertEqual(policy_directive(policy, "base-uri"), "base-uri 'self'")
        self.assertIn("blob:", policy_directive(policy, "img-src"))  # thumbnails and rendered views are blobs

    def test_every_inline_script_of_a_report_is_covered_by_the_policy(self):
        """A report opens from a blob and inherits the policy: a script that is not listed would not run."""
        drawing = make_drawing([make_line("a", straight()), make_line("b", straight(), color=(40, 60, 220))])
        player = {"id": "p1", "name": "N", "age": 9, "gender": "F", "dominantHand": "Right"}
        html = reports.drawing_report(drawing, analyze_drawing(drawing), [], "T", player, None)
        scripts = re.findall(r"<script(?![^>]*\btype=)[^>]*>(.*?)</script>", html, re.S)  # not the JSON data block
        self.assertTrue(scripts, "the report has its viewer script")
        policy = webapp.content_security_policy()
        for script in scripts:
            digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
            self.assertIn(f"'sha256-{digest}'", policy)
        self.assertEqual(reports.inline_script_hashes(), [f"'sha256-{base64.b64encode(hashlib.sha256(reports._VIEWER_SCRIPT.encode('utf-8')).digest()).decode('ascii')}'"])

    def test_the_policy_is_in_the_header_of_every_answer_of_the_app(self):
        app = self.make_app()
        for path in ("/app/", "/app/assets/index-a1b2c3.js", "/app/favicon.svg", "/app/sessions/1"):
            self.assertEqual(self.call(app, path)[1]["content-security-policy"], webapp.content_security_policy(), path)


if __name__ == "__main__":
    unittest.main()
