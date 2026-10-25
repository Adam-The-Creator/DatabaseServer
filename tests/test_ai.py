import json
import struct
import tempfile
import threading
import time
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi import HTTPException

import config
import security
from analysis import reports, service as data
from analysis.metrics import analyze_drawing, metric_values
from analysis.store import AnalysisStore
from ai import (
    answers, clustering, cohort_ml, envelope, features, fuzzy, insights, jobs, llm, ollama, registry, reporting,
    semantic, service, settings, vlm,
)
from ai.render import VIEWS, render_view
from ai.store import AiStore
from routers import ai as ai_router
from tests.fixtures import T0, make_drawing, make_line, straight
from tests.test_feedback import DatabaseTestCase, cid, credentials, request_for

CALM = {"x0": 0.0}


def synthetic(drawing_id: str, strokes: int = 5, length: float = 1.0, step_ms: int = 100, game_type: int = 2,
              owner: str = "p1", session: str = "s1") -> dict:
    """A drawing with `strokes` parallel straight strokes (known counts and lengths)."""

    lines = [make_line(f"{drawing_id}-{n}", straight(length, 21, y=1.0 + 0.1 * n, z=0.0), start_ms=T0 + n * 3000,
                       step_ms=step_ms, color=(255, 0, 0) if n % 2 else (0, 0, 255), author=owner)
             for n in range(strokes)]
    document = make_drawing(lines, drawing_id=drawing_id, owner=owner, game_type=game_type)
    document["metadata"]["sessionID"] = session
    return document


def cohort_entries(count: int = 13) -> list[dict]:
    """A group of similar drawings; the metrics come from the real analyzer."""

    entries = []
    for n in range(count):
        document = synthetic(f"d{n:02d}", strokes=4 + n % 3, length=0.8 + 0.1 * (n % 4), step_ms=100 + 10 * (n % 3))
        entries.append({"id": f"d{n:02d}", "metrics": metric_values(analyze_drawing(document))})
    return entries


class FeaturesTest(unittest.TestCase):
    def test_percentile_interpolates_like_numpy(self):
        values = sorted([1.0, 2.0, 4.0, 8.0, 16.0])
        for q in (0, 10, 25, 50, 90, 100):
            self.assertAlmostEqual(features.percentile(values, q), float(np.percentile(values, q)))

    def test_robust_scale_known_answer(self):
        # median 3, deviations 2 1 0 1 2 -> MAD 1 -> scale 1 / 0.6745
        centre, scale, tied = features.robust_scale([1, 2, 3, 4, 5])
        self.assertEqual(centre, 3)
        self.assertAlmostEqual(scale, 1 / 0.6745)
        self.assertFalse(tied)

    def test_robust_scale_falls_back_to_the_mean_deviation_when_the_mad_is_zero(self):
        centre, scale, tied = features.robust_scale([1, 1, 1, 1, 6])
        self.assertEqual(centre, 1)
        self.assertAlmostEqual(scale, (5 / 5) / 1.253314)
        self.assertTrue(tied)  # more than half of the values are equal: the scale is shaky
        self.assertEqual(features.robust_scale([2, 2, 2])[1], 0.0)

    def test_constant_and_sparse_metrics_are_not_used(self):
        rows = [{"strokesDrawn": 5.0 + n, "totalLength": 3.0, "meanSpeed": None} for n in range(6)]
        self.assertEqual(features.usable_metrics(rows), ["strokesDrawn"])

    def test_a_missing_value_is_imputed_with_the_median_and_marked(self):
        rows = [{"strokesDrawn": v} for v in (3.0, 4.0, 5.0, 6.0, None)]
        metrics, z, imputed, tied = features.z_matrix(rows)
        self.assertEqual((metrics, tied), (["strokesDrawn"], [False]))
        self.assertEqual((z[4][0], imputed[4][0]), (0.0, True))
        self.assertFalse(imputed[0][0])

    def test_a_tiny_difference_in_a_tight_group_is_not_far_away(self):
        # nine drawings with 10.0 .. 10.8 strokes and one with 11.5: about 10 % more, but the group is very tight
        rows = [{"strokesDrawn": 10.0 + n * 0.1} for n in range(9)] + [{"strokesDrawn": 11.5}]
        _, z, _, _ = features.z_matrix(rows)
        self.assertLess(abs(z[9][0]), 2.0)  # without the minimum spread this would be far beyond 3.5
        # a real outlier is still found
        rows[9] = {"strokesDrawn": 40.0}
        self.assertGreater(features.z_matrix(rows)[1][9][0], features.Z_LIMIT)

    def test_z_scores_are_clipped(self):
        rows = [{"strokesDrawn": v} for v in (3.0, 3.1, 3.2, 3.3, 500.0)]
        _, z, _, _ = features.z_matrix(rows)
        self.assertEqual(z[4][0], features.Z_CLIP)


class CohortMlTest(unittest.TestCase):
    def test_a_planted_outlier_is_the_only_atypical_drawing(self):
        entries = cohort_entries(13)
        big = synthetic("big", strokes=40, length=4.0)
        entries.append({"id": "big", "metrics": metric_values(analyze_drawing(big))})
        result = cohort_ml.analyze_cohort(entries)
        flagged = [row["id"] for row in result["drawings"] if row["atypical"]]
        self.assertEqual(flagged, ["big"])
        big_row = next(row for row in result["drawings"] if row["id"] == "big")
        self.assertEqual(big_row["isolation"]["rank"], 1)  # the forest agrees (14 drawings are enough for it)
        self.assertEqual(big_row["isolation"]["of"], 14)
        self.assertTrue(big_row["notable"] and big_row["notable"][0]["direction"] == "high")

    def test_a_metric_with_no_spread_in_the_group_describes_but_never_makes_a_drawing_atypical(self):
        # nine drawings draw 0 % with the left hand, one draws 100 %: the scale of that metric is shaky (tied values)
        entries = [{"id": f"d{n}", "metrics": {"strokesDrawn": 5.0 + (n % 3), "leftHandShare": 100.0 if n == 9 else 0.0}}
                   for n in range(10)]
        row = next(r for r in cohort_ml.analyze_cohort(entries)["drawings"] if r["id"] == "d9")
        hand = next(n for n in row["notable"] if n["metric"] == "leftHandShare")
        self.assertEqual((hand["direction"], hand["beyondLimit"]), ("high", False))
        self.assertFalse(row["atypical"])

    def test_the_map_has_a_point_per_drawing_and_is_reproducible(self):
        entries = cohort_entries(10)
        first, second = cohort_ml.analyze_cohort(entries), cohort_ml.analyze_cohort(list(reversed(entries)))
        self.assertEqual(len(first["map"]["points"]), 10)
        self.assertEqual(first["map"], second["map"])  # the order of the input does not matter
        self.assertTrue(0 < first["map"]["axes"][0]["explained"] <= 1)
        loadings = first["map"]["axes"][0]["top"]
        self.assertGreater(max(loadings, key=lambda l: abs(l["loading"]))["loading"], 0)  # the sign is normalised

    def test_similar_drawings_are_the_nearest_ones(self):
        entries = [{"id": f"a{n}", "metrics": {"strokesDrawn": 5.0 + n * 0.01, "totalLength": 5.0 + n * 0.02, "meanSpeed": 0.5}}
                   for n in range(5)]
        entries += [{"id": f"b{n}", "metrics": {"strokesDrawn": 50.0 + n * 0.5, "totalLength": 60.0 + n, "meanSpeed": 0.5 + n * 0.1}}
                    for n in range(5)]
        result = cohort_ml.analyze_cohort(entries, "a0")
        self.assertEqual([d["id"] for d in result["drawings"]], ["a0"])
        self.assertEqual({s["id"] for s in result["drawings"][0]["similar"]}, {"a1", "a2", "a3"})

    def test_the_isolation_forest_needs_enough_drawings(self):
        self.assertIsNone(cohort_ml.analyze_cohort(cohort_entries(8), "d00")["drawings"][0]["isolation"])
        self.assertIsNotNone(cohort_ml.analyze_cohort(cohort_entries(13), "d00")["drawings"][0]["isolation"])


class ClusteringTest(unittest.TestCase):
    @staticmethod
    def drawing(groups: list[tuple[float, int]], decimation=None) -> dict:
        """groups: (x offset in metres, start time in ms) of a group of three parallel strokes."""

        lines = []
        for g, (x0, start) in enumerate(groups):
            for n in range(3):
                lines.append(make_line(f"g{g}s{n}", straight(0.5, 21, y=1.0, z=0.03 * n, x0=x0), start_ms=T0 + start + n * 1000))
        return make_drawing(lines)

    def test_two_far_apart_groups_and_a_stray_tap(self):
        document = self.drawing([(0.0, 0), (2.0, 0)])
        document["lines"].append(make_line("tap", [(5.0, 1.0, 0.0)], start_ms=T0))
        result = clustering.cluster_strokes(document)
        groups = [sorted(g["strokeIds"]) for g in result["groups"]]
        self.assertEqual(sorted(groups), [["g0s0", "g0s1", "g0s2"], ["g1s0", "g1s1", "g1s2"]])
        self.assertEqual(result["unclustered"], ["tap"])
        self.assertTrue(result["timeUsed"])
        group = result["groups"][0]
        self.assertEqual(group["strokeCount"], 3)
        self.assertAlmostEqual(group["size"][0], 0.5, places=2)  # 0.5 m wide
        self.assertEqual(group["colours"][0], {"hex": "#ff0000", "share": 1.0})
        self.assertAlmostEqual(sum(g["inkShare"] for g in result["groups"]), 1.0 - 0.0, places=2)

    def test_same_place_but_far_apart_in_time_are_two_groups(self):
        later = self.drawing([(0.0, 0), (0.0, 300_000)])  # five minutes later
        self.assertEqual(len(clustering.cluster_strokes(later)["groups"]), 2)
        sooner = self.drawing([(0.0, 0), (0.0, 5_000)])
        self.assertEqual(len(clustering.cluster_strokes(sooner)["groups"]), 1)

    def test_without_time_data_only_the_place_counts(self):
        lines = [make_line(f"s{n}", straight(0.5, 21, z=0.03 * n), utc=False) for n in range(3)]
        for line in lines:
            for point in line["points"]:
                point["timestamp"] = None
        result = clustering.cluster_strokes(make_drawing(lines))
        self.assertFalse(result["timeUsed"])
        self.assertEqual(len(result["groups"]), 1)

    def test_a_drawing_without_strokes(self):
        result = clustering.cluster_strokes(make_drawing([]))
        self.assertEqual((result["groups"], result["unclustered"]), ([], []))

    def test_resampling_keeps_the_ends(self):
        points = [{"x": i * 0.001, "y": 0.0, "z": 0.0} for i in range(101)]
        kept = clustering.resample(points)
        self.assertEqual((kept[0], kept[-1]), (0, 100))
        self.assertLess(len(kept), 15)


class FuzzyTest(unittest.TestCase):
    def test_membership_degrees_at_known_points(self):
        degrees = lambda x: fuzzy.membership_degrees(x, 10, 20, 40)  # noqa: E731
        self.assertEqual(degrees(5), {"low": 1.0, "medium": 0.0, "high": 0.0})
        self.assertEqual(degrees(20), {"low": 0.0, "medium": 1.0, "high": 0.0})
        self.assertEqual(degrees(50), {"low": 0.0, "medium": 0.0, "high": 1.0})
        self.assertEqual(degrees(15), {"low": 0.5, "medium": 0.5, "high": 0.0})
        self.assertEqual(degrees(30), {"low": 0.0, "medium": 0.5, "high": 0.5})

    def test_conditions(self):
        self.assertEqual(fuzzy.parse_condition("a is high and b is not low or c is medium"),
                         [[("a", "high", False), ("b", "low", True)], [("c", "medium", False)]])
        with self.assertRaises(ValueError):
            fuzzy.parse_condition("a is huge")

    def test_the_rule_base_that_comes_with_the_server_is_valid_and_documented(self):
        base = fuzzy.load_rule_base()
        self.assertGreaterEqual(len(base["indicators"]), 5)
        self.assertEqual(base["status"], "proposal")
        for indicator in base["indicators"]:
            self.assertTrue(indicator["rationale"])
            for rule in indicator["rules"]:
                self.assertTrue(rule["rationale"], rule["id"])

    @staticmethod
    def tiny_base() -> dict:
        return {"name": "t", "version": "1", "status": "proposal", "reviewed": "",
                "fallbackRanges": {"strokesDrawn": [2, 8, 30], "totalLength": [1, 5, 20]},
                "indicators": [{"id": "amount", "name": "Amount", "rationale": "r",
                                "levels": {"low": "little", "medium": "some", "high": "a lot"},
                                "rules": [
                                    {"id": "r1", "if": "strokesDrawn is high and totalLength is high", "then": "high", "weight": 1.0, "rationale": "x"},
                                    {"id": "r2", "if": "strokesDrawn is high or totalLength is high", "then": "high", "weight": 0.5, "rationale": "x"},
                                    {"id": "r3", "if": "strokesDrawn is low", "then": "low", "weight": 1.0, "rationale": "x"}]}]}

    def test_evaluation_with_fixed_ranges_known_answer(self):
        # strokes 19 is half way between medium (8) and high (30): medium .5, high .5; length 12.5: medium .5, high .5
        # r1 = min(.5, .5) = .5; r2 = max(.5, .5) * .5 = .25; high = max(.5, .25) = .5
        result = fuzzy.evaluate(self.tiny_base(), {"strokesDrawn": 19.0, "totalLength": 12.5}, [], 8)
        indicator = result["indicators"][0]
        self.assertEqual(result["basis"], "fixed")
        self.assertEqual(indicator["degrees"], {"low": 0.0, "medium": 0.0, "high": 0.5})
        self.assertEqual((indicator["level"], indicator["text"], indicator["degree"]), ("high", "a lot", 0.5))
        self.assertEqual([f["rule"] for f in indicator["fired"]], ["r1", "r2"])
        self.assertEqual({i["metric"] for i in indicator["inputs"]}, {"strokesDrawn", "totalLength"})

    def test_cohort_percentiles_replace_the_fixed_ranges_in_a_big_enough_group(self):
        cohort = [{"strokesDrawn": float(v), "totalLength": float(v)} for v in range(1, 21)]
        result = fuzzy.evaluate(self.tiny_base(), {"strokesDrawn": 20.0, "totalLength": 20.0}, cohort, 8)
        self.assertEqual(result["basis"], "cohort")
        self.assertEqual(result["indicators"][0]["level"], "high")
        small = fuzzy.evaluate(self.tiny_base(), {"strokesDrawn": 20.0, "totalLength": 20.0}, cohort[:5], 8)
        self.assertEqual(small["basis"], "fixed")

    def test_a_missing_input_skips_the_rule_and_never_fakes_a_value(self):
        result = fuzzy.evaluate(self.tiny_base(), {"strokesDrawn": 19.0, "totalLength": None}, [], 8)
        indicator = result["indicators"][0]
        self.assertIn({"rule": "r1", "missing": ["totalLength"]}, indicator["skipped"])
        # r2 is an "or": it still fires on the input it has (high .5), at its weight of .5
        self.assertEqual(indicator["degrees"]["high"], 0.25)
        nothing = fuzzy.evaluate(self.tiny_base(), {"strokesDrawn": None, "totalLength": None}, [], 8)
        self.assertIsNone(nothing["indicators"][0]["level"])

    def test_not_negates(self):
        base = self.tiny_base()
        base["indicators"][0]["rules"] = [{"id": "n", "if": "strokesDrawn is not high", "then": "low", "weight": 1.0, "rationale": "x"}]
        self.assertEqual(fuzzy.evaluate(base, {"strokesDrawn": 2.0}, [], 8)["indicators"][0]["degrees"]["low"], 1.0)
        self.assertEqual(fuzzy.evaluate(base, {"strokesDrawn": 40.0}, [], 8)["indicators"][0]["degrees"]["low"], 0.0)

    def test_an_invalid_rule_base_names_every_problem(self):
        text = """
version = "1"
name = "bad"
status = "reviewed"
[fallbackRanges]
strokesDrawn = [5, 3, 9]
[[indicator]]
id = "a"
name = "A"
rationale = "r"
[indicator.levels]
low = "l"
medium = "m"
[[indicator.rule]]
id = "r1"
if = "nonsense is high"
then = "high"
rationale = ""
[[indicator.rule]]
id = "r2"
if = "strokesDrawn is huge"
then = "middle"
weight = 3
rationale = "x"
"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "rules.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(fuzzy.RuleBaseError) as raised:
                fuzzy.load_rule_base(path)
        problems = " ".join(raised.exception.problems)
        for expected in ("increasing", "levels.high", "rationale", "not a metric", "huge", "low, medium or high", "weight"):
            self.assertIn(expected, problems)
        with self.assertRaises(fuzzy.RuleBaseError):
            fuzzy.load_rule_base(Path("does-not-exist.toml"))

    def test_the_rule_file_can_be_changed_by_configuration(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "mine.toml"
            path.write_text(fuzzy.DEFAULT_PATH.read_text(encoding="utf-8").replace('name = "Starter rule base"', 'name = "Mine"'),
                            encoding="utf-8")
            with mock.patch.object(config, "AI_RULES_PATH", str(path), create=True):
                self.assertEqual(fuzzy.load_rule_base()["name"], "Mine")


class EnvelopeTest(unittest.TestCase):
    def test_every_output_is_labelled_and_carries_its_provenance(self):
        model = envelope.model_info("m", "Model", "1", "MIT", "neural")
        out = envelope.make("labels", model=model, params={"k": 1}, result={"a": 1}, prompt_version="v1", confidence=0.4567,
                            confidence_kind="vocabularyShare", evidence=[{"kind": "view", "view": "front"}])
        self.assertTrue(out["ai"])
        self.assertEqual(out["label"], "AI-generated")
        self.assertIn("not a diagnosis", out["disclaimer"])
        self.assertIn("reviewed by a psychologist", out["disclaimer"])
        for key in ("model", "params", "promptVersion", "confidence", "confidenceKind", "evidence", "generatedAt", "evaluated"):
            self.assertIn(key, out)
        self.assertEqual(out["confidence"], 0.457)
        self.assertFalse(out["evaluated"])  # nothing was evaluated with psychologists yet (AI-07)

    def test_no_confidence_means_no_kind(self):
        out = envelope.make("groups", model=envelope.model_info("m", "M", "1", "x", "statistical"), confidence_kind="share")
        self.assertIsNone(out["confidence"])
        self.assertIsNone(out["confidenceKind"])

    def test_diagnostic_wording_is_found_anywhere_in_an_answer(self):
        for text in ("This points to a depressive disorder", "signs of ADHD", "A diagnosis would be", "possible PTSD", "bipolar"):
            self.assertTrue(envelope.contains_diagnostic_language({"a": [text]}), text)
        for text in ("a house with a red roof", "warm colours dominate", "careful, slow lines"):
            self.assertFalse(envelope.contains_diagnostic_language({"a": [text]}), text)

    def test_clean_text(self):
        self.assertEqual(envelope.clean_text("  a \x00 b\n\tc  ", 50), "a b c")
        self.assertEqual(envelope.clean_text("x" * 100, 10), "x" * 10)
        self.assertIsNone(envelope.clean_text("   ", 10))
        self.assertIsNone(envelope.clean_text(5, 10))


class FakeOllama:
    """A tiny stand-in of an Ollama server on a free local port."""

    def __init__(self, models=("qwen2.5vl:7b", "qwen2.5:7b-instruct"), chat_answer=None, delay=0.0):
        self.models, self.chat_answer, self.delay, self.requests = list(models), chat_answer, delay, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, body: bytes, code=200):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(json.dumps({"models": [{"name": m, "size": 1} for m in outer.models]}).encode())
                elif self.path == "/api/version":
                    self._send(b'{"version": "0.0-test"}')
                else:
                    self._send(b"{}", 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                outer.requests.append((self.path, body))
                if self.path == "/api/chat":
                    time.sleep(outer.delay)
                    answer = outer.chat_answer(body) if callable(outer.chat_answer) else outer.chat_answer
                    self._send(json.dumps({"message": {"role": "assistant", "content": answer}}).encode())
                elif self.path == "/api/pull":
                    self.send_response(200)
                    self.end_headers()
                    for line in ({"status": "pulling manifest"}, {"status": "pulling abc", "digest": "abc", "total": 100, "completed": 40},
                                 {"status": "pulling abc", "digest": "abc", "total": 100, "completed": 100}, {"status": "success"}):
                        self.wfile.write(json.dumps(line).encode() + b"\n")
                else:
                    self._send(b"{}", 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True).start()  # polls fast: stops fast

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class OllamaTestCase(unittest.TestCase):
    def setUp(self):
        # A closed local port takes about 2 s to be refused on Windows: the probes of these tests give up sooner
        patcher = mock.patch.object(config, "AI_OLLAMA_PROBE_SECONDS", 0.2, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        registry.clear_cache()
        self.addCleanup(registry.clear_cache)

    def serve(self, **kwargs) -> FakeOllama:
        fake = FakeOllama(**kwargs)
        self.addCleanup(fake.close)
        for name, value in (("OLLAMA_URL", fake.url), ("AI_OLLAMA_TIMEOUT_SECONDS", 5)):
            patcher = mock.patch.object(config, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        registry.clear_cache()
        self.addCleanup(registry.clear_cache)
        return fake


class OllamaTest(OllamaTestCase):
    def test_models_and_chat(self):
        fake = self.serve(chat_answer='{"ok": true}')
        self.assertTrue(ollama.has_model("qwen2.5vl:7b"))
        self.assertTrue(ollama.has_model("qwen2.5vl:7b", [{"name": "x:latest"}, {"name": "qwen2.5vl:7b"}]))
        self.assertFalse(ollama.has_model("llama3"))
        self.assertTrue(ollama.has_model("x", [{"name": "x:latest"}]))
        self.assertEqual(ollama.chat("m", [{"role": "user", "content": "hi"}], schema={"type": "object"}), '{"ok": true}')
        path, body = fake.requests[-1]
        self.assertEqual((path, body["format"], body["stream"]), ("/api/chat", {"type": "object"}, False))
        self.assertEqual((body["options"]["temperature"], body["options"]["seed"]), (0, 0))  # repeatable

    def test_only_local_addresses_are_accepted(self):
        for url in ("http://example.com:11434", "http://10.0.0.5:11434", "https://api.openai.com", "ftp://127.0.0.1"):
            with mock.patch.object(config, "OLLAMA_URL", url, create=True):
                with self.assertRaises(ollama.OllamaError):
                    ollama.base_url()

    def test_an_unreachable_server_and_a_slow_one(self):
        with mock.patch.object(config, "OLLAMA_URL", "http://127.0.0.1:1", create=True):
            with self.assertRaises(ollama.OllamaError) as raised:
                ollama.installed_models()
            self.assertEqual(raised.exception.reason, "serviceUnreachable")
        self.serve(chat_answer="{}", delay=1.5)
        with mock.patch.object(config, "AI_OLLAMA_TIMEOUT_SECONDS", 0.3, create=True):
            with self.assertRaises(ollama.OllamaError) as raised:
                ollama.chat("m", [])
            self.assertEqual(raised.exception.reason, "timeout")

    def test_pull_reports_progress(self):
        self.serve()
        seen = []
        ollama.pull("m", lambda fraction, message: seen.append((fraction, message)))
        self.assertEqual(seen[-1][1], "success")
        self.assertIn(0.4, [round(f, 2) for f, _ in seen])
        self.assertTrue(all(f < 1 for f, _ in seen))


GOOD_DESCRIPTION = {"depictedObjects": [{"name": "house", "certainty": "medium"}, {"name": "tree", "certainty": "weird"}],
                    "scene": "A small house next to a tree.", "notableFeatures": ["a red roof"], "colourMoodWords": ["warm"],
                    "abstract": False}
GOOD_SUMMARY = {"overview": "A compact drawing.", "observations": ["Few strokes."], "thingsToExplore": ["What was it about?"],
                "limitations": ["Only one drawing."]}


class ModelAnswerTest(OllamaTestCase):
    def test_a_good_description_is_validated_and_cleaned(self):
        fake = self.serve(chat_answer=json.dumps(GOOD_DESCRIPTION))
        steps = []
        description, evidence = vlm.describe_drawing(synthetic("d1"), "qwen2.5vl:7b", lambda f, m: steps.append(f))
        self.assertEqual(description["depictedObjects"][1]["certainty"], "low")  # an unknown certainty becomes the weakest
        self.assertEqual([e["view"] for e in evidence], list(vlm.VIEWS))
        path, body = fake.requests[-1]
        self.assertEqual(len(body["messages"][1]["images"]), 3)  # the rendered views went to the model
        self.assertIn("Do not make a diagnosis", body["messages"][0]["content"])
        self.assertEqual(body["format"], vlm.SCHEMA)
        self.assertEqual(steps, sorted(steps))

    def test_answers_that_do_not_pass_are_discarded(self):
        cases = {
            "invalidAnswer": ["not json at all", "[1, 2]", json.dumps({**GOOD_DESCRIPTION, "scene": ""}),
                              json.dumps({k: v for k, v in GOOD_DESCRIPTION.items() if k != "abstract"})],
            "diagnosticLanguage": [json.dumps({**GOOD_DESCRIPTION, "scene": "The child seems to suffer from depression."}),
                                   json.dumps({**GOOD_DESCRIPTION, "notableFeatures": ["a sign of trauma disorder"]})],
        }
        for reason, answers_ in cases.items():
            for answer in answers_:
                self.serve(chat_answer=answer)
                with self.assertRaises(ollama.OllamaError) as raised:
                    vlm.describe_drawing(synthetic("d1"), "m", lambda *_: None)
                self.assertEqual(raised.exception.reason, reason, answer)

    def test_a_code_fence_around_the_json_is_tolerated(self):
        self.serve(chat_answer="```json\n" + json.dumps(GOOD_DESCRIPTION) + "\n```")
        self.assertEqual(vlm.describe_drawing(synthetic("d1"), "m", lambda *_: None)[0]["scene"], GOOD_DESCRIPTION["scene"])

    def test_the_conclusion_gets_facts_without_a_name_and_is_validated(self):
        metrics = metric_values(analyze_drawing(synthetic("d1")))
        indicators = [{"name": "Amount drawn", "level": "high", "text": "a lot was drawn", "degree": 0.8},
                      {"name": "Pace", "level": None, "text": None, "degree": 0.0}]
        facts = llm.build_facts(2, "adult (18 or older)", metrics, ["fewStrokes"], indicators, None, None, None)
        self.assertEqual([i["name"] for i in facts["indicators"]], ["Amount drawn"])  # an indicator without a level is left out
        self.assertEqual(facts["gameType"], "Beach")
        self.assertNotIn("name", {k for k in facts} - {"gameType"})
        fake = self.serve(chat_answer=json.dumps(GOOD_SUMMARY))
        self.assertEqual(llm.write_conclusion(facts, "m", lambda *_: None)["overview"], "A compact drawing.")
        sent = fake.requests[-1][1]["messages"]
        self.assertIn("make no diagnosis", sent[0]["content"])
        self.assertIn("Beach", sent[1]["content"])

    def test_a_conclusion_with_clinical_wording_is_discarded(self):
        self.serve(chat_answer=json.dumps({**GOOD_SUMMARY, "observations": ["This could indicate an anxiety disorder."]}))
        with self.assertRaises(ollama.OllamaError) as raised:
            llm.write_conclusion({}, "m", lambda *_: None)
        self.assertEqual(raised.exception.reason, "diagnosticLanguage")

    def test_string_lists_are_capped(self):
        data_ = {"items": [f"item {n}" for n in range(20)] + [None, 5, "  "]}
        self.assertEqual(len(answers.strings(data_, "items", 6)), 6)
        with self.assertRaises(ollama.OllamaError):
            answers.strings({}, "items", 6)


def decode_png(png: bytes) -> np.ndarray:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    position, idat, width = 8, b"", 0
    height = 0
    while position < len(png):
        length, tag = struct.unpack(">I4s", png[position:position + 8])
        body = png[position + 8:position + 8 + length]
        assert zlib.crc32(tag + body) & 0xFFFFFFFF == struct.unpack(">I", png[position + 8 + length:position + 12 + length])[0]
        if tag == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        position += 12 + length
    raw = zlib.decompress(idat)
    rows = [np.frombuffer(raw[r * (width * 3 + 1) + 1:(r + 1) * (width * 3 + 1)], dtype=np.uint8) for r in range(height)]
    return np.array(rows).reshape(height, width, 3)


class RenderTest(unittest.TestCase):
    def test_a_view_is_a_valid_png_with_the_stroke_colours_on_white(self):
        image = decode_png(render_view(synthetic("d1"), "front"))
        self.assertEqual(image.shape, (336, 336, 3))
        colours = {tuple(p) for p in image.reshape(-1, 3)}
        self.assertIn((255, 255, 255), colours)
        self.assertIn((255, 0, 0), colours)
        self.assertIn((0, 0, 255), colours)
        self.assertLess((image == 255).all(axis=2).mean(), 0.99)  # something was drawn

    def test_views_differ_and_an_empty_drawing_is_blank(self):
        # a flat drawing (all z = 0) is a line in the top view and a drawing in the front view
        views = {view: decode_png(render_view(synthetic("d1"), view)) for view in VIEWS}
        self.assertFalse(np.array_equal(views["front"], views["top"]))
        self.assertFalse(np.array_equal(views["front"], views["iso"]))
        blank = decode_png(render_view(make_drawing([]), "front"))
        self.assertTrue((blank == 255).all())

    def test_only_the_chosen_strokes_are_drawn(self):
        document = synthetic("d1")
        everything = decode_png(render_view(document, "front"))
        one = decode_png(render_view(document, "front", {"d1-0"}))
        self.assertLess((one != 255).any(axis=2).sum(), (everything != 255).any(axis=2).sum())

    def test_erased_strokes_are_not_drawn(self):
        document = make_drawing([make_line("gone", straight(), status=0)])
        self.assertTrue((decode_png(render_view(document, "front")) == 255).all())


class FakeProvider:
    """Scores a text high when it mentions the given word, so the expected label is known."""

    def __init__(self, word=None):
        self.word, self.calls = word, 0

    def score(self, images, texts):
        self.calls += 1
        return [[12.0 if self.word and self.word in text else 0.0 for text in texts] for _ in images]


class SemanticTest(unittest.TestCase):
    def test_the_default_vocabulary_is_valid(self):
        vocabulary = semantic.load_vocabulary()
        ids = [label["id"] for label in vocabulary["labels"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(any(label["other"] for label in vocabulary["labels"]))  # a way to say "nothing recognisable"
        self.assertTrue(all("{}" in t for t in vocabulary["templates"]))

    def test_invalid_vocabularies(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "v.toml"
            path.write_text('version = "1"\ntemplates = ["no placeholder"]\n[[label]]\nid = "a"\nname = "A"\nphrases = ["a"]\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                semantic.load_vocabulary(path)
        with self.assertRaises(ValueError):
            semantic.load_vocabulary(Path("missing.toml"))

    def test_the_label_the_model_prefers_comes_out_on_top(self):
        updates = []
        document = synthetic("d1", strokes=6)
        groups = clustering.cluster_strokes(document)["groups"]
        result, evidence = semantic.label_drawing(document, groups, FakeProvider("a tree"), semantic.load_vocabulary(),
                                                  lambda f, m: updates.append(f))
        self.assertEqual(result["whole"]["top"][0]["label"], "tree")
        self.assertGreater(result["whole"]["top"][0]["score"], 0.5)
        self.assertFalse(result["whole"]["unspecific"])
        self.assertEqual([v["view"] for v in result["whole"]["views"]], list(VIEWS))
        self.assertEqual(len(result["groups"]), len(groups))
        self.assertEqual(result["vocabulary"]["version"], "1")
        self.assertEqual(len(evidence), 4 + 2 * len(groups))
        self.assertEqual(updates, sorted(updates))

    def test_no_preference_or_an_other_entry_is_marked_unspecific(self):
        document = synthetic("d1")
        flat, _ = semantic.label_drawing(document, [], FakeProvider(None), semantic.load_vocabulary(), lambda *_: None)
        self.assertTrue(flat["whole"]["unspecific"])  # every entry gets the same share
        other, _ = semantic.label_drawing(document, [], FakeProvider("random scribbles"), semantic.load_vocabulary(), lambda *_: None)
        self.assertEqual(other["whole"]["top"][0]["label"], "scribble")
        self.assertTrue(other["whole"]["unspecific"])

    def test_without_the_libraries_the_model_says_so_instead_of_downloading(self):
        provider = semantic.ClipProvider("ViT-B-32", "laion2b_s34b_b79k")
        with mock.patch.object(settings, "module_available", return_value=False):
            with self.assertRaises(semantic.ProviderUnavailable) as raised:
                provider.score([b"x"], ["a"])
        self.assertEqual(raised.exception.reason, "missingLibrary")


class ClipLoadingTest(unittest.TestCase):
    """The CLIP checkpoint is found in the local cache and loaded from its file; only the download goes online."""

    def setUp(self):
        import sys
        import types

        self.cache: dict[str, str] = {}  # file name -> path, what the Hugging Face cache holds
        self.calls: list[tuple] = []
        hub = types.ModuleType("huggingface_hub")
        hub.try_to_load_from_cache = lambda repo, name, **kw: self.cache.get(name)
        clip = types.ModuleType("open_clip")
        clip.pretrained = types.SimpleNamespace(
            get_pretrained_cfg=lambda model, tag: {"hf_hub": "org/clip-repo/", "mean": (0.5, 0.5, 0.5), "std": (0.2, 0.2, 0.2), "quick_gelu": True},
            download_pretrained=self.download)
        clip.create_model_and_transforms = lambda *args, **kw: self.calls.append(("create", args, kw)) or (_FakeModel(), None, lambda image: None)
        clip.get_tokenizer = lambda name: None
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        for name, module in (("huggingface_hub", hub), ("open_clip", clip), ("torch", torch)):
            patcher = mock.patch.dict(sys.modules, {name: module})
            patcher.start()
            self.addCleanup(patcher.stop)
        for patcher in (mock.patch.object(settings, "module_available", return_value=True),
                        mock.patch.object(config, "AI_OLLAMA_PROBE_SECONDS", 0.2, create=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        registry.clear_cache()
        self.addCleanup(registry.clear_cache)

    def download(self, cfg):
        self.calls.append(("download", cfg["hf_hub"]))
        self.cache["open_clip_pytorch_model.bin"] = "/cache/open_clip_pytorch_model.bin"
        return self.cache["open_clip_pytorch_model.bin"]

    def test_the_checkpoint_is_found_in_the_cache_and_safetensors_comes_first(self):
        self.assertIsNone(registry.clip_checkpoint())
        self.cache["open_clip_pytorch_model.bin"] = "/cache/model.bin"
        self.assertEqual(registry.clip_checkpoint(), "/cache/model.bin")
        self.cache["open_clip_model.safetensors"] = "/cache/model.safetensors"
        self.assertEqual(registry.clip_checkpoint(), "/cache/model.safetensors")

    def test_the_model_details_a_path_does_not_carry_are_passed_along(self):
        self.cache["open_clip_model.safetensors"] = "/cache/model.safetensors"
        self.assertEqual(registry.clip_load_arguments(), {
            "pretrained": "/cache/model.safetensors", "force_quick_gelu": True,
            "image_mean": (0.5, 0.5, 0.5), "image_std": (0.2, 0.2, 0.2)})

    def test_without_the_weights_loading_says_so_and_never_touches_the_network(self):
        provider = semantic.ClipProvider("ViT-B-32", "laion2b_s34b_b79k")
        with self.assertRaises(semantic.ProviderUnavailable) as raised:
            provider.score([b"png"], ["a tree"])
        self.assertEqual(raised.exception.reason, "missingModel")
        self.assertEqual(self.calls, [])  # no create_model_and_transforms, so no download either

    def test_with_the_weights_the_checkpoint_file_is_loaded_not_a_tag(self):
        self.cache["open_clip_pytorch_model.bin"] = "/cache/model.bin"
        provider = semantic.ClipProvider("ViT-B-32", "laion2b_s34b_b79k")
        provider._load()
        (call,) = self.calls
        self.assertEqual(call[0], "create")
        self.assertEqual(call[2]["pretrained"], "/cache/model.bin")  # a file, so nothing can be fetched

    def test_a_failing_load_is_reported_as_a_missing_model(self):
        self.cache["open_clip_pytorch_model.bin"] = "/cache/model.bin"
        import open_clip

        def broken(*args, **kwargs):
            raise RuntimeError("corrupt file")

        open_clip.create_model_and_transforms = broken
        with self.assertRaises(semantic.ProviderUnavailable) as raised:
            semantic.ClipProvider("ViT-B-32", "x")._load()
        self.assertEqual(raised.exception.reason, "missingModel")
        self.assertIn("corrupt file", str(raised.exception))

    def test_the_status_follows_the_cache_and_the_download_is_the_one_online_step(self):
        entry = registry.model_by_id("clip")
        self.assertEqual((entry["status"], entry["downloadable"]), ("notDownloaded", True))
        updates = []
        registry.download("clip", lambda fraction, message: updates.append(message))
        self.assertEqual([c for c in self.calls if c[0] == "download"], [("download", "org/clip-repo/")])
        self.assertTrue(updates)
        self.assertEqual(registry.model_by_id("clip")["status"], "installed")  # the cache was cleared, the new status shows
        self.assertEqual((registry.model_by_id("clip")["downloadable"]), False)

    def test_a_download_that_leaves_nothing_usable_fails_with_a_message(self):
        import open_clip

        open_clip.pretrained.download_pretrained = lambda cfg: None
        with self.assertRaises(ollama.OllamaError):
            registry.download("clip", lambda *_: None)


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.manager = jobs.JobManager()
        self.addCleanup(self.manager.shutdown)

    def wait(self, job, status, seconds=5.0):
        end = time.time() + seconds
        while job.status not in status and time.time() < end:
            time.sleep(0.01)
        self.assertIn(job.status, status)

    def test_a_job_runs_and_reports_progress(self):
        def work(job):
            job.update(0.5, "halfway")

        job, started = self.manager.submit("labels", "d1", work)
        self.assertTrue(started)
        self.wait(job, (jobs.DONE,))
        self.assertEqual((job.progress, job.message, job.error), (1.0, "halfway", None))
        self.assertIsNotNone(job.finished_at)

    def test_the_same_work_is_not_started_twice(self):
        gate = threading.Event()
        first, started = self.manager.submit("labels", "d1", lambda job: gate.wait(5))
        second, started_again = self.manager.submit("labels", "d1", lambda job: None)
        self.assertEqual((second.id, started_again), (first.id, False))
        other, _ = self.manager.submit("labels", "d2", lambda job: None)
        self.assertNotEqual(other.id, first.id)
        gate.set()
        self.wait(first, (jobs.DONE,))

    def test_cancel_stops_a_running_job_at_its_next_step(self):
        gate = threading.Event()

        def work(job):
            gate.wait(5)
            job.update(0.5)  # the check after the wait raises JobCancelled

        job, _ = self.manager.submit("summary", "d1", work)
        self.wait(job, (jobs.RUNNING,))
        self.manager.cancel(job.id)
        gate.set()
        self.wait(job, (jobs.CANCELLED,))

    def test_cancel_removes_a_job_that_waits_in_the_queue(self):
        gate = threading.Event()
        blocker, _ = self.manager.submit("labels", "d1", lambda job: gate.wait(5))
        waiting, _ = self.manager.submit("labels", "d2", lambda job: self.fail("must not run"))
        self.manager.cancel(waiting.id)
        gate.set()
        self.wait(blocker, (jobs.DONE,))
        self.wait(waiting, (jobs.CANCELLED,))

    def test_a_failing_job_keeps_its_message_and_does_not_stop_the_next(self):
        def broken(job):
            raise RuntimeError("The model did not answer in time.")

        bad, _ = self.manager.submit("description", "d1", broken)
        self.wait(bad, (jobs.FAILED,))
        self.assertEqual(bad.error, "The model did not answer in time.")
        good, _ = self.manager.submit("description", "d2", lambda job: None)
        self.wait(good, (jobs.DONE,))
        self.assertEqual(self.manager.latest("description", "d1").id, bad.id)

    def test_only_a_limited_number_run_at_once(self):
        running, peak, lock = [0], [0], threading.Lock()

        def work(job):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1

        with mock.patch.object(config, "AI_MAX_CONCURRENT_JOBS", 2, create=True):
            submitted = [self.manager.submit("labels", f"d{n}", work)[0] for n in range(6)]
        for job in submitted:
            self.wait(job, (jobs.DONE,))
        self.assertEqual(peak[0], 2)

    def test_finished_jobs_are_forgotten_after_a_while(self):
        with mock.patch.object(config, "AI_JOB_HISTORY", 3, create=True):
            for n in range(8):
                job, _ = self.manager.submit("labels", f"d{n}", lambda job: None)
                self.wait(job, (jobs.DONE,))
        self.assertLessEqual(len(self.manager.list()), 4)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.folder.cleanup)
        self.store = AiStore(self.folder.name)

    def test_a_result_is_returned_only_for_its_own_key(self):
        self.store.put("labels", "d1", {"fingerprint": "a"}, {"status": "ok"})
        self.assertEqual(self.store.get("labels", "d1", {"fingerprint": "a"}), {"status": "ok"})
        self.assertIsNone(self.store.get("labels", "d1", {"fingerprint": "b"}))
        self.assertEqual(self.store.peek("labels", "d1")["key"], {"fingerprint": "a"})  # still there, for "out of date"
        self.assertTrue(self.store.delete("labels", "d1"))
        self.assertFalse(self.store.delete("labels", "d1"))

    def test_file_names_only_come_from_validated_ids(self):
        for feature, target in (("labels", "../x"), ("labels", "a/b"), ("labels", ""), ("cohort", "d1"), ("../labels", "d1")):
            with self.assertRaises(ValueError, msg=(feature, target)):
                self.store.put(feature, target, {}, {})

    def test_generated_stamps_only_list_good_results(self):
        self.store.put("summary", "d1", {}, {"status": "ok", "generatedAt": "T1"})
        self.store.put("labels", "d1", {}, {"status": "unavailable", "generatedAt": "T2"})
        self.assertEqual(self.store.generated_stamps("d1"), {"summary": "T1"})


class ServiceTestCase(DatabaseTestCase):
    """The AI service on a real temporary SQLite database and drawings kept in memory instead of MongoDB."""

    def setUp(self):
        super().setUp()
        self.files = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.files.cleanup)
        self.store, self.ai_store = AnalysisStore(self.files.name), AiStore(self.files.name)
        self.drawings: dict[str, dict] = {}
        for target, replacement in (
                ("load_drawing", lambda drawing_id: self.drawings.get(drawing_id)),
                ("drawing_fingerprint", lambda drawing_id: self.fingerprint(drawing_id))):
            patcher = mock.patch.object(data, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in (("AI_ENABLED", True),
                            ("AI_FEATURES", {n: True for n in settings.FEATURES}),
                            ("AI_EVALUATED", {n: False for n in settings.FEATURES}),
                            ("AI_OLLAMA_PROBE_SECONDS", 0.2)):
            patcher = mock.patch.object(config, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        registry.clear_cache()
        self.addCleanup(registry.clear_cache)
        self.session_id = cid()
        self.db.execute("INSERT INTO Sessions (ID, Name, Description, StartDate, EndDate, ShowBoy, ShowGirl, Multiplayer) "
                        "VALUES (?, 'Group A', 'First <b>group</b>', '2026.10.01-10:00', NULL, 1, 1, 0)", (self.session_id,))
        self.db.commit()
        self.player = self.add_player("Alex", age=30)

    def fingerprint(self, drawing_id: str):
        document = self.drawings.get(drawing_id)
        if document is None:
            return None
        import hashlib
        return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:16]

    def add_drawing(self, drawing_id=None, **kwargs) -> str:
        drawing_id = drawing_id or cid()
        document = synthetic(drawing_id, owner=self.player, session=self.session_id, **kwargs)
        self.drawings[drawing_id] = document
        self.db.execute("INSERT INTO DrawingMeta (ID, PlayerID, Name, Path, GameType, SessionID) VALUES (?, ?, ?, '', ?, ?)",
                        (drawing_id, self.player, f"Drawing {drawing_id[:4]}", kwargs.get("game_type", 2), self.session_id))
        self.db.commit()
        return drawing_id

    def context(self) -> service.Context:
        return service.Context(self.db, self.store, self.ai_store)

    def fill(self, count: int) -> list[str]:
        return [self.add_drawing(strokes=4 + n % 3, length=0.8 + 0.1 * (n % 4), step_ms=100 + 10 * (n % 3)) for n in range(count)]


class CohortServiceTest(ServiceTestCase):
    def test_too_small_a_group_says_not_enough_data(self):
        ids = self.fill(3)
        out = service.drawing_cohort(self.context(), ids[0])
        self.assertEqual((out["status"], out["reason"], out["result"]), ("notEnoughData", "notEnoughData", None))
        self.assertEqual((out["params"]["have"], out["params"]["need"]), (3, 8))
        self.assertTrue(out["ai"] and out["label"] == "AI-generated")

    def test_a_big_enough_group(self):
        ids = self.fill(14)
        out = service.drawing_cohort(self.context(), ids[0])
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["result"]["cohortSize"], 14)
        self.assertEqual(out["result"]["subject"]["id"], ids[0])
        self.assertEqual(len(out["result"]["map"]["points"]), 14)
        for neighbour in out["result"]["subject"]["similar"]:
            self.assertIn(neighbour["id"], out["result"]["directory"])
        self.assertEqual(out["model"]["licence"], "BSD-3-Clause")
        self.assertTrue(out["evidence"])

    def test_a_planted_big_drawing_is_atypical(self):
        self.fill(12)
        big = self.add_drawing(strokes=40, length=4.0)
        out = service.drawing_cohort(self.context(), big)
        self.assertTrue(out["result"]["subject"]["atypical"])
        session = service.session_cohort(self.context(), self.session_id)
        self.assertEqual([r["id"] for r in session["result"]["drawings"] if r["atypical"]], [big])

    def test_the_comparison_group_can_be_chosen(self):
        self.fill(10)
        other_type = self.add_drawing(game_type=4)
        out = service.drawing_cohort(self.context(), other_type, scope="gameType")
        self.assertEqual(out["status"], "notEnoughData")  # nobody else made a drawing of this game type
        self.assertEqual(service.drawing_cohort(self.context(), other_type, scope="all")["status"], "ok")

    def test_unknown_drawings_are_none(self):
        self.assertIsNone(service.drawing_cohort(self.context(), "nope"))
        self.assertIsNone(service.session_cohort(self.context(), "nope"))

    def test_a_function_that_is_switched_off_says_so(self):
        ids = self.fill(10)
        with mock.patch.object(config, "AI_FEATURES", {**config.AI_FEATURES, "cohort": False}):
            out = service.drawing_cohort(self.context(), ids[0])
        self.assertEqual((out["status"], out["reason"]), ("disabled", "switchedOff"))
        with mock.patch.object(config, "AI_ENABLED", False):
            self.assertEqual(service.drawing_cohort(self.context(), ids[0])["status"], "disabled")
            self.assertEqual(service.drawing_indicators(self.context(), ids[0])["status"], "disabled")
            self.assertFalse(service.status()["enabled"])
            self.assertEqual(service.report_sections(self.db, "drawing", ids[0]), [])

    def test_without_the_library_the_function_is_unavailable_not_broken(self):
        ids = self.fill(10)
        with mock.patch.object(settings, "module_available", return_value=False):
            out = service.drawing_cohort(self.context(), ids[0])
        self.assertEqual((out["status"], out["reason"]), ("unavailable", "missingLibrary"))


class IndicatorsAndGroupsServiceTest(ServiceTestCase):
    def test_indicators_with_a_small_group_use_fixed_ranges_and_say_so(self):
        ids = self.fill(3)
        out = service.drawing_indicators(self.context(), ids[0])
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["result"]["basis"], "fixed")
        self.assertTrue(any("small" in note for note in out["notes"]))
        self.assertTrue(any("proposal" in note for note in out["notes"]))  # the rule base is not reviewed yet
        self.assertIn("proposal", out["promptVersion"])
        names = {i["id"] for i in out["result"]["indicators"]}
        self.assertTrue({"amountDrawn", "spaceUse", "pace"} <= names)
        amount = next(i for i in out["result"]["indicators"] if i["id"] == "amountDrawn")
        self.assertTrue(amount["fired"] and amount["fired"][0]["rationale"])

    def test_a_broken_rule_base_switches_the_indicators_off_with_the_reason(self):
        ids = self.fill(3)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            bad = Path(folder) / "bad.toml"
            bad.write_text("this is not toml = = =", encoding="utf-8")
            with mock.patch.object(config, "AI_RULES_PATH", str(bad), create=True):
                out = service.drawing_indicators(self.context(), ids[0])
                self.assertEqual((out["status"], out["reason"]), ("unavailable", "invalidRules"))
                self.assertEqual(registry.model_by_id("fuzzy-rules")["status"], "invalid")

    def test_groups_are_computed_once_and_stored(self):
        drawing_id = self.add_drawing(strokes=6)
        first = service.drawing_groups(self.context(), drawing_id)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(sum(g["strokeCount"] for g in first["result"]["groups"]) + len(first["result"]["unclustered"]), 6)
        with mock.patch.object(clustering, "cluster_strokes", side_effect=AssertionError("computed again")):
            self.assertEqual(service.drawing_groups(self.context(), drawing_id)["generatedAt"], first["generatedAt"])
        self.drawings[drawing_id]["lines"].pop()  # the drawing changed: the stored groups are not used any more
        self.assertNotEqual(service.drawing_groups(self.context(), drawing_id)["result"]["strokeCount"], 6)

    def test_a_drawing_without_strokes_has_no_groups(self):
        drawing_id = self.add_drawing(strokes=0)
        out = service.drawing_groups(self.context(), drawing_id)
        self.assertEqual((out["status"], out["reason"]), ("notEnoughData", "noStrokes"))


class HeavyServiceTest(ServiceTestCase, OllamaTestCase):
    def wait_for_job(self, drawing_id, feature):
        end = time.time() + 10
        while time.time() < end:
            state = service.heavy_state(self.context(), feature, drawing_id)
            if state["job"] and state["job"]["status"] not in ("queued", "running"):
                return state
            time.sleep(0.02)
        self.fail("the job did not finish")

    def setUp(self):
        super().setUp()
        self.addCleanup(lambda: jobs.manager.shutdown())
        # AiStore() inside a job uses the default root: point it at the temporary folder
        patcher = mock.patch.object(config, "ANALYSIS_STORE_PATH", self.files.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict("os.environ", {"ANALYSIS_STORE_PATH": self.files.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_description_runs_as_a_job_and_the_result_is_reused(self):
        self.serve(chat_answer=json.dumps(GOOD_DESCRIPTION))
        drawing_id = self.add_drawing()
        started = service.start_heavy(self.context(), "description", drawing_id)
        self.assertTrue(started["available"])
        state = self.wait_for_job(drawing_id, "description")
        self.assertEqual(state["job"]["status"], "done")
        self.assertEqual(state["job"]["progress"], 1.0)
        out = state["result"]
        self.assertEqual((out["status"], out["feature"], out["label"]), ("ok", "description", "AI-generated"))
        self.assertEqual(out["promptVersion"], vlm.PROMPT_VERSION)
        self.assertEqual(out["params"]["temperature"], 0)
        self.assertEqual(out["model"]["name"], "qwen2.5vl:7b")
        self.assertFalse(state["stale"])
        # a second start reuses the up to date result: no new job
        job_id = state["job"]["id"]
        again = service.start_heavy(self.context(), "description", drawing_id)
        self.assertEqual(again["job"]["id"], job_id)
        # a drawing that changed makes the stored result stale, and regenerating replaces it
        self.drawings[drawing_id]["lines"].pop()
        self.assertTrue(service.heavy_state(self.context(), "description", drawing_id)["stale"])
        service.start_heavy(self.context(), "description", drawing_id)
        state = self.wait_for_job(drawing_id, "description")
        self.assertNotEqual(state["job"]["id"], job_id)
        self.assertFalse(state["stale"])

    def test_a_model_that_misbehaves_fails_the_job_with_a_readable_message(self):
        self.serve(chat_answer=json.dumps({**GOOD_DESCRIPTION, "scene": "Looks like a psychosis."}))
        drawing_id = self.add_drawing()
        service.start_heavy(self.context(), "description", drawing_id)
        state = self.wait_for_job(drawing_id, "description")
        self.assertEqual(state["job"]["status"], "failed")
        self.assertIn("clinical or diagnostic wording", state["job"]["error"])
        self.assertIsNone(state["result"])  # nothing of the answer is kept

    def test_without_the_service_or_the_model_nothing_is_started(self):
        drawing_id = self.add_drawing()
        with mock.patch.object(config, "OLLAMA_URL", "http://127.0.0.1:1", create=True):
            registry.clear_cache()
            state = service.start_heavy(self.context(), "summary", drawing_id)
        self.assertEqual((state["available"], state["reason"], state["job"]), (False, "serviceUnreachable", None))
        self.serve(models=("something-else",))
        state = service.start_heavy(self.context(), "description", drawing_id)
        self.assertEqual((state["available"], state["reason"], state["job"]), (False, "missingModel", None))

    def test_labels_with_a_model_that_is_there(self):
        clip = {"id": "clip", "name": "OpenCLIP ViT-B-32", "version": "laion2b", "kind": "neural", "licence": "MIT",
                "licenceChecked": False, "source": "", "sizeHint": None, "usedBy": ["labels"], "status": "installed",
                "statusDetail": None, "downloadable": False, "hint": None}
        drawing_id = self.add_drawing(strokes=6)
        with mock.patch.object(registry, "_clip", return_value=clip), mock.patch.object(semantic, "get_provider", return_value=FakeProvider("a house")):
            registry.clear_cache()
            service.start_heavy(self.context(), "labels", drawing_id)
            state = self.wait_for_job(drawing_id, "labels")
        out = state["result"]
        self.assertEqual(out["result"]["whole"]["top"][0]["label"], "house")
        self.assertEqual((out["confidenceKind"], out["confidence"] is not None), ("vocabularyShare", True))
        self.assertTrue(any(e["kind"] == "view" for e in out["evidence"]))
        self.assertTrue(any("probability" in note for note in out["notes"]))

    def test_the_conclusion_is_keyed_on_the_facts_it_was_written_from(self):
        self.serve(chat_answer=json.dumps(GOOD_SUMMARY))
        ids = self.fill(3)
        service.start_heavy(self.context(), "summary", ids[0])
        state = self.wait_for_job(ids[0], "summary")
        self.assertEqual(state["result"]["result"]["overview"], "A compact drawing.")
        self.assertFalse(state["stale"])
        self.assertEqual(state["result"]["evidence"][0]["kind"], "facts")
        self.assertNotIn("Alex", json.dumps(state["result"]["evidence"]))  # the model never sees the name
        self.add_drawing(strokes=30, length=3.0)  # the group it was compared with changed -> other indicators or numbers
        self.drawings[ids[0]]["lines"].pop()
        self.assertTrue(service.heavy_state(self.context(), "summary", ids[0])["stale"])

    def test_a_model_download_is_an_explicit_job_for_a_listed_model_only(self):
        self.serve(models=())
        job = service.start_download("vision")
        end = time.time() + 5
        while job.status in ("queued", "running") and time.time() < end:
            time.sleep(0.02)
        self.assertEqual(job.status, "done")
        self.assertIsNone(service.start_download("../../etc/passwd"))
        self.assertIsNone(service.start_download("not-listed"))


class RegistryTest(OllamaTestCase):
    def test_every_entry_has_a_licence_and_a_status(self):
        self.serve(models=("qwen2.5:7b-instruct",))
        entries = {m["id"]: m for m in registry.models()}
        self.assertEqual(set(entries), {"numpy", "scikit-learn", "fuzzy-rules", "clip", "ollama", "vision", "text"})
        for entry in entries.values():
            self.assertTrue(entry["licence"] and entry["status"] in ("installed", "notInstalled", "notDownloaded", "unreachable", "invalid"))
            self.assertFalse(entry["licenceChecked"] and entry["id"] != "fuzzy-rules")  # no licence is claimed as verified
        self.assertEqual(entries["text"]["status"], "installed")
        self.assertEqual(entries["vision"]["status"], "notDownloaded")
        self.assertTrue(entries["vision"]["downloadable"])
        self.assertEqual(entries["scikit-learn"]["status"], "installed")
        self.assertEqual(entries["text"]["licence"], "Apache-2.0")

    def test_without_ollama_its_models_are_unreachable(self):
        with mock.patch.object(config, "OLLAMA_URL", "http://127.0.0.1:1", create=True):
            registry.clear_cache()
            entries = {m["id"]: m for m in registry.models()}
        self.assertEqual({entries[i]["status"] for i in ("ollama", "vision", "text")}, {"unreachable"})
        self.assertFalse(entries["vision"]["downloadable"])

    def test_the_status_lists_every_function_with_its_requirement(self):
        status = service.status()
        self.assertEqual({f["id"] for f in status["features"]}, set(settings.FEATURES))
        self.assertEqual({f["requirement"] for f in status["features"]},
                         {"AI-10", "AI-11", "AI-12", "AI-13", "AI-14", "AI-15", "AI-16"})
        self.assertTrue(all(f["evaluated"] is False for f in status["features"]))  # nothing was evaluated yet (AI-07)
        self.assertIn("not a diagnosis", status["disclaimer"])


class InsightsTest(unittest.TestCase):
    @staticmethod
    def labels(top: str, other=False, unspecific=False) -> dict:
        verdict = {"top": [{"label": top, "name": top.title(), "other": other, "score": 0.6}], "margin": 0.3, "unspecific": unspecific}
        return {"whole": verdict, "groups": []}

    @staticmethod
    def indicator(level):
        return {"id": "pace", "name": "Pace", "level": level}

    def test_a_label_that_recurs_across_participants(self):
        drawings = [{"id": f"d{n}", "name": "", "playerId": f"p{n % 2}", "indicators": [], "labels": self.labels(label), "atypical": None}
                    for n, label in enumerate(["tree", "tree", "tree", "house"])]
        found = {i["label"]: i for i in insights.aggregate(drawings) if i["kind"] == "recurringLabel"}
        self.assertEqual(set(found), {"tree"})  # a label named once does not recur
        self.assertEqual((found["tree"]["drawings"], found["tree"]["participants"]), (3, 2))

    def test_unspecific_and_other_labels_do_not_count(self):
        drawings = [{"id": f"d{n}", "name": "", "playerId": "p", "indicators": [], "atypical": None,
                     "labels": self.labels("scribble", other=True) if n else self.labels("tree", unspecific=True)} for n in range(4)]
        self.assertEqual([i for i in insights.aggregate(drawings) if i["kind"] == "recurringLabel"], [])

    def test_indicator_levels_and_the_common_one(self):
        drawings = [{"id": f"d{n}", "name": "", "playerId": "p", "labels": None, "atypical": False,
                     "indicators": [self.indicator(level)]} for n, level in enumerate(["high", "high", "high", "low", None])]
        item = next(i for i in insights.aggregate(drawings) if i["kind"] == "indicatorLevels")
        self.assertEqual(item["counts"], {"low": 1, "medium": 0, "high": 3})
        self.assertEqual((item["total"], item["common"]), (4, "high"))
        mixed = [{**d, "indicators": [self.indicator(level)]} for d, level in zip(drawings, ["high", "low", "medium", "low"])]
        self.assertIsNone(next(i for i in insights.aggregate(mixed) if i["kind"] == "indicatorLevels")["common"])

    def test_atypical_drawings(self):
        drawings = [{"id": f"d{n}", "name": "", "playerId": "p", "indicators": [], "labels": None, "atypical": n == 2} for n in range(3)]
        self.assertEqual(next(i for i in insights.aggregate(drawings) if i["kind"] == "atypical")["drawingIds"], ["d2"])


class InsightsServiceTest(ServiceTestCase):
    def test_session_insights(self):
        self.fill(3)
        out = service.session_insights(self.context(), self.session_id)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["result"]["drawingCount"], 3)
        self.assertEqual(out["result"]["coverage"], {"labels": 0, "indicators": 3})
        self.assertEqual(out["result"]["atypicalSkipped"], "notEnoughData")  # fewer than 8 drawings: no atypical detection
        self.assertTrue(any(i["kind"] == "indicatorLevels" for i in out["result"]["insights"]))

    def test_one_drawing_is_not_a_session_to_summarise(self):
        self.fill(1)
        self.assertEqual(service.session_insights(self.context(), self.session_id)["status"], "notEnoughData")


class ReportServiceTest(ServiceTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(lambda: jobs.manager.shutdown())

    def test_the_drawing_report_has_marked_ai_sections_and_notices_new_results(self):
        ids = self.fill(3)
        created, reused = data.create_report(self.db, self.store, "drawing", ids[0])
        self.assertFalse(reused)
        html = self.store.read_report_html(created["id"])
        self.assertIn('class="ai-badge">AI-generated<', html)
        self.assertIn("not a diagnosis", html)
        self.assertIn("reviewed by a psychologist", html)
        self.assertIn("Indicators from the fuzzy rule base", html)
        self.assertIn("Model", html)
        self.assertFalse(created["stale"])
        again, reused_again = data.create_report(self.db, self.store, "drawing", ids[0])
        self.assertTrue(reused_again)
        # a stored AI result that did not exist when the report was made makes the report out of date
        out = envelope.make("description", model=envelope.model_info("m", "M", "1", "x", "language"), result=GOOD_DESCRIPTION)
        found = data.drawing_analysis(ids[0], self.store)
        self.ai_ctx_store = AiStore(self.files.name)
        self.ai_ctx_store.put("description", ids[0], {"fingerprint": found[1], "analyzer": found[0]["analyzerVersion"],
                                                      "moduleVersion": "1.0", "model": settings.number("AI_VISION_MODEL", ""),
                                                      "prompt": vlm.PROMPT_VERSION}, out)
        with mock.patch.object(service, "AiStore", lambda: self.ai_ctx_store), \
                mock.patch("ai.service.AiStore", lambda root=None: self.ai_ctx_store):
            listed = [data.report_info(self.db, meta) for meta in self.store.list_reports()]
        self.assertTrue(listed[0]["stale"])

    def test_a_stored_result_that_is_out_of_date_is_still_in_the_report_marked_as_such(self):
        ids = self.fill(3)
        out = envelope.make("description", model=envelope.model_info("m", "M", "1", "x", "language"), result=GOOD_DESCRIPTION)
        self.ai_store.put("description", ids[0], {"fingerprint": "an-older-drawing"}, out)
        with mock.patch("ai.service.AiStore", lambda root=None: self.ai_store):
            sections = service.report_sections(self.db, "drawing", ids[0])
        found = [s for s in sections if s["feature"] == "description"]
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["stale"])
        html = reporting.sections_html(found)
        self.assertIn("Out of date", html)
        self.assertIn(GOOD_DESCRIPTION["scene"], html)
        self.assertIn('class="ai-badge"', html)  # still marked as AI-generated

    def test_a_report_is_made_without_the_ai_part_when_ai_is_switched_off(self):
        ids = self.fill(3)
        with mock.patch.object(config, "AI_ENABLED", False):
            created, _ = data.create_report(self.db, self.store, "drawing", ids[0])
            html = self.store.read_report_html(created["id"])
            self.assertNotIn("AI-generated", html)
            self.assertFalse(created["stale"])
        self.assertTrue(data.report_info(self.db, self.store.list_reports()[0])["stale"])  # switched on again: out of date

    def test_a_problem_in_the_ai_part_never_stops_a_report(self):
        ids = self.fill(3)
        with mock.patch.object(service, "report_sections", side_effect=RuntimeError("boom")), \
                self.assertLogs("analysis", level="ERROR"):
            created, _ = data.create_report(self.db, self.store, "drawing", ids[0])
        self.assertIn("Metrics", self.store.read_report_html(created["id"]))

    def test_the_session_report_has_the_insights(self):
        self.fill(3)
        created, _ = data.create_report(self.db, self.store, "session", self.session_id)
        html = self.store.read_report_html(created["id"])
        self.assertIn("Session insights", html)
        self.assertIn("AI-generated", html)

    def test_text_from_models_and_data_is_escaped_in_reports(self):
        hostile = {**GOOD_DESCRIPTION, "scene": "<script>alert(1)</script>", "depictedObjects": [{"name": "<img src=x onerror=1>", "certainty": "high"}]}
        out = envelope.make("description", model=envelope.model_info("m", "<b>M</b>", "1", "x", "language"), result=hostile,
                            notes=["<i>note</i>"])
        html = reporting.sections_html([out])
        for raw in ("<script>", "<img", "<b>M</b>", "<i>note</i>"):
            self.assertNotIn(raw, html)
        self.assertIn("&lt;script&gt;", html)

    def test_every_section_kind_renders(self):
        ids = self.fill(14)
        ctx = self.context()
        sections = [service.drawing_indicators(ctx, ids[0]), service.drawing_cohort(ctx, ids[0]), service.drawing_groups(ctx, ids[0]),
                    service.session_insights(ctx, self.session_id)]
        sections.append(envelope.make("labels", model=envelope.model_info("clip", "CLIP", "1", "MIT", "neural"),
                                      result=semantic.label_drawing(self.drawings[ids[0]], [], FakeProvider("a tree"),
                                                                    semantic.load_vocabulary(), lambda *_: None)[0],
                                      confidence=0.7, confidence_kind="vocabularyShare"))
        sections.append(envelope.make("summary", model=envelope.model_info("m", "M", "1", "x", "language"), result=GOOD_SUMMARY))
        html = reporting.sections_html(sections)
        self.assertEqual(html.count('class="ai-badge"'), len(sections))
        self.assertEqual(html.count(envelope.DISCLAIMER.replace("'", "&#x27;")), len(sections))
        self.assertIn("vocabularyShare", html)


class AccessTest(unittest.TestCase):
    def test_every_ai_endpoint_needs_a_psychologist(self):
        self.assertIn(security.require_psychologist, [d.dependency for d in ai_router.router.dependencies])
        with mock.patch.object(security.throttle, "account_limiter", security.throttle.TokenBuckets(300, 20)):
            for role in (security.Role.PLAYER, security.Role.ADMIN):
                path = "/ai/status"
                request = request_for(path)
                if role == security.Role.ADMIN:  # the general rule keeps an admin token away from everything but /admin
                    with self.assertRaises(HTTPException) as refused:
                        security.get_current_player(request, credentials("a", role))
                    self.assertEqual(refused.exception.status_code, 403)
                else:
                    current = security.get_current_player(request, credentials("p", role))
                    with self.assertRaises(HTTPException) as refused:
                        security.require_psychologist(current)
                    self.assertEqual(refused.exception.status_code, 403)
            with self.assertRaises(HTTPException) as anonymous:
                security.get_current_player(request_for("/ai/status"), None)
            self.assertEqual(anonymous.exception.status_code, 401)

    def test_ids_and_parameters_are_validated_by_the_routes(self):
        paths = {route.path for route in ai_router.router.routes}
        self.assertIn("/ai/drawings/{drawing_id}/{feature}", paths)
        names = " ".join(str(route.dependant.path_params) for route in ai_router.router.routes if hasattr(route, "dependant"))
        self.assertIn("drawing_id", names)


if __name__ == "__main__":
    unittest.main()
