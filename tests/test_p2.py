import base64
import json
import math
import struct
import tempfile
import unittest

from analysis import reports
from analysis.common import BREAK_DISPLAY_MS
from analysis.exports import export_drawing, session_csv
from analysis.geometry import drawing_svg, simplify_indices, stroke_geometry
from analysis.heatmap import build_heatmap
from analysis.metrics import analyze_drawing
from analysis.multiplayer import analyze_multiplayer
from analysis.narrative import build_narrative
from analysis.store import AnalysisStore, is_valid_id
from analysis.timeline import build_timeline
from analysis.tracking import forward_vector, summarize_tracking
from tests.fixtures import LOCAL_SHIFT_MS, T0, legacy_stamp, make_drawing, make_line, pose, straight


class TimelineTest(unittest.TestCase):
    def test_times_are_relative_to_the_first_stroke(self):
        drawing = make_drawing([make_line("a", straight(), start_ms=T0 + 5000)])
        timeline = build_timeline(drawing)
        stroke = timeline["items"][0]
        self.assertEqual((stroke["startMs"], stroke["endMs"]), (0, 1000))
        self.assertEqual(timeline["originMs"], T0 + 5000)

    def test_history_events_are_aligned_with_the_utc_axis(self):
        # The history only has the legacy local-time stamp. The offset between a point's two stamps fixes the axis.
        history = [{"eventType": 0, "invoker": "p1", "hand": 1, "timestamp": legacy_stamp(T0 + 1500 + LOCAL_SHIFT_MS)},
                   {"eventType": 2, "invoker": "p1", "hand": 1, "timestamp": legacy_stamp(T0 + 1800 + LOCAL_SHIFT_MS)}]
        timeline = build_timeline(make_drawing([make_line("a", straight(), history=history)]))
        marks = {i["kind"]: i["atMs"] for i in timeline["items"] if i["kind"] in ("erase", "undo")}
        self.assertEqual(marks, {"erase": 1500, "undo": 1800})

    def test_events_and_summary(self):
        events = [{"type": 0, "playerID": "p1", "hand": 1, "utcMs": T0 + 200},
                  {"type": 1, "playerID": "p1", "hand": 1, "utcMs": T0 + 300},
                  {"type": 2, "playerID": "p1", "hand": 1, "utcMs": T0 + 400, "value": "Draw"},
                  {"type": 3, "playerID": "p1", "hand": 1, "utcMs": T0 + 500, "value": "start=#FF0000FF"}]
        drawing = make_drawing([make_line("a", straight()), make_line("b", straight(), status=0, start_ms=T0 + 20000)], events=events)
        timeline = build_timeline(drawing)
        kinds = [i["kind"] for i in timeline["items"]]
        self.assertEqual(kinds.count("menuOpen"), 1)
        self.assertEqual(kinds.count("menuClose"), 1)
        self.assertEqual(kinds.count("modeChange"), 1)
        self.assertEqual(kinds.count("colorChange"), 1)
        self.assertEqual(timeline["summary"]["strokes"], 1)
        self.assertEqual(timeline["summary"]["erasedStrokes"], 1)
        self.assertEqual(timeline["summary"]["menuOpens"], 1)
        self.assertEqual(timeline["summary"]["toolChanges"], 2)
        self.assertEqual(sorted(i.get("startMs", i.get("atMs")) for i in timeline["items"]),
                         [i.get("startMs", i.get("atMs")) for i in timeline["items"]])  # sorted in time

    def test_idle_period_between_strokes(self):
        drawing = make_drawing([make_line("a", straight()), make_line("b", straight(), start_ms=T0 + 9000)])
        timeline = build_timeline(drawing, idle_gap_seconds=5)
        idle = [i for i in timeline["items"] if i["kind"] == "idle"]
        self.assertEqual([(i["startMs"], i["endMs"]) for i in idle], [(1000, 9000)])
        self.assertEqual(timeline["summary"]["idleSeconds"], 8.0)
        self.assertEqual(build_timeline(drawing, idle_gap_seconds=10)["summary"]["idlePeriods"], 0)

    def test_a_break_between_sittings_is_cut_down_on_the_axis(self):
        two_days = 2 * 24 * 3600 * 1000
        drawing = make_drawing([make_line("a", straight()), make_line("b", straight(), start_ms=T0 + two_days)])
        timeline = build_timeline(drawing)
        strokes = [i for i in timeline["items"] if i["kind"] == "stroke"]
        self.assertEqual((strokes[0]["startMs"], strokes[0]["endMs"]), (0, 1000))
        self.assertEqual(strokes[1]["startMs"], 1000 + BREAK_DISPLAY_MS)  # the break is shown as 5 s
        self.assertEqual(timeline["durationMs"], 1000 + BREAK_DISPLAY_MS + 1000)
        self.assertEqual(timeline["breaks"], [{"atMs": 1000, "realMs": two_days - 1000}])
        self.assertEqual([i for i in timeline["items"] if i["kind"] == "idle"], [])  # not a pause
        geometry = stroke_geometry(drawing, 0)
        self.assertEqual(geometry["strokes"][1]["points"][0][3], 1000 + BREAK_DISPLAY_MS)
        self.assertEqual(geometry["durationMs"], timeline["durationMs"])

    def test_an_event_inside_a_break_lands_at_its_start(self):
        two_days = 2 * 24 * 3600 * 1000
        events = [{"type": 0, "playerID": "p1", "hand": 1, "utcMs": T0 + 3600 * 1000}]  # an hour into the break
        drawing = make_drawing([make_line("a", straight()), make_line("b", straight(), start_ms=T0 + two_days)], events=events)
        menu = next(i for i in build_timeline(drawing)["items"] if i["kind"] == "menuOpen")
        self.assertEqual(menu["atMs"], 1000)

    def test_empty_drawing(self):
        timeline = build_timeline(make_drawing([]))
        self.assertEqual(timeline["items"], [])
        self.assertFalse(timeline["hasTimeData"])
        self.assertEqual(timeline["breaks"], [])


class TrackingTest(unittest.TestCase):
    def test_distance_and_unavailable(self):
        head = [pose(0, 1.5, 0, T0), pose(3, 1.5, 4, T0 + 1000)]
        behavior = {"playerID": "p1", "head": head, "rightHand": [], "leftHand": [], "player": []}
        result = summarize_tracking(make_drawing([make_line("a", straight())], tracked=[behavior]))
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["players"][0]["movement"]["head"], 5.0)
        self.assertEqual(result["players"][0]["durationMs"], 1000)
        self.assertEqual(result["players"][0]["paths"]["head"][0][0], 0)  # relative to the drawing origin
        self.assertFalse(summarize_tracking(make_drawing([make_line("a", straight())]))["available"])

    def test_forward_vector_of_known_rotations(self):
        self.assertEqual(tuple(round(c, 6) for c in forward_vector(pose(0, 0, 0, 0))), (0.0, 0.0, 1.0))
        turned = pose(0, 0, 0, 0, q=(0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)))  # 90 degrees about Y
        self.assertEqual(tuple(round(c, 6) for c in forward_vector(turned)), (1.0, 0.0, 0.0))

    def test_head_direction_and_explored_area(self):
        quarter = math.sin(math.pi / 4)
        head = [pose(0, 1.6, 0, T0), pose(1, 1.6, 0, T0 + 100, q=(0, quarter, 0, quarter)), pose(1, 1.6, 1, T0 + 200)]
        result = summarize_tracking(make_drawing([make_line("a", straight())],
                                                 tracked=[{"playerID": "p1", "head": head, "rightHand": [], "leftHand": [], "player": []}]))
        player = result["players"][0]
        self.assertEqual(len(player["headDirection"]["yawShares"]), 12)
        self.assertAlmostEqual(sum(player["headDirection"]["yawShares"]), 1.0, places=2)
        self.assertGreaterEqual(player["headDirection"]["yawCoverage"], 2)
        self.assertGreater(player["explored"]["areaM2"], 0)

    def test_thinned_tracking_is_flagged_approximate(self):
        behavior = {"playerID": "p1", "head": [pose(0, 1, 0, T0)], "rightHand": [], "leftHand": [], "player": []}
        result = summarize_tracking(make_drawing([make_line("a", straight())], tracked=[behavior], decimation=3))
        self.assertTrue(result["approximate"])
        self.assertEqual(result["decimation"], 3)

    def test_paths_are_limited(self):
        head = [pose(i * 0.01, 1.5, 0, T0 + i * 10) for i in range(1000)]
        result = summarize_tracking(make_drawing([make_line("a", straight())], tracked=[
            {"playerID": "p1", "head": head, "rightHand": [], "leftHand": [], "player": []}]), max_points=100)
        self.assertLessEqual(len(result["players"][0]["paths"]["head"]), 101)
        self.assertEqual(result["players"][0]["samples"]["head"], 1000)


class HeatmapTest(unittest.TestCase):
    def test_all_ink_is_distributed_over_the_voxels(self):
        heatmap = build_heatmap(make_drawing([make_line("a", straight(1.0, 11))]), voxel_size=0.1)
        self.assertAlmostEqual(sum(cell[3] for cell in heatmap["cells"]), 1.0, places=3)
        self.assertAlmostEqual(heatmap["totalInk"], 1.0, places=3)
        self.assertIn(len(heatmap["cells"]), (10, 11, 12))
        self.assertEqual(heatmap["voxelSize"], 0.1)
        self.assertEqual(heatmap["dims"][1:], [1, 1])
        self.assertLessEqual(heatmap["maxWeight"], 0.2)

    def test_cells_are_relative_to_the_origin(self):
        heatmap = build_heatmap(make_drawing([make_line("a", straight(0.5, 6, x0=2.0))]), voxel_size=0.1)
        self.assertTrue(all(cell[0] < heatmap["dims"][0] for cell in heatmap["cells"]))
        self.assertAlmostEqual(heatmap["origin"][0], 2.0, delta=0.1)

    def test_erased_strokes_are_not_in_the_heatmap(self):
        heatmap = build_heatmap(make_drawing([make_line("a", straight(1.0), status=0)]), voxel_size=0.1)
        self.assertEqual(heatmap["cells"], [])

    def test_a_fine_grid_is_coarsened(self):
        long_line = make_line("a", straight(500.0, 1000), stats=False)  # 50,000 cells at a 1 cm voxel
        heatmap = build_heatmap(make_drawing([long_line]), voxel_size=0.01)
        self.assertGreater(heatmap["voxelSize"], 0.01)
        self.assertLessEqual(len(heatmap["cells"]), 40_000)


class MultiplayerTest(unittest.TestCase):
    def test_contribution_and_proximity(self):
        lines = [make_line("a", straight(1.0), author="p1"),
                 make_line("b", straight(1.0, z=0.02), author="p2"),  # 2 cm beside the first
                 make_line("c", straight(1.0, z=1.0), author="p3")]   # a metre away
        result = analyze_multiplayer(make_drawing(lines), proximity=0.05)
        self.assertTrue(result["multiAuthor"])
        self.assertEqual([a["playerId"] for a in result["authors"]], ["p1", "p2", "p3"])
        self.assertAlmostEqual(sum(a["inkShare"] for a in result["authors"]), 1.0, places=3)
        pairs = {(p["a"], p["b"]): p for p in result["proximity"]["pairs"]}
        self.assertAlmostEqual(pairs[("p1", "p2")]["aNearB"], 1.0)
        self.assertEqual(pairs[("p1", "p2")]["touchingStrokesA"], 1)
        self.assertAlmostEqual(pairs[("p1", "p3")]["aNearB"], 0.0)
        self.assertAlmostEqual(pairs[("p1", "p3")]["minDistance"], 1.0)

    def test_single_author(self):
        result = analyze_multiplayer(make_drawing([make_line("a", straight())]))
        self.assertFalse(result["multiAuthor"])
        self.assertEqual(result["proximity"]["pairs"], [])

    def test_turn_taking_in_turns(self):
        lines = [make_line("a", straight(), author="p1"),
                 make_line("b", straight(), author="p2", start_ms=T0 + 1200),
                 make_line("c", straight(), author="p1", start_ms=T0 + 2400)]
        turns = analyze_multiplayer(make_drawing(lines))["turns"]
        self.assertEqual([b["playerId"] for b in turns["blocks"]], ["p1", "p2", "p1"])
        self.assertEqual(turns["switches"], 2)
        self.assertEqual(turns["simultaneousMs"], 0)

    def test_simultaneous_drawing(self):
        lines = [make_line("a", straight(), author="p1"), make_line("b", straight(), author="p2", start_ms=T0 + 500)]
        turns = analyze_multiplayer(make_drawing(lines))["turns"]
        self.assertEqual(turns["simultaneousMs"], 500)  # both were drawing between 500 and 1000 ms
        self.assertAlmostEqual(turns["simultaneousShare"], 500 / 1500, places=3)

    def test_consecutive_strokes_of_one_author_form_one_turn(self):
        lines = [make_line("a", straight(), author="p1"), make_line("b", straight(), author="p1", start_ms=T0 + 2000)]
        blocks = analyze_multiplayer(make_drawing(lines))["turns"]["blocks"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual((blocks[0]["startMs"], blocks[0]["endMs"]), (0, 3000))


class GeometryTest(unittest.TestCase):
    def test_simplify_removes_collinear_points_and_keeps_corners(self):
        line = [(i * 0.01, 0.0, 0.0) for i in range(50)]
        self.assertEqual(simplify_indices(line, 0.001), [0, 49])
        corner = line + [(0.49, i * 0.01, 0.0) for i in range(1, 30)]
        self.assertIn(49, simplify_indices(corner, 0.001))  # the corner survives
        self.assertEqual(simplify_indices(line, 0), list(range(50)))

    def test_geometry_has_relative_times_and_the_metadata(self):
        drawing = make_drawing([make_line("a", straight(), start_ms=T0), make_line("b", straight(), status=0, start_ms=T0 + 3000)],
                               placed=[{"modelName": "Tree", "positionX": 1, "positionY": 0, "positionZ": 2}])
        geometry = stroke_geometry(drawing, 0)
        self.assertEqual([s["erased"] for s in geometry["strokes"]], [False, True])
        self.assertEqual(geometry["strokes"][0]["points"][0][3], 0)
        self.assertEqual(geometry["strokes"][1]["points"][0][3], 3000)
        self.assertEqual(geometry["pointCountOriginal"], 22)
        self.assertEqual(geometry["metadata"]["owner"], "p1")
        self.assertEqual(geometry["placedModels"][0]["modelName"], "Tree")

    def test_tolerance_reduces_the_points_and_reports_it(self):
        drawing = make_drawing([make_line("a", straight(1.0, 200))])
        geometry = stroke_geometry(drawing, 0.001)
        self.assertEqual(geometry["pointCountReturned"], 2)
        self.assertEqual(geometry["pointCountOriginal"], 200)

    def test_legacy_drawing_has_times_from_the_legacy_stamps(self):
        drawing = make_drawing([make_line("a", straight(), utc=False)], schema=None)
        points = stroke_geometry(drawing, 0)["strokes"][0]["points"]
        self.assertEqual((points[0][3], points[-1][3]), (0, 1000))

    def test_svg_snapshot_has_the_strokes_and_survives_an_empty_drawing(self):
        self.assertIn("<polyline", drawing_svg(make_drawing([make_line("a", straight())]), "front"))
        for view in ("front", "top", "side"):
            self.assertTrue(drawing_svg(make_drawing([]), view).startswith("<svg"))


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.drawing = make_drawing([make_line("a", [(0, 1, 0.5), (1, 1, 0.5), (1, 2, 0.5)], color=(255, 0, 0)),
                                     make_line("b", straight(), status=0)])

    def test_csv_dialects(self):
        rows = [{"drawingId": "d1", "drawingName": "Rose, red", "totalLength": 3.5, "strokesDrawn": 2, "age": 9}]
        plain = session_csv(rows, "r").decode("utf-8")
        header, first = plain.split("\r\n")[:2]
        self.assertTrue(header.startswith("drawingId,drawingName,sessionId"))
        self.assertIn('"Rose, red"', first)  # a comma in a value is quoted
        self.assertIn("3.5", first)
        excel = session_csv(rows, "excel")
        self.assertTrue(excel.startswith(b"\xef\xbb\xbf"))  # byte order mark for Excel
        text = excel.decode("utf-8-sig")
        self.assertTrue(text.split("\r\n")[0].startswith("drawingId;drawingName;sessionId"))
        self.assertIn("3,5", text.split("\r\n")[1])
        with self.assertRaises(ValueError):
            session_csv(rows, "tsv")

    def test_gltf_is_valid_and_not_mirrored(self):
        content, media_type, extension = export_drawing(self.drawing, "gltf")
        gltf = json.loads(content)
        self.assertEqual(gltf["asset"]["version"], "2.0")
        self.assertEqual(extension, ".gltf")
        self.assertEqual(len(gltf["meshes"][0]["primitives"]), 1)  # the erased stroke is left out
        buffer = base64.b64decode(gltf["buffers"][0]["uri"].split(",", 1)[1])
        self.assertEqual(len(buffer), gltf["buffers"][0]["byteLength"])
        first = struct.unpack("<3f", buffer[:12])
        self.assertEqual(first, (0.0, 1.0, -0.5))  # z negated for a right-handed system
        position = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
        self.assertEqual((position["count"], position["min"][1], position["max"][1]), (3, 1.0, 2.0))
        self.assertEqual(len(export_drawing(self.drawing, "gltf", include_erased=True)[0]) > len(content), True)

    def test_empty_drawing_gives_a_valid_gltf(self):
        gltf = json.loads(export_drawing(make_drawing([]), "gltf")[0])
        self.assertEqual(gltf["asset"]["version"], "2.0")
        self.assertNotIn("buffers", gltf)

    def test_obj_and_ply(self):
        obj = export_drawing(self.drawing, "obj")[0].decode()
        self.assertEqual(sum(1 for line in obj.splitlines() if line.startswith("v ")), 3)
        self.assertIn("l 1 2 3", obj)
        self.assertIn("v 0.00000 1.00000 -0.50000 1.0000 0.0000 0.0000", obj)
        ply = export_drawing(self.drawing, "ply")[0].decode()
        self.assertIn("element vertex 3", ply)
        self.assertIn("element edge 2", ply)
        self.assertTrue(ply.splitlines()[-1] == "1 2")
        with self.assertRaises(ValueError):
            export_drawing(self.drawing, "stl")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.store = AnalysisStore(self._directory.name)

    def test_identifiers_cannot_escape_the_folder(self):
        self.assertFalse(is_valid_id("../evil"))
        self.assertFalse(is_valid_id("a/b"))
        self.assertFalse(is_valid_id(""))
        self.assertTrue(is_valid_id("default-session"))
        self.assertTrue(is_valid_id("3f2c9a1e-8d41-4b57-9c0a-2f5d7e1b6a90"))
        for bad in ("../evil", "a\\b", "x" * 80):
            with self.assertRaises(ValueError):
                self.store.get_analysis(bad, "abcdef0123456789", "1.0")
        with self.assertRaises(ValueError):
            self.store.get_report("../../etc/passwd")
        with self.assertRaises(ValueError):
            self.store.delete_report("not-a-report-id")

    def test_analysis_cache_is_per_content(self):
        self.store.put_analysis("d1", "aaaaaaaaaaaaaaaa", "1.0", {"v": 1})
        self.assertEqual(self.store.get_analysis("d1", "aaaaaaaaaaaaaaaa", "1.0"), {"v": 1})
        self.assertIsNone(self.store.get_analysis("d1", "bbbbbbbbbbbbbbbb", "1.0"))  # changed content
        self.assertIsNone(self.store.get_analysis("d1", "aaaaaaaaaaaaaaaa", "2.0"))  # other analyzer version
        self.store.put_analysis("d1", "bbbbbbbbbbbbbbbb", "1.0", {"v": 2})
        self.assertIsNone(self.store.get_analysis("d1", "aaaaaaaaaaaaaaaa", "1.0"))  # the old result is removed

    def test_reports_round_trip(self):
        report_id = "0123456789abcdef0123456789abcdef"
        meta = {"id": report_id, "kind": "drawing", "targetId": "d1", "title": "T", "createdAt": "2026-10-01T10:00:00+00:00"}
        self.store.save_report(meta, "<html>x</html>")
        self.assertEqual(self.store.get_report(report_id), meta)
        self.assertEqual(self.store.read_report_html(report_id), "<html>x</html>")
        self.assertEqual(self.store.find_report("drawing", "d1"), meta)
        self.assertEqual([r["id"] for r in self.store.list_reports()], [report_id])
        self.assertTrue(self.store.delete_report(report_id))
        self.assertFalse(self.store.delete_report(report_id))
        self.assertIsNone(self.store.read_report_html(report_id))

    def test_thumbnail_cache(self):
        self.store.put_thumbnail("d1", "aaaaaaaaaaaaaaaa", "<svg/>")
        self.assertEqual(self.store.get_thumbnail("d1", "aaaaaaaaaaaaaaaa"), "<svg/>")
        self.store.put_thumbnail("d1", "bbbbbbbbbbbbbbbb", "<svg>2</svg>")
        self.assertIsNone(self.store.get_thumbnail("d1", "aaaaaaaaaaaaaaaa"))


class ReportTest(unittest.TestCase):
    EVIL_TITLE = "<script>alert(1)</script>"
    EVIL_NAME = '"><img src=x onerror=alert(2)>'

    def setUp(self):
        self.drawing = make_drawing([make_line("a", straight(), pressure=0.4), make_line("b", straight(), color=(40, 60, 220))])
        self.analysis = analyze_drawing(self.drawing)
        self.player = {"id": "p1", "name": self.EVIL_NAME, "age": 9, "gender": "F", "dominantHand": "Right"}

    @staticmethod
    def external_urls(html: str) -> list[str]:
        import re
        return re.findall(r"https?://[^\s\"'<)]+", html.replace("http://www.w3.org/2000/svg", ""))

    def test_drawing_report_escapes_everything_from_the_data(self):
        html = reports.drawing_report(self.drawing, self.analysis, build_narrative(self.analysis, "Right"),
                                      self.EVIL_TITLE, self.player, {"name": "<b>Session</b>"})
        self.assertNotIn("<script>alert(1)", html)
        self.assertNotIn("<img src=x", html)
        self.assertNotIn("<b>Session</b>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_report_is_self_contained(self):
        html = reports.drawing_report(self.drawing, self.analysis, [], "T", self.player, None)
        self.assertEqual(self.external_urls(html), [])
        self.assertIn("<svg", html)  # static snapshots
        self.assertIn('id="viewer-canvas"', html)  # the inline 3D view
        self.assertIn("Total length", html)
        self.assertIn("not a diagnosis", html)

    def test_session_and_participant_reports(self):
        stats = {"sessionId": "s1", "drawingCount": 1, "distributions": [], "perParticipant": [
            {"playerId": "p1", "drawings": 1, "metrics": {"strokesDrawn": 2, "totalLength": 2.0, "activeTime": 2.0}}]}
        gallery = [{"name": self.EVIL_NAME, "playerId": "p1", "gameType": 2, "svg": drawing_svg(self.drawing)}]
        session = reports.session_report({"name": self.EVIL_TITLE, "description": "d", "startDate": "x", "endDate": None},
                                         stats, {"p1": self.player}, gallery, "1.0")
        self.assertNotIn("<script>alert(1)", session)
        self.assertNotIn("<img src=x", session)
        history = [{"name": "A", "sessionName": "S", "gameType": 2, "date": "2026.10.01-10:00",
                    "metrics": {"strokesDrawn": 2, "totalLength": 2.0, "activeTime": 2.0}},
                   {"name": "B", "sessionName": "S", "gameType": 2, "date": "2026.10.02-10:00",
                    "metrics": {"strokesDrawn": 4, "totalLength": 3.0, "activeTime": 2.0}}]
        participant = reports.participant_report(self.player, history, "1.0")
        self.assertNotIn("<img src=x", participant)
        self.assertIn("+100 %", participant)  # strokes 2 -> 4
        self.assertIn("+50 %", participant)  # length 2 -> 3
        self.assertEqual(self.external_urls(participant), [])


if __name__ == "__main__":
    unittest.main()
