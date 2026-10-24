import math
import unittest

from analysis import catalog
from analysis.cohort import MIN_COHORT, compare_with_cohort, percentile_rank, session_stats
from analysis.common import number, round_half_up
from analysis.metrics import analyze_drawing, metric_values
from analysis.narrative import build_narrative, render_sentence
from tests.fixtures import T0, make_drawing, make_line, straight


def values(drawing, **kwargs):
    return metric_values(analyze_drawing(drawing, **kwargs))


def reasons(drawing):
    return {m["id"]: m.get("unavailableReason") for m in analyze_drawing(drawing)["metrics"]}


class MetricsTest(unittest.TestCase):
    def test_straight_line(self):
        v = values(make_drawing([make_line("a", straight(1.0, 11), pressure=0.5)]))
        self.assertEqual(v["strokesDrawn"], 1)
        self.assertAlmostEqual(v["totalLength"], 1.0)
        self.assertAlmostEqual(v["meanStrokeLength"], 1.0)
        self.assertAlmostEqual(v["drawingSpan"], 1.0)  # 10 steps of 100 ms
        self.assertAlmostEqual(v["activeTime"], 1.0)
        self.assertAlmostEqual(v["meanSpeed"], 1.0)  # 1 m in 1 s
        self.assertAlmostEqual(v["directionalChange"], 0.0)
        self.assertAlmostEqual(v["meanPressure"], 0.5)
        self.assertAlmostEqual(v["boundingVolume"], 0.0)  # a flat line has no volume
        self.assertAlmostEqual(v["heightRange"], 0.0)
        self.assertEqual(v["pointCount"], 11)

    def test_catalogue_order_and_completeness(self):
        analysis = analyze_drawing(make_drawing([make_line("a", straight())]))
        self.assertEqual([m["id"] for m in analysis["metrics"]], catalog.METRIC_IDS)
        self.assertEqual(len(catalog.METRIC_IDS), 20)

    def test_square_turns_a_right_angle_at_every_corner(self):
        square = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
        self.assertAlmostEqual(values(make_drawing([make_line("a", square)]))["directionalChange"], 0.5)

    def test_bounding_box_and_height(self):
        box = [(0, 0, 0), (2, 3, 4)]
        v = values(make_drawing([make_line("a", box, step_ms=100)]))
        self.assertAlmostEqual(v["boundingVolume"], 24.0)
        self.assertAlmostEqual(v["heightRange"], 3.0)

    def test_hand_share_counts_strokes(self):
        lines = [make_line("a", straight(), hand=-1), make_line("b", straight(), hand=1), make_line("c", straight(), hand=1)]
        self.assertAlmostEqual(values(make_drawing(lines))["leftHandShare"], 100 / 3)

    def test_erased_strokes_are_behaviour_not_drawing(self):
        lines = [make_line("a", straight(1.0)), make_line("b", straight(5.0), status=0)]
        v = values(make_drawing(lines))
        self.assertEqual(v["strokesDrawn"], 1)
        self.assertEqual(v["strokesErased"], 1)
        self.assertAlmostEqual(v["totalLength"], 1.0)  # the erased 5 m do not count
        self.assertAlmostEqual(v["eraseRatio"], 50.0)

    def test_colour_shares_use_ink_length(self):
        lines = [make_line("red", straight(2.0), color=(220, 40, 40)), make_line("blue", straight(1.0), color=(40, 60, 220))]
        analysis = analyze_drawing(make_drawing(lines))
        v = metric_values(analysis)
        self.assertAlmostEqual(v["warmShare"], 200 / 3)
        self.assertEqual(v["distinctColours"], 2)
        self.assertEqual([c["hex"] for c in analysis["palette"]], ["#dc2828", "#283cdc"])
        self.assertAlmostEqual(analysis["palette"][0]["share"], 2 / 3)

    def test_grey_ink_has_no_warm_share(self):
        drawing = make_drawing([make_line("a", straight(), color=(90, 90, 100))])
        self.assertIsNone(values(drawing)["warmShare"])
        self.assertEqual(reasons(drawing)["warmShare"], "noChromaticInk")

    def test_overlapping_strokes_count_once_in_active_time(self):
        lines = [make_line("a", straight(), author="p1"), make_line("b", straight(), author="p2", start_ms=T0 + 500)]
        v = values(make_drawing(lines))
        self.assertAlmostEqual(v["activeTime"], 1.5)  # the union, not 2.0
        self.assertAlmostEqual(v["drawingSpan"], 1.5)

    def test_pauses_longer_than_the_limit(self):
        lines = [make_line("a", straight()), make_line("b", straight(), start_ms=T0 + 9000),
                 make_line("c", straight(), start_ms=T0 + 11000)]
        drawing = make_drawing(lines)
        self.assertAlmostEqual(values(drawing)["idleTime"], 8.0)  # the 1 s pause is below the 5 s limit
        self.assertAlmostEqual(values(drawing, idle_gap_seconds=0.5)["idleTime"], 9.0)

    def test_a_break_between_sittings_is_not_a_pause(self):
        # A real drawing was continued two days later: the days must not become "pause time" or "time span"
        two_days = 2 * 24 * 3600 * 1000
        lines = [make_line("a", straight()), make_line("b", straight(), start_ms=T0 + two_days)]
        analysis = analyze_drawing(make_drawing(lines))
        v = metric_values(analysis)
        self.assertAlmostEqual(v["drawingSpan"], 2.0)  # the two strokes of 1 s each, the break cut out
        self.assertAlmostEqual(v["activeTime"], 2.0)
        self.assertAlmostEqual(v["idleTime"], 0.0)
        self.assertIn("multipleSittings", analysis["flags"])

    def test_a_pause_just_below_the_break_limit_is_still_a_pause(self):
        pause = 29 * 60 * 1000
        lines = [make_line("a", straight()), make_line("b", straight(), start_ms=T0 + 1000 + pause)]
        analysis = analyze_drawing(make_drawing(lines))
        self.assertAlmostEqual(metric_values(analysis)["idleTime"], pause / 1000)
        self.assertNotIn("multipleSittings", analysis["flags"])

    def test_tool_events(self):
        events = [{"type": 0, "playerID": "p1", "hand": 1, "utcMs": T0}, {"type": 0, "playerID": "p1", "hand": 1, "utcMs": T0 + 5},
                  {"type": 3, "playerID": "p1", "hand": 1, "utcMs": T0 + 9, "value": "x"},
                  {"type": 4, "playerID": "p1", "hand": 1, "utcMs": T0 + 11}, {"type": 1, "playerID": "p1", "hand": 1, "utcMs": T0 + 12}]
        v = values(make_drawing([make_line("a", straight())], events=events))
        self.assertEqual(v["menuOpens"], 2)
        self.assertEqual(v["toolChanges"], 2)  # colour and width, the menu close is neither

    def test_schema_two_without_events_means_none_happened(self):
        self.assertEqual(values(make_drawing([make_line("a", straight())]))["menuOpens"], 0)

    def test_legacy_drawing_never_gets_a_made_up_value(self):
        line = make_line("a", straight(), utc=False, stats=False)
        drawing = make_drawing([line], schema=None)
        analysis = analyze_drawing(drawing)
        v = metric_values(analysis)
        self.assertIsNone(v["meanPressure"])
        self.assertEqual(reasons(drawing)["meanPressure"], "noPressure")
        self.assertIsNone(v["menuOpens"])
        self.assertEqual(reasons(drawing)["menuOpens"], "noEvents")
        self.assertAlmostEqual(v["drawingSpan"], 1.0)  # the legacy stamps still give time differences
        self.assertAlmostEqual(v["totalLength"], 1.0)  # no stored stats: derived from the points
        self.assertIn("legacySchema", analysis["flags"])

    def test_nothing_drawn(self):
        drawing = make_drawing([make_line("a", straight(), status=0)])
        analysis = analyze_drawing(drawing)
        v = metric_values(analysis)
        self.assertIsNone(v["totalLength"])
        self.assertEqual(reasons(drawing)["totalLength"], "noStrokes")
        self.assertEqual(v["strokesDrawn"], 0)
        self.assertEqual(analysis["palette"], [])
        self.assertIn("allErased", analysis["flags"])

    def test_a_tap_has_no_ink_and_does_not_crash(self):
        # A stroke whose points all coincide (the trigger was tapped without moving) has length 0
        drawing = make_drawing([make_line("a", [(0, 1, 0), (0, 1, 0), (0, 1, 0)])])
        analysis = analyze_drawing(drawing)
        v = metric_values(analysis)
        self.assertEqual(v["strokesDrawn"], 1)
        self.assertEqual(v["totalLength"], 0.0)
        self.assertIsNone(v["distinctColours"])
        self.assertIsNone(v["warmShare"])
        self.assertEqual(reasons(drawing)["distinctColours"], "noInk")
        self.assertEqual(reasons(drawing)["warmShare"], "noInk")
        self.assertEqual(analysis["palette"], [])

    def test_empty_drawing_does_not_crash(self):
        analysis = analyze_drawing(make_drawing([]))
        self.assertEqual(metric_values(analysis)["strokesDrawn"], 0)
        self.assertAlmostEqual(metric_values(analysis)["eraseRatio"], 0.0)

    def test_figure_distance_needs_the_figure(self):
        line = make_line("a", [(0, 0, 0), (0, 0, 0)])
        self.assertEqual(reasons(make_drawing([line]))["figureDistance"], "noFigurePosition")
        near = values(make_drawing([line], figure={"x": 0.0, "y": 3.0, "z": 4.0}))["figureDistance"]
        self.assertAlmostEqual(near, 5.0)

    def test_quality_flags(self):
        few = analyze_drawing(make_drawing([make_line("a", straight())], decimation=3))["flags"]
        self.assertIn("fewStrokes", few)
        self.assertIn("thinnedTracking", few)
        many = analyze_drawing(make_drawing([make_line(str(i), straight()) for i in range(3)], game_type=2))["flags"]
        self.assertEqual(many, [])
        inner_child = analyze_drawing(make_drawing([make_line(str(i), straight()) for i in range(3)], game_type=0))["flags"]
        self.assertEqual(inner_child, ["missingFigurePosition"])

    def test_unknown_game_type_is_labelled(self):
        self.assertEqual(catalog.game_type_name(9), "Unknown (9)")
        self.assertEqual(catalog.game_type_name(4), "Bilateral drawing")

    def test_halves_round_up_like_on_screen(self):
        # Python's round() would give 0.12, 2 and 81.2 here: the web app shows 0.13, 3 and 81.3
        self.assertEqual(round_half_up(0.125, 2), 0.13)
        self.assertEqual(round_half_up(2.5), 3)
        self.assertEqual(number(81.25, 1), 81.3)
        self.assertEqual(catalog.format_number(0.125, 2), "0.13")
        self.assertEqual(catalog.format_duration(2.5), "3 s")
        self.assertEqual(number(-0.0), 0.0)

    def test_formatting(self):
        self.assertEqual(catalog.format_metric("totalLength", 1.2345), "1.23 m")
        self.assertEqual(catalog.format_metric("activeTime", 125), "2 min 05 s")
        self.assertEqual(catalog.format_metric("activeTime", 45), "45 s")
        self.assertEqual(catalog.format_duration(3900), "1 h 05 min")
        self.assertEqual(catalog.format_duration(3 * 86400 + 2 * 3600 + 30), "3 d 2 h")
        self.assertEqual(catalog.format_metric("strokesDrawn", 3), "3")
        self.assertEqual(catalog.format_metric("totalLength", None), "–")
        self.assertEqual(catalog.format_metric("boundingVolume", 1234.5), "1,234.5 m³")


class NarrativeTest(unittest.TestCase):
    def test_sentences_follow_the_numbers(self):
        analysis = analyze_drawing(make_drawing([make_line("a", straight(), pressure=0.4), make_line("b", straight(), status=0)]))
        items = build_narrative(analysis, "Right")
        ids = [item["id"] for item in items]
        self.assertIn("activityTimed", ids)
        self.assertIn("handDominant", ids)
        self.assertIn("erase", ids)
        self.assertIn("tools", ids)
        for item in items:
            self.assertNotIn("{", render_sentence(item))  # every placeholder is filled

    def test_missing_numbers_leave_the_sentence_out(self):
        drawing = make_drawing([make_line("a", straight(), utc=False, stats=False)], schema=None)
        ids = [item["id"] for item in build_narrative(analyze_drawing(drawing))]
        self.assertIn("activityTimed", ids)  # legacy stamps still give times
        self.assertNotIn("tools", ids)  # no events in a legacy drawing
        self.assertIn("quality", ids)


class CohortTest(unittest.TestCase):
    def test_percentile_rank_counts_ties_half(self):
        self.assertAlmostEqual(percentile_rank([1, 2, 3, 4, 5, 6, 7, 8, 9], 5), 50.0)
        self.assertAlmostEqual(percentile_rank([1, 2, 3], 10), 100.0)
        self.assertAlmostEqual(percentile_rank([1, 2, 3], 0), 0.0)

    def test_small_cohort_gives_no_percentile(self):
        small = compare_with_cohort({"totalLength": 2.0}, [{"totalLength": 1.0}, {"totalLength": 3.0}])
        self.assertFalse(small["sufficient"])
        self.assertIsNone(next(m for m in small["metrics"] if m["id"] == "totalLength")["percentile"])

    def test_cohort_percentile_and_range(self):
        cohort = [{"totalLength": float(v)} for v in range(1, MIN_COHORT + 3)]  # 1..5
        row = next(m for m in compare_with_cohort({"totalLength": 3.0}, cohort)["metrics"] if m["id"] == "totalLength")
        self.assertAlmostEqual(row["percentile"], 50.0)
        self.assertEqual((row["min"], row["median"], row["max"], row["cohortSize"]), (1.0, 3.0, 5.0, 5))

    def test_metric_the_subject_lacks_has_no_percentile(self):
        cohort = [{"meanPressure": 0.5}] * 4
        row = next(m for m in compare_with_cohort({"meanPressure": None}, cohort)["metrics"] if m["id"] == "meanPressure")
        self.assertIsNone(row["percentile"])

    def test_session_stats(self):
        a = analyze_drawing(make_drawing([make_line("a", straight(1.0))], drawing_id="d1"))
        b = analyze_drawing(make_drawing([make_line("b", straight(3.0))], drawing_id="d2"))
        stats = session_stats("s1", [("p1", a), ("p2", b), ("p1", b)])
        length = next(d for d in stats["distributions"] if d["id"] == "totalLength")
        self.assertEqual((length["min"], length["max"], length["count"]), (1.0, 3.0, 3))
        self.assertAlmostEqual(length["median"], 3.0)
        p1 = next(p for p in stats["perParticipant"] if p["playerId"] == "p1")
        self.assertEqual(p1["drawings"], 2)
        self.assertAlmostEqual(p1["metrics"]["totalLength"], 2.0)  # the mean of 1 m and 3 m
        self.assertTrue(math.isfinite(p1["metrics"]["meanSpeed"]))


if __name__ == "__main__":
    unittest.main()
