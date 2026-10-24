"""
The behaviour timeline of a drawing (requirement DRW-07): strokes, erasing and undoing, menu use, tool changes and
the pauses, all on one time axis (milliseconds from the start of the drawing's activity).
"""

from typing import Any

from analysis.common import (
    Drawing, IDLE_GAP_SECONDS, TimeAxis, all_lines, drawn_lines, history_time, idle_periods, legacy_offset_ms,
    stroke_times, time_base,
)

# Line history event types (LineEventType) and tool event types (DrawingEventType) of the game
HISTORY_ERASE = 0
HISTORY_UNDO = 2
EVENT_KINDS = {0: "menuOpen", 1: "menuClose", 2: "modeChange", 3: "colorChange", 4: "widthChange"}


def build_timeline(doc: Drawing, idle_gap_seconds: float = IDLE_GAP_SECONDS) -> dict[str, Any]:
    axis = TimeAxis(doc)
    offset = legacy_offset_ms(doc)
    items: list[dict[str, Any]] = []
    end_of_activity = 0

    for line in all_lines(doc):
        times = stroke_times(line)
        author = line.get("userID")
        hand = line.get("hand")
        if times:
            items.append({"kind": "stroke", "lineId": line.get("id"), "authorId": author, "hand": hand,
                          "startMs": axis.map(times[0]), "endMs": axis.map(times[1]),
                          "erased": line.get("status") != 1})
            end_of_activity = max(end_of_activity, axis.map(times[1]))
        # The history only has the legacy stamp: the offset puts it on the same axis as the strokes
        for event in line.get("history") or []:
            kind = {HISTORY_ERASE: "erase", HISTORY_UNDO: "undo"}.get(event.get("eventType"))
            at = history_time(event.get("timestamp"), offset)
            if kind and at is not None:
                items.append({"kind": kind, "lineId": line.get("id"), "authorId": event.get("invoker"),
                              "hand": event.get("hand"), "atMs": axis.map(at)})
                end_of_activity = max(end_of_activity, axis.map(at))

    for event in doc.get("events") or []:
        kind = EVENT_KINDS.get(event.get("type"))
        if kind and event.get("utcMs") is not None:
            items.append({"kind": kind, "playerId": event.get("playerID"), "hand": event.get("hand"),
                          "atMs": axis.map(event["utcMs"]), "value": event.get("value")})
            end_of_activity = max(end_of_activity, axis.map(event["utcMs"]))

    drawn_times = [t for t in (stroke_times(line) for line in drawn_lines(doc)) if t]
    pauses = idle_periods(drawn_times, idle_gap_seconds * 1000)
    for start, end in pauses:
        items.append({"kind": "idle", "startMs": axis.map(start), "endMs": axis.map(end)})

    items.sort(key=lambda item: item.get("startMs", item.get("atMs", 0)))

    def count(*kinds: str) -> int:
        return sum(1 for item in items if item["kind"] in kinds)

    summary = {
        "strokes": sum(1 for i in items if i["kind"] == "stroke" and not i["erased"]),
        "erasedStrokes": sum(1 for i in items if i["kind"] == "stroke" and i["erased"]),
        "erases": count("erase"),
        "undos": count("undo"),
        "menuOpens": count("menuOpen"),
        "toolChanges": count("modeChange", "colorChange", "widthChange"),
        "idlePeriods": len(pauses),
        "idleSeconds": sum(end - start for start, end in pauses) / 1000,
    }
    return {
        "drawingId": (doc.get("metadata") or {}).get("id"),
        "originMs": axis.origin,
        "timeBase": time_base(doc),
        "breaks": axis.break_marks(),  # breaks between sittings, cut down on the axis
        "durationMs": end_of_activity,
        "hasTimeData": bool(drawn_times),
        "idleGapSeconds": idle_gap_seconds,
        "items": items,
        "summary": summary,
    }
