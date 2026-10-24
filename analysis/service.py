"""
Data access of the analysis API: reads the existing SQLite tables and the MongoDB drawings and hands plain dicts to the
pure analysis functions. It never writes to either database (SC-02). Only the file store (analysis.store) is written.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

import database
from analysis import reports
from analysis.cohort import session_stats
from analysis.geometry import drawing_svg
from analysis.metrics import ANALYZER_VERSION, analyze_drawing, metric_values
from analysis.narrative import build_narrative
from analysis.store import AnalysisStore, new_report_id

REPORT_KINDS = ("drawing", "session", "participant")


# --- MongoDB ---

def _collection():
    collection = database.mongo_collection
    if collection is None:
        raise RuntimeError("MongoDB is not connected")
    return collection


def load_drawing(drawing_id: str) -> Optional[dict[str, Any]]:
    return _collection().find_one({"metadata.id": drawing_id}, {"_id": 0})


def drawing_owner(drawing_id: str) -> Optional[str]:
    document = _collection().find_one({"metadata.id": drawing_id}, {"metadata.owner": 1})
    return document["metadata"]["owner"] if document else None


def drawing_dates() -> dict[str, str]:
    """{drawing id: date} of every drawing, read without the strokes."""

    return {
        row["id"]: row["date"]
        for row in _collection().aggregate([{"$project": {"_id": 0, "id": "$metadata.id", "date": "$metadata.date"}}])
    }


def drawing_fingerprint(drawing_id: str) -> Optional[str]:
    """
    A cheap fingerprint of the content of a drawing (requirement TEC-04): the stroke ids, statuses and point counts,
    the number of events and tracked samples, and the metadata that matters. Computed by MongoDB from a projection, so
    the (up to 15 MB) document is not transferred. It changes whenever the drawing is updated.
    """

    rows = list(_collection().aggregate([
        {"$match": {"metadata.id": drawing_id}},
        {"$project": {
            "_id": 0,
            "lines": {"$map": {"input": {"$ifNull": ["$lines", []]}, "as": "l", "in": {
                "i": "$$l.id", "s": "$$l.status", "n": {"$size": {"$ifNull": ["$$l.points", []]}}}}},
            "events": {"$size": {"$ifNull": ["$events", []]}},
            "tracked": {"$map": {"input": {"$ifNull": ["$trackedBehaviors", []]}, "as": "b", "in": {
                "$size": {"$ifNull": ["$$b.head", []]}}}},
            "models": {"$size": {"$ifNull": ["$placedModels", []]}},
            "meta": ["$metadata.schemaVersion", "$metadata.trackingDecimation", "$metadata.sessionID",
                     "$metadata.gameType", "$metadata.sadChildCoordinates"],
        }},
    ]))
    if not rows:
        return None
    return hashlib.sha256(json.dumps(rows[0], sort_keys=True, default=str).encode()).hexdigest()[:16]


def drawing_analysis(drawing_id: str, store: AnalysisStore) -> Optional[tuple[dict[str, Any], str]]:
    """The metrics of a drawing and its fingerprint. A stored result of the same content and version is reused."""

    fingerprint = drawing_fingerprint(drawing_id)
    if fingerprint is None:
        return None
    cached = store.get_analysis(drawing_id, fingerprint, ANALYZER_VERSION)
    if cached:
        return cached, fingerprint
    document = load_drawing(drawing_id)
    if document is None:
        return None
    analysis = analyze_drawing(document)
    store.put_analysis(drawing_id, fingerprint, ANALYZER_VERSION, analysis)
    return analysis, fingerprint


def drawing_thumbnail(drawing_id: str, store: AnalysisStore) -> Optional[str]:
    fingerprint = drawing_fingerprint(drawing_id)
    if fingerprint is None:
        return None
    cached = store.get_thumbnail(drawing_id, fingerprint)
    if cached:
        return cached
    document = load_drawing(drawing_id)
    if document is None:
        return None
    svg = drawing_svg(document, "front")
    store.put_thumbnail(drawing_id, fingerprint, svg)
    return svg


def drawings_over_time(db: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    Drawings per day. Only drawings that also have a row in SQLite count, like everywhere else in the app: a drawing
    that exists in MongoDB alone (a failed or partial save) has no session, no name and no owner to show.
    """

    known = {row[0] for row in db.execute("SELECT ID FROM DrawingMeta")}
    per_day: dict[str, int] = {}
    for drawing_id, date in drawing_dates().items():
        if drawing_id in known and date:
            per_day[date[:10]] = per_day.get(date[:10], 0) + 1
    return [{"date": day, "count": count} for day, count in sorted(per_day.items())]


# --- SQLite ---

def _session(row: tuple) -> dict[str, Any]:
    return {"id": row[0], "name": row[1], "description": row[2], "startDate": row[3], "endDate": row[4],
            "showBoy": row[5] if row[5] is not None else 1, "showGirl": row[6] if row[6] is not None else 1,
            "multiplayer": row[7] if row[7] is not None else 0}


_SESSION_COLUMNS = "ID, Name, Description, StartDate, EndDate, ShowBoy, ShowGirl, Multiplayer"


def session_row(db: sqlite3.Connection, session_id: str) -> Optional[dict[str, Any]]:
    row = db.execute(f"SELECT {_SESSION_COLUMNS} FROM Sessions WHERE ID = ?", (session_id,)).fetchone()
    return _session(row) if row else None


def player_info(db: sqlite3.Connection, player_id: str) -> Optional[dict[str, Any]]:
    row = db.execute(
        "SELECT pi.ID, pi.Name, pi.Gender, pi.Age, pi.DominantHand FROM PlayerInfo pi "
        "JOIN Players p ON pi.ID = p.PlayerInfoID WHERE p.ID = ?", (player_id,)).fetchone()
    return {"id": row[0], "name": row[1], "gender": row[2], "age": row[3], "dominantHand": row[4]} if row else None


def player_infos(db: sqlite3.Connection, player_ids: list[str]) -> dict[str, dict[str, Any]]:
    infos = {}
    for player_id in dict.fromkeys(player_ids):
        info = player_info(db, player_id)
        if info:
            infos[player_id] = info
    return infos


def _meta(row: tuple) -> dict[str, Any]:
    return {"id": row[0], "playerID": row[1], "name": row[2], "path": row[3], "gameType": row[4],
            "sessionID": row[5] or ""}


_META_COLUMNS = "ID, PlayerID, Name, Path, GameType, SessionID"


def drawing_meta(db: sqlite3.Connection, drawing_id: str) -> Optional[dict[str, Any]]:
    row = db.execute(f"SELECT {_META_COLUMNS} FROM DrawingMeta WHERE ID = ?", (drawing_id,)).fetchone()
    return _meta(row) if row else None


def drawing_metas(db: sqlite3.Connection, session_id: Optional[str] = None,
                  player_id: Optional[str] = None) -> list[dict[str, Any]]:
    query = f"SELECT {_META_COLUMNS} FROM DrawingMeta"
    conditions, params = [], []
    if session_id is not None:
        conditions.append("SessionID = ?")
        params.append(session_id)
    if player_id is not None:
        conditions.append("PlayerID = ?")
        params.append(player_id)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    return [_meta(row) for row in db.execute(query + " ORDER BY Name ASC", params).fetchall()]


def session_summaries(db: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every session with its number of drawings, its participants and its drawings per game type, in one pass."""

    sessions = [_session(row) for row in db.execute(
        f"SELECT {_SESSION_COLUMNS} FROM Sessions ORDER BY StartDate ASC").fetchall()]
    summaries = {s["id"]: {**s, "drawingCount": 0, "participantIds": [], "gameTypeCounts": {}} for s in sessions}
    for session_id, player_id, game_type in db.execute("SELECT SessionID, PlayerID, GameType FROM DrawingMeta"):
        summary = summaries.get(session_id)
        if summary is None:
            continue
        summary["drawingCount"] += 1
        if player_id not in summary["participantIds"]:
            summary["participantIds"].append(player_id)
        summary["gameTypeCounts"][str(game_type)] = summary["gameTypeCounts"].get(str(game_type), 0) + 1
    return list(summaries.values())


def overview(db: sqlite3.Connection) -> dict[str, Any]:
    summaries = session_summaries(db)
    active_id = database.get_active_session_id(db)
    by_game_type: dict[int, int] = {}
    for summary in summaries:
        for game_type, count in summary["gameTypeCounts"].items():
            by_game_type[int(game_type)] = by_game_type.get(int(game_type), 0) + count
    open_count = sum(1 for s in summaries if not s["endDate"])
    return {
        "sessions": {"total": len(summaries), "open": open_count, "closed": len(summaries) - open_count},
        "drawings": sum(s["drawingCount"] for s in summaries),
        "participants": len({p for s in summaries for p in s["participantIds"]}),
        "activeSession": next((s for s in summaries if s["id"] == active_id), None),
        "drawingsByGameType": [{"gameType": g, "count": c} for g, c in sorted(by_game_type.items())],
        "recentSessions": sorted(summaries, key=lambda s: s["startDate"] or "", reverse=True)[:5],
    }


def drawing_list(db: sqlite3.Connection, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Every drawing with the names the screens show, newest first (the date is only stored in MongoDB)."""

    dates = drawing_dates()
    sessions = {s["id"]: s["name"] for s in session_summaries(db)}
    players = {row[0]: row[1] for row in db.execute(
        "SELECT p.ID, pi.Name FROM Players p JOIN PlayerInfo pi ON pi.ID = p.PlayerInfoID")}
    items = [{"meta": meta, "playerName": players.get(meta["playerID"]) or meta["playerID"],
              "sessionName": sessions.get(meta["sessionID"], meta["sessionID"]), "date": dates.get(meta["id"], "")}
             for meta in drawing_metas(db)]
    items.sort(key=lambda item: item["date"], reverse=True)
    return items[:limit] if limit else items


def cohort_drawing_ids(
        db: sqlite3.Connection, exclude_id: str, session_id: Optional[str] = None, game_type: Optional[int] = None,
        age_min: Optional[int] = None, age_max: Optional[int] = None, limit: int = 500,
) -> list[str]:
    """The drawings of the chosen group: same session, same game type and/or owners in an age range."""

    query = ("SELECT d.ID FROM DrawingMeta d JOIN Players p ON p.ID = d.PlayerID "
             "JOIN PlayerInfo i ON i.ID = p.PlayerInfoID WHERE d.ID != ?")
    params: list[Any] = [exclude_id]
    if session_id is not None:
        query += " AND d.SessionID = ?"
        params.append(session_id)
    if game_type is not None:
        query += " AND d.GameType = ?"
        params.append(game_type)
    if age_min is not None:
        query += " AND i.Age >= ?"
        params.append(age_min)
    if age_max is not None:
        query += " AND i.Age <= ?"
        params.append(age_max)
    return [row[0] for row in db.execute(query + " LIMIT ?", params + [limit]).fetchall()]


def metrics_of(drawing_ids: list[str], store: AnalysisStore) -> list[dict[str, Optional[float]]]:
    results = []
    for drawing_id in drawing_ids:
        found = drawing_analysis(drawing_id, store)
        if found:
            results.append(metric_values(found[0]))
    return results


def session_csv_rows(db: sqlite3.Connection, store: AnalysisStore, session_id: str) -> list[dict[str, Any]]:
    """One row per drawing of the session for the CSV export: drawing, participant and every metric (EXP-01)."""

    from analysis.catalog import game_type_name  # local import: only the export needs it

    session = session_row(db, session_id) or {}
    metas = drawing_metas(db, session_id=session_id)
    dates = drawing_dates()
    players = player_infos(db, [m["playerID"] for m in metas])
    rows = []
    for meta in metas:
        found = drawing_analysis(meta["id"], store)
        if not found:
            continue
        info = players.get(meta["playerID"], {})
        rows.append({
            "drawingId": meta["id"], "drawingName": meta["name"], "sessionId": session_id,
            "sessionName": session.get("name"), "gameType": game_type_name(meta["gameType"]),
            "date": dates.get(meta["id"], ""), "participantId": meta["playerID"],
            "participantName": info.get("name"), "age": info.get("age"), "gender": info.get("gender"),
            "dominantHand": info.get("dominantHand"), **metric_values(found[0]),
        })
    return rows


# --- Narrative of a drawing ---

def drawing_narrative(db: sqlite3.Connection, drawing_id: str, analysis: dict[str, Any],
                      idle_gap_seconds: Optional[float] = None) -> list[dict[str, Any]]:
    owner = drawing_owner(drawing_id)
    info = player_info(db, owner) if owner else None
    if idle_gap_seconds is None:
        return build_narrative(analysis, (info or {}).get("dominantHand"))
    return build_narrative(analysis, (info or {}).get("dominantHand"), idle_gap_seconds)


# --- Participant history ---

def participant_history(db: sqlite3.Connection, store: AnalysisStore, player_id: str) -> Optional[dict[str, Any]]:
    info = player_info(db, player_id)
    if info is None:
        return None
    dates = drawing_dates()
    sessions = {s["id"]: s["name"] for s in session_summaries(db)}
    rows = []
    for meta in drawing_metas(db, player_id=player_id):
        found = drawing_analysis(meta["id"], store)
        if not found:
            continue
        rows.append({"meta": meta, "sessionName": sessions.get(meta["sessionID"], meta["sessionID"]),
                     "date": dates.get(meta["id"], ""), "metrics": metric_values(found[0]), "flags": found[0]["flags"]})
    rows.sort(key=lambda row: row["date"])
    # The id inside "player" is the id of the PlayerInfo row, which differs from the player's own id
    return {"playerId": player_id, "player": info, "drawings": rows}


# --- Reports (REP-01, REP-04) ---

def target_fingerprint(db: sqlite3.Connection, kind: str, target_id: str) -> Optional[str]:
    """What the report depends on, as a hash. None when the target does not exist (any more)."""

    if kind == "drawing":
        return drawing_fingerprint(target_id)
    if kind == "session":
        session = session_row(db, target_id)
        if session is None:
            return None
        parts: Any = [session, sorted((m["id"], drawing_fingerprint(m["id"])) for m in drawing_metas(db, session_id=target_id))]
    elif kind == "participant":
        info = player_info(db, target_id)
        if info is None:
            return None
        parts = [info, sorted((m["id"], drawing_fingerprint(m["id"])) for m in drawing_metas(db, player_id=target_id))]
    else:
        return None
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _is_stale(db: sqlite3.Connection, meta: dict[str, Any]) -> bool:
    current = target_fingerprint(db, meta["kind"], meta["targetId"])
    return (current is None or current != meta.get("fingerprint")
            or meta.get("analyzerVersion") != ANALYZER_VERSION or meta.get("templateVersion") != reports.TEMPLATE_VERSION)


def report_info(db: sqlite3.Connection, meta: dict[str, Any]) -> dict[str, Any]:
    return {"id": meta["id"], "kind": meta["kind"], "targetId": meta["targetId"], "title": meta["title"],
            "createdAt": meta["createdAt"], "stale": _is_stale(db, meta)}


def _generate(db: sqlite3.Connection, store: AnalysisStore, kind: str, target_id: str) -> Optional[tuple[str, str]]:
    """(title, html) of a report, or None when the target does not exist."""

    if kind == "drawing":
        found = drawing_analysis(target_id, store)
        document = load_drawing(target_id)
        if not found or document is None:
            return None
        analysis = found[0]
        metadata = document.get("metadata") or {}
        player = player_info(db, metadata.get("owner", ""))
        session = session_row(db, metadata.get("sessionID", ""))
        meta = drawing_meta(db, target_id)
        title = f"{(meta or {}).get('name') or target_id}" + (f" ({player['name']})" if player and player["name"] else "")
        narrative = build_narrative(analysis, (player or {}).get("dominantHand"))
        return title, reports.drawing_report(document, analysis, narrative, title, player, session)

    if kind == "session":
        session = session_row(db, target_id)
        if session is None:
            return None
        metas = drawing_metas(db, session_id=target_id)
        entries, gallery = [], []
        for meta in metas:
            found = drawing_analysis(meta["id"], store)
            if found:
                entries.append((meta["playerID"], found[0]))
                gallery.append({"name": meta["name"], "playerId": meta["playerID"], "gameType": meta["gameType"],
                                "svg": drawing_thumbnail(meta["id"], store) or ""})
        players = player_infos(db, [m["playerID"] for m in metas])
        stats = session_stats(target_id, entries)
        return session["name"], reports.session_report(session, stats, players, gallery, ANALYZER_VERSION)

    if kind == "participant":
        history = participant_history(db, store, target_id)
        if history is None:
            return None
        rows = [{"name": r["meta"]["name"], "sessionName": r["sessionName"], "gameType": r["meta"]["gameType"],
                 "date": r["date"], "metrics": r["metrics"]} for r in history["drawings"]]
        title = f"{history['player']['name'] or 'Participant'}: history"
        return title, reports.participant_report(history["player"], rows, ANALYZER_VERSION)
    return None


def create_report(db: sqlite3.Connection, store: AnalysisStore, kind: str, target_id: str,
                  regenerate: bool = False) -> Optional[tuple[dict[str, Any], bool]]:
    """
    Generates the report, or reuses the stored one when nothing it depends on has changed (requirement REP-04).
    Returns (report info, reused), or None when the target does not exist.
    """

    fingerprint = target_fingerprint(db, kind, target_id)
    if fingerprint is None:
        return None
    existing = store.find_report(kind, target_id)
    if existing and not regenerate and not _is_stale(db, existing) and store.read_report_html(existing["id"]) is not None:
        return report_info(db, existing), True

    generated = _generate(db, store, kind, target_id)
    if generated is None:
        return None
    title, html = generated
    meta = {
        "id": existing["id"] if existing else new_report_id(),
        "kind": kind, "targetId": target_id, "title": title,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint, "analyzerVersion": ANALYZER_VERSION, "templateVersion": reports.TEMPLATE_VERSION,
    }
    store.save_report(meta, html)
    return report_info(db, meta), False
