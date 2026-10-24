"""
What the admin sees (requirements ADM-05..ADM-13): the usage numbers, the results of the questionnaires, the feedback
messages and their exports. Every sender is a pseudonym code (SC-09, ADM-12): the username, the real name and the
account ID never leave this module.

The statistics are computed in Python over the stored submissions (the data is small: a few dozen accounts), the way
the analysis module computes its numbers. A group with fewer submissions than config.FEEDBACK_MIN_GROUP_SIZE gets no
statistics at all, only its size (requirement ADM-10).
"""

import csv
import io
import json
import logging
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import config
from analysis.common import number
from feedback import instruments
from feedback.store import FeedbackError, iso
from security import Role

log = logging.getLogger("feedback")

CSV_DIALECTS = ("r", "excel")
# Two-sided 95 % limit of Student's t by degrees of freedom (1..30); 2.0 up to 60 and 1.96 beyond.
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
_INSTRUMENT_ORDER = {"SUS": 0, "UEQ": 1, "SMILEY": 2}


def _t975(degrees: int) -> float:
    return _T975[degrees - 1] if degrees <= 30 else 2.0 if degrees <= 60 else 1.96


# --- Filters ---

@dataclass
class Filters:
    target: Optional[str] = None
    role: Optional[int] = None
    age_band: Optional[str] = None
    version: Optional[int] = None
    date_from: Optional[str] = None  # YYYY-MM-DD, that day included
    date_to: Optional[str] = None  # YYYY-MM-DD, that day included
    counting: str = "latest"  # "latest" submission per account, or "all" submissions

    def validate(self) -> "Filters":
        if self.target is not None and self.target not in instruments.TARGETS:
            raise FeedbackError(422, "Unknown target.")
        if self.role is not None and self.role not in (Role.PLAYER, Role.PSYCHOLOGIST):
            raise FeedbackError(422, "Unknown role.")
        if self.age_band is not None and self.age_band not in instruments.AGE_BANDS:
            raise FeedbackError(422, "Unknown age band.")
        if self.counting not in ("latest", "all"):
            raise FeedbackError(422, "Unknown counting rule.")
        self.from_ms()
        self.to_ms()
        return self

    @staticmethod
    def _day_ms(day: str) -> int:
        try:
            parsed = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise FeedbackError(422, "A date must look like 2026-10-24.")
        return int(parsed.timestamp() * 1000)

    def from_ms(self) -> Optional[int]:
        return self._day_ms(self.date_from) if self.date_from else None

    def to_ms(self) -> Optional[int]:
        """Exclusive end: the start of the day after date_to."""
        return self._day_ms(self.date_to) + 24 * 3600 * 1000 if self.date_to else None

    def where(self, alias: str = "") -> tuple[str, list[Any]]:
        """The WHERE clause for QuestionnaireSubmissions (under the table alias, if the query has one)."""

        p = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("Target", self.target), ("Role", self.role), ("AgeBand", self.age_band),
                              ("InstrumentVersion", self.version)):
            if value is not None:
                clauses.append(f"{p}{column} = ?")
                params.append(value)
        if self.from_ms() is not None:
            clauses.append(f"{p}CreatedMs >= ?")
            params.append(self.from_ms())
        if self.to_ms() is not None:
            clauses.append(f"{p}CreatedMs < ?")
            params.append(self.to_ms())
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


# --- Usage numbers (ADM-05) ---

def _week_start(day: datetime) -> str:
    return (day - timedelta(days=day.weekday())).strftime("%Y-%m-%d")


def usage(db: sqlite3.Connection) -> dict[str, Any]:
    """
    Registered accounts per role, the accounts that used the game, new accounts per week and the number of submissions.
    "Used the game": a player with at least one drawing, a psychologist who signed in at least once. Sessions have no
    owner (SC-01) and a login of the web app sets the same sign-in time, so this is the best the data allows.
    """

    by_role = {row[0]: row[1] for row in db.execute("SELECT Role, COUNT(*) FROM Players GROUP BY Role")}
    used_players = db.execute(
        "SELECT COUNT(DISTINCT p.ID) FROM Players p JOIN DrawingMeta d ON d.PlayerID = p.ID WHERE p.Role = ?",
        (Role.PLAYER,),
    ).fetchone()[0]
    used_psychologists = db.execute(
        "SELECT COUNT(*) FROM Players WHERE Role = ? AND SignedIn IS NOT NULL", (Role.PSYCHOLOGIST,)
    ).fetchone()[0]

    weeks: dict[str, dict[str, int]] = {}
    for created, role in db.execute("SELECT Created, Role FROM Players"):
        try:
            day = datetime.strptime(created, "%Y.%m.%d-%H:%M")
        except (TypeError, ValueError):
            continue
        week = weeks.setdefault(_week_start(day), {"players": 0, "psychologists": 0, "total": 0})
        week["total"] += 1
        if role == Role.PLAYER:
            week["players"] += 1
        elif role == Role.PSYCHOLOGIST:
            week["psychologists"] += 1
    accounts_per_week = []
    if weeks:  # every week between the first and the last, so a quiet week shows as zero
        first = datetime.strptime(min(weeks), "%Y-%m-%d")
        last = datetime.strptime(max(weeks), "%Y-%m-%d")
        while first <= last:
            key = first.strftime("%Y-%m-%d")
            accounts_per_week.append({"weekStart": key, **weeks.get(key, {"players": 0, "psychologists": 0, "total": 0})})
            first += timedelta(days=7)

    return {
        "registered": {
            "total": sum(by_role.values()),
            "players": by_role.get(Role.PLAYER, 0),
            "psychologists": by_role.get(Role.PSYCHOLOGIST, 0),
            "admins": by_role.get(Role.ADMIN, 0),
        },
        "used": {"players": used_players, "psychologists": used_psychologists},
        "accountsPerWeek": accounts_per_week,
        "submissions": {
            "questionnaires": db.execute("SELECT COUNT(*) FROM QuestionnaireSubmissions").fetchone()[0],
            "messages": db.execute("SELECT COUNT(*) FROM FeedbackMessages").fetchone()[0],
            "unreadMessages": db.execute("SELECT COUNT(*) FROM FeedbackMessages WHERE IsRead = 0").fetchone()[0],
        },
    }


# --- Questionnaire results (ADM-06..ADM-10) ---

def _load(db: sqlite3.Connection, f: Filters) -> list[dict[str, Any]]:
    where, params = f.where()
    rows = db.execute(
        "SELECT PlayerID, Role, AgeBand, Target, Instrument, InstrumentVersion, Answers, Scores, CreatedMs "
        f"FROM QuestionnaireSubmissions{where} ORDER BY CreatedMs",
        params,
    ).fetchall()
    return [
        {"player": r[0], "role": r[1], "ageBand": r[2], "target": r[3], "instrument": r[4], "version": r[5],
         "answers": json.loads(r[6]), "scores": json.loads(r[7]), "created": r[8]}
        for r in rows
    ]


def _latest_per_account(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:  # sorted by time, so the last one wins
        latest[(row["player"], row["instrument"], row["target"])] = row
    return list(latest.values())


def _spread(values: list[float], digits: int = 2) -> dict[str, float]:
    return {
        "mean": number(statistics.fmean(values), digits),
        "sd": number(statistics.stdev(values) if len(values) > 1 else 0.0, digits),
        "median": number(statistics.median(values), digits),
        "min": number(min(values), digits),
        "max": number(max(values), digits),
    }


def _interval(values: list[float]) -> dict[str, float]:
    """The mean with its 95 % confidence interval."""

    n = len(values)
    mean = statistics.fmean(values)
    half = _t975(n - 1) * statistics.stdev(values) / n ** 0.5 if n > 1 else 0.0
    return {"mean": number(mean, 2), "low": number(mean - half, 2), "high": number(mean + half, 2)}


def _group_statistics(instrument_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if instrument_id == "SUS":
        scores = [r["scores"]["score"] for r in rows]
        bins = [0] * 10
        for score in scores:
            bins[min(int(score // 10), 9)] += 1  # 0-10 .. 90-100, 100 belongs to the last bin
        contributions = [instruments.sus_contributions(r["answers"]) for r in rows]
        return {
            "sus": {
                **_spread(scores),
                "histogram": [{"from": i * 10, "to": i * 10 + 10, "count": count} for i, count in enumerate(bins)],
                "items": [{"id": item_id, "mean": number(statistics.fmean(c[item_id] for c in contributions), 2)}
                          for item_id in instruments.SUS.item_ids],
            }
        }
    if instrument_id == "UEQ":
        values = [instruments.ueq_values(r["answers"]) for r in rows]
        return {
            "ueq": {
                "scales": [{"id": name, **_interval([r["scores"]["scales"][name] for r in rows])}
                           for name in instruments.UEQ_SCALES],
                "pragmatic": number(statistics.fmean(r["scores"]["pragmatic"] for r in rows), 2),
                "hedonic": number(statistics.fmean(r["scores"]["hedonic"] for r in rows), 2),
                "items": [{"id": item_id, "mean": number(statistics.fmean(v[item_id] for v in values), 2)}
                          for item_id in instruments.UEQ.item_ids],
            }
        }
    items = []
    for item_id in instruments.SMILEY.item_ids:
        answers = [r["answers"][item_id] for r in rows]
        items.append({"id": item_id, "mean": number(statistics.fmean(answers), 2),
                      "distribution": [answers.count(face) for face in range(1, 6)]})
    return {"smiley": {"mean": number(statistics.fmean(r["scores"]["mean"] for r in rows), 2), "items": items}}


def questionnaire_results(db: sqlite3.Connection, f: Filters) -> dict[str, Any]:
    f.validate()
    everything = _load(db, f)
    counted = _latest_per_account(everything) if f.counting == "latest" else everything
    minimum = config.FEEDBACK_MIN_GROUP_SIZE

    def group_key(row):
        return row["instrument"], row["target"], row["role"]

    def sort_key(key):
        return _INSTRUMENT_ORDER.get(key[0], 9), key[1], key[2]

    groups = []
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in counted:
        grouped.setdefault(group_key(row), []).append(row)
    for key in sorted(grouped, key=sort_key):
        rows = grouped[key]
        age_bands = {band: sum(1 for r in rows if r["ageBand"] == band) for band in instruments.AGE_BANDS}
        group = {
            "instrument": key[0], "target": key[1], "role": key[2], "n": len(rows),
            "accounts": len({r["player"] for r in rows}),
            "versions": sorted({r["version"] for r in rows}),
            "ageBands": {band: count for band, count in age_bands.items() if count},
            "enough": len(rows) >= minimum,
        }
        if group["enough"]:
            group.update(_group_statistics(key[0], rows))
        groups.append(group)

    # The trend follows all submissions in time (the latest-only rule would remove the earlier ones)
    weekly: dict[tuple, list[float]] = {}
    for row in everything:
        week = _week_start(datetime.fromtimestamp(row["created"] / 1000, tz=timezone.utc))
        weekly.setdefault(group_key(row) + (week,), []).append(instruments.headline(row["instrument"], row["scores"]))
    trend = [
        {"instrument": key[0], "target": key[1], "role": key[2], "week": key[3], "n": len(values),
         "mean": number(statistics.fmean(values), 2)}
        for key, values in sorted(weekly.items(), key=lambda kv: (sort_key(kv[0][:3]), kv[0][3]))
        if len(values) >= minimum
    ]
    return {
        "counting": f.counting, "minGroupSize": minimum, "submissions": len(everything), "counted": len(counted),
        "groups": groups, "trend": trend,
    }


# --- Messages (ADM-11, ADM-12) ---

def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _message_filter(*, target, category, role, age_band, date_from, date_to, read, q) -> tuple[str, list[Any]]:
    if target is not None and target not in instruments.TARGETS:
        raise FeedbackError(422, "Unknown target.")
    if category is not None and category not in instruments.CATEGORIES:
        raise FeedbackError(422, "Unknown category.")
    if role is not None and role not in (Role.PLAYER, Role.PSYCHOLOGIST):
        raise FeedbackError(422, "Unknown role.")
    if age_band is not None and age_band not in instruments.AGE_BANDS:
        raise FeedbackError(422, "Unknown age band.")
    if read not in (None, "read", "unread"):
        raise FeedbackError(422, "Unknown read filter.")
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (("m.Target", target), ("m.Category", category), ("m.Role", role), ("m.AgeBand", age_band)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    window = Filters(date_from=date_from, date_to=date_to)
    if window.from_ms() is not None:
        clauses.append("m.CreatedMs >= ?")
        params.append(window.from_ms())
    if window.to_ms() is not None:
        clauses.append("m.CreatedMs < ?")
        params.append(window.to_ms())
    if read:
        clauses.append("m.IsRead = ?")
        params.append(1 if read == "read" else 0)
    if q:
        clauses.append("m.Text LIKE ? ESCAPE '\\'")
        params.append(_like(q[:100]))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _message_dict(row) -> dict[str, Any]:
    return {"id": row[0], "code": row[1], "role": row[2], "ageBand": row[3], "target": row[4], "category": row[5],
            "text": row[6], "createdAt": iso(row[7]), "isRead": bool(row[8])}


_MESSAGE_COLUMNS = ("m.ID, COALESCE(f.Code, '?'), m.Role, m.AgeBand, m.Target, m.Category, m.Text, m.CreatedMs, m.IsRead "
                    "FROM FeedbackMessages m LEFT JOIN FeedbackProfiles f ON f.PlayerID = m.PlayerID")


def list_messages(db: sqlite3.Connection, *, limit: int, offset: int, **filters) -> dict[str, Any]:
    where, params = _message_filter(**filters)
    limit = max(1, min(limit, config.FEEDBACK_PAGE_SIZE_MAX))
    total = db.execute(f"SELECT COUNT(*) FROM FeedbackMessages m{where}", params).fetchone()[0]
    rows = db.execute(
        f"SELECT {_MESSAGE_COLUMNS}{where} ORDER BY m.CreatedMs DESC, m.ID LIMIT ? OFFSET ?",
        params + [limit, max(0, offset)],
    ).fetchall()
    return {
        "total": total,
        "unread": db.execute("SELECT COUNT(*) FROM FeedbackMessages WHERE IsRead = 0").fetchone()[0],
        "items": [_message_dict(r) for r in rows],
    }


def set_read(db: sqlite3.Connection, message_id: str, read: bool) -> dict[str, Any]:
    cursor = db.execute("UPDATE FeedbackMessages SET IsRead = ? WHERE ID = ?", (1 if read else 0, message_id))
    db.commit()
    if cursor.rowcount == 0:
        raise FeedbackError(404, "Message not found.")
    row = db.execute(f"SELECT {_MESSAGE_COLUMNS} WHERE m.ID = ?", (message_id,)).fetchone()
    return _message_dict(row)


def delete_message(db: sqlite3.Connection, message_id: str, admin_id: str) -> None:
    cursor = db.execute("DELETE FROM FeedbackMessages WHERE ID = ?", (message_id,))
    db.commit()
    if cursor.rowcount == 0:
        raise FeedbackError(404, "Message not found.")
    log.info("Admin %s deleted feedback message %s", admin_id, message_id)  # never the text (requirement SEC-11)


# --- Exports (ADM-13) ---

def _safe(text: str) -> str:
    """
    A cell that starts with = + - @ (or a tab or line break) is read as a formula by a spreadsheet, which can run
    commands or leak data (CSV injection). A leading apostrophe makes it plain text again.
    """

    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _cell(value: Any, dialect: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text.replace(".", ",") if dialect == "excel" else text
    if isinstance(value, int):
        return str(value)
    return _safe(str(value))  # only text can be a formula, a negative number is just a number


def _csv(header: list[str], rows: list[list[Any]], dialect: str) -> bytes:
    if dialect not in CSV_DIALECTS:
        raise FeedbackError(422, "Unknown CSV dialect.")
    out = io.StringIO()
    writer = csv.writer(out, delimiter=";" if dialect == "excel" else ",", lineterminator="\r\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_cell(value, dialect) for value in row])
    return out.getvalue().encode("utf-8-sig" if dialect == "excel" else "utf-8")


def _role_name(role: Optional[int]) -> str:
    return {Role.PLAYER: "player", Role.PSYCHOLOGIST: "psychologist"}.get(role, "")


def questionnaires_csv(db: sqlite3.Connection, instrument_id: str, f: Filters, dialect: str) -> bytes:
    """One row per submission of one instrument: pseudonym, group, scores and the raw answer of every item."""

    instrument = instruments.INSTRUMENTS.get(instrument_id)
    if instrument is None:
        raise FeedbackError(422, "Unknown questionnaire.")
    f.validate()
    where, params = f.where("q")
    where = (where + " AND" if where else " WHERE") + " q.Instrument = ?"
    rows = db.execute(
        "SELECT COALESCE(f.Code, '?'), q.Role, q.AgeBand, q.Target, q.InstrumentVersion, q.CreatedMs, q.Answers, q.Scores "
        "FROM QuestionnaireSubmissions q LEFT JOIN FeedbackProfiles f ON f.PlayerID = q.PlayerID"
        f"{where} ORDER BY q.CreatedMs",
        params + [instrument.id],
    ).fetchall()

    if instrument.id == "SUS":
        score_names = ["score"]
    elif instrument.id == "UEQ":
        score_names = list(instruments.UEQ_SCALES) + ["pragmatic", "hedonic"]
    else:
        score_names = ["mean"]

    def score_values(scores: dict[str, Any]) -> list[Any]:
        if instrument.id == "UEQ":
            return [scores["scales"][name] for name in instruments.UEQ_SCALES] + [scores["pragmatic"], scores["hedonic"]]
        return [scores[score_names[0]]]

    header = ["code", "role", "ageBand", "target", "version", "createdAt"] + score_names + list(instrument.item_ids)
    body = []
    for code, role, age_band, target, version, created, answers, scores in rows:
        answers, scores = json.loads(answers), json.loads(scores)
        body.append([code, _role_name(role), age_band, target, version, iso(created)]
                    + score_values(scores) + [answers[item_id] for item_id in instrument.item_ids])
    return _csv(header, body, dialect)


def messages_csv(db: sqlite3.Connection, dialect: str, **filters) -> bytes:
    where, params = _message_filter(**filters)
    rows = db.execute(f"SELECT {_MESSAGE_COLUMNS}{where} ORDER BY m.CreatedMs", params).fetchall()
    header = ["code", "role", "ageBand", "target", "category", "createdAt", "read", "text"]
    body = [[r[1], _role_name(r[2]), r[3], r[4], r[5], iso(r[7]), bool(r[8]), r[6]] for r in rows]
    return _csv(header, body, dialect)
