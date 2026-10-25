"""
Fuzzy logic indicators (requirement AI-14): linguistic variables (low, medium, high) derived from the metrics by the
percentiles of a comparison group (fixed ranges as the fallback), and a human-readable, versioned rule base file
(rules/fuzzy_rules.toml) that psychologists can edit. Every answer names the indicator, its membership degrees, the
rules that fired and the input values, so each step can be checked.

Pure Python, no extra library: the web app's demo mode (src/api/mock/ai.ts) is a port of this file and reads a JSON copy
of the rule base. The indicators describe how a drawing compares with others on measurements. They are not a diagnosis.
"""

import re
import tomllib
from pathlib import Path
from typing import Any, Optional

from analysis import catalog
from analysis.common import number
from ai import settings
from ai.features import percentile

LEVELS = ("low", "medium", "high")
DEFAULT_PATH = Path(__file__).parent / "rules" / "fuzzy_rules.toml"
_TERM = re.compile(r"^(?P<metric>[A-Za-z][A-Za-z0-9]*)\s+is\s+(?P<not>not\s+)?(?P<level>low|medium|high)$", re.IGNORECASE)


class RuleBaseError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


# --- Reading the rule base ---

def rule_base_path() -> Path:
    configured = settings.number("AI_RULES_PATH", None)
    return Path(configured) if configured else DEFAULT_PATH


def parse_condition(text: str) -> list[list[tuple[str, str, bool]]]:
    """'a is high and b is not low or c is medium' -> [[(a, high, False), (b, low, True)], [(c, medium, False)]]."""

    branches = []
    for branch in re.split(r"\s+or\s+", text.strip(), flags=re.IGNORECASE):
        terms = []
        for term in re.split(r"\s+and\s+", branch.strip(), flags=re.IGNORECASE):
            match = _TERM.match(term.strip())
            if not match:
                raise ValueError(f'cannot read "{term.strip()}", expected "<metric> is [not] low|medium|high"')
            terms.append((match["metric"], match["level"].lower(), bool(match["not"])))
        branches.append(terms)
    return branches


def load_rule_base(path: Optional[Path] = None) -> dict[str, Any]:
    """The rule base, checked. Raises RuleBaseError with every problem found, so the file can be corrected in one go."""

    path = path or rule_base_path()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuleBaseError([f"The rule base file {path.name} was not found."])
    except tomllib.TOMLDecodeError as error:
        raise RuleBaseError([f"The rule base file is not valid TOML: {error}"])

    problems: list[str] = []
    for key in ("version", "name", "status"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            problems.append(f'"{key}" is missing.')
    if raw.get("status") not in (None, "proposal", "reviewed"):
        problems.append('"status" must be "proposal" or "reviewed".')

    fallback = raw.get("fallbackRanges") or {}
    for metric_id, values in fallback.items():
        ok = (isinstance(values, list) and len(values) == 3 and all(isinstance(v, (int, float)) for v in values)
              and values[0] < values[1] < values[2])
        if metric_id not in catalog.METRIC_IDS:
            problems.append(f'fallbackRanges: "{metric_id}" is not a metric.')
        elif not ok:
            problems.append(f"fallbackRanges.{metric_id} needs three increasing numbers.")

    indicators = raw.get("indicator") or []
    if not indicators:
        problems.append("There is no [[indicator]].")
    seen: set[str] = set()
    for indicator in indicators:
        indicator_id = indicator.get("id", "?")
        if not indicator.get("id") or indicator["id"] in seen:
            problems.append(f'indicator "{indicator_id}": the id is missing or used twice.')
        seen.add(indicator.get("id", ""))
        if not indicator.get("name") or not indicator.get("rationale"):
            problems.append(f'indicator "{indicator_id}": name and rationale are required.')
        levels = indicator.get("levels") or {}
        if any(not levels.get(level) for level in LEVELS):
            problems.append(f'indicator "{indicator_id}": levels.low, levels.medium and levels.high need a text.')
        rules = indicator.get("rule") or []
        if not rules:
            problems.append(f'indicator "{indicator_id}": it has no [[indicator.rule]].')
        for rule in rules:
            rule_id = rule.get("id", "?")
            where = f'rule "{rule_id}" of "{indicator_id}"'
            if not rule.get("rationale"):
                problems.append(f"{where}: every rule needs a rationale.")
            if rule.get("then") not in LEVELS:
                problems.append(f'{where}: "then" must be low, medium or high.')
            weight = rule.get("weight", 1.0)
            if not isinstance(weight, (int, float)) or not 0 < weight <= 1:
                problems.append(f'{where}: "weight" must be above 0 and at most 1.')
            try:
                branches = parse_condition(str(rule.get("if", "")))
            except ValueError as error:
                problems.append(f"{where}: {error}")
                continue
            for metric_id, _, _ in (term for branch in branches for term in branch):
                if metric_id not in catalog.METRIC_IDS:
                    problems.append(f'{where}: "{metric_id}" is not a metric.')
                elif metric_id not in fallback:
                    problems.append(f'{where}: "{metric_id}" needs an entry in fallbackRanges.')
    if problems:
        raise RuleBaseError(problems)

    return {
        "name": raw["name"], "version": raw["version"], "status": raw["status"], "reviewed": raw.get("reviewed", ""),
        "description": raw.get("description", ""), "fallbackRanges": {k: list(v) for k, v in fallback.items()},
        "indicators": [{
            "id": i["id"], "name": i["name"], "rationale": i["rationale"],
            "levels": {level: i["levels"][level] for level in LEVELS},
            "rules": [{"id": r["id"], "if": r["if"], "then": r["then"], "weight": float(r.get("weight", 1.0)),
                       "rationale": r["rationale"]} for r in i["rule"]],
        } for i in indicators],
    }


def rule_base_entry() -> dict[str, Any]:
    """The rule base as a line of the model list (requirement AI-06)."""

    entry: dict[str, Any] = {
        "id": "fuzzy-rules", "name": "Fuzzy rule base", "version": "-", "kind": "rules",
        "licence": "Own work of the project (editable)", "licenceChecked": True, "source": rule_base_path().name,
        "sizeHint": None, "usedBy": ["indicators"], "downloadable": False, "hint": None, "statusDetail": None}
    try:
        base = load_rule_base()
    except RuleBaseError as error:
        return {**entry, "status": "invalid", "statusDetail": "; ".join(error.problems[:3]),
                "hint": "Correct the rule base file, the indicators are off until it is valid."}
    detail = f"{base['name']}, status: {base['status']}" + (f", reviewed: {base['reviewed']}" if base["reviewed"] else "")
    return {**entry, "version": base["version"], "status": "installed", "statusDetail": detail}


# --- Evaluating it ---

def membership_degrees(x: float, a: float, b: float, c: float) -> dict[str, float]:
    """Low is 1 up to a and 0 from b, medium is a triangle a-b-c, high is 0 up to b and 1 from c."""

    low = 1.0 if x <= a else (b - x) / (b - a) if x < b else 0.0
    high = 1.0 if x >= c else (x - b) / (c - b) if x > b else 0.0
    medium = 0.0 if x <= a or x >= c else (x - a) / (b - a) if x < b else (c - x) / (c - b)
    return {"low": low, "medium": medium, "high": high}


def _breaks(metric_id: str, cohort_values: list[float], fallback: dict[str, list[float]],
            min_cohort: int) -> Optional[tuple[tuple[float, float, float], str]]:
    """The break points of a metric and where they come from: the cohort's percentiles, or the fixed range."""

    if len(cohort_values) >= min_cohort:
        ordered = sorted(cohort_values)
        a, b, c = (percentile(ordered, q) for q in (10, 50, 90))
        if a < b < c:
            return (a, b, c), "cohort"
    if metric_id in fallback:
        a, b, c = fallback[metric_id]
        return (a, b, c), "fixed"
    return None


def evaluate(rule_base: dict[str, Any], subject: dict[str, Optional[float]],
             cohort: list[dict[str, Optional[float]]], min_cohort: int) -> dict[str, Any]:
    """The indicators of one drawing against a comparison group (which must not contain the drawing itself)."""

    sets_cache: dict[str, Optional[dict[str, Any]]] = {}

    def sets_of(metric_id: str) -> Optional[dict[str, Any]]:
        if metric_id not in sets_cache:
            value = subject.get(metric_id)
            found = None
            if value is not None:
                breaks = _breaks(metric_id, [c[metric_id] for c in cohort if c.get(metric_id) is not None],
                                 rule_base["fallbackRanges"], min_cohort)
                if breaks:
                    degrees = membership_degrees(value, *breaks[0])
                    found = {"value": value, "basis": breaks[1], "degrees": degrees}
            sets_cache[metric_id] = found
        return sets_cache[metric_id]

    results = []
    for indicator in rule_base["indicators"]:
        degrees = {level: 0.0 for level in LEVELS}
        fired, skipped, used = [], [], []
        for rule in indicator["rules"]:
            best: Optional[float] = None
            missing: list[str] = []
            for branch in parse_condition(rule["if"]):
                terms = []
                for metric_id, level, negated in branch:
                    if metric_id not in used:
                        used.append(metric_id)
                    found = sets_of(metric_id)
                    if found is None:
                        missing.append(metric_id)
                        break
                    degree = found["degrees"][level]
                    terms.append(1 - degree if negated else degree)
                else:
                    strength = min(terms)
                    best = strength if best is None else max(best, strength)
            if best is None:
                skipped.append({"rule": rule["id"], "missing": sorted(set(missing))})
                continue
            contribution = best * rule["weight"]
            if contribution > 0:
                fired.append({"rule": rule["id"], "if": rule["if"], "then": rule["then"], "weight": rule["weight"],
                              "strength": number(best, 3), "rationale": rule["rationale"]})
                degrees[rule["then"]] = max(degrees[rule["then"]], contribution)
        top = max(degrees.values())
        level = next((lvl for lvl in LEVELS if degrees[lvl] == top), None) if top > 0 else None
        fired.sort(key=lambda f: (-f["strength"], f["rule"]))
        results.append({
            "id": indicator["id"], "name": indicator["name"], "rationale": indicator["rationale"],
            "level": level, "text": indicator["levels"][level] if level else None, "degree": number(top, 3),
            "degrees": {lvl: number(d, 3) for lvl, d in degrees.items()},
            "inputs": [{"metric": m, "value": (sets_of(m) or {}).get("value", subject.get(m)),
                        "basis": (sets_of(m) or {}).get("basis"),
                        "sets": {k: number(v, 3) for k, v in sets_of(m)["degrees"].items()} if sets_of(m) else None}
                       for m in used],
            "fired": fired, "skipped": skipped,
        })
    bases = {i["basis"] for r in results for i in r["inputs"] if i["basis"]}
    return {"cohortSize": len(cohort), "basis": "cohort" if bases == {"cohort"} else "fixed" if bases == {"fixed"} else "mixed",
            "indicators": results}
