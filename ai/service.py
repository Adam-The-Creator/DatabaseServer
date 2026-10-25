"""
Orchestration of the AI functions: reads the existing data through the analysis service (never writes to a database),
calls the pure functions, wraps every result in the AI envelope (ai/envelope.py), caches what is worth caching in the
file store, and starts the heavy functions as background jobs (AI-02).
"""

import sqlite3
from typing import Any, Optional

from analysis import service as data
from analysis.common import drawn_lines
from analysis.metrics import ANALYZER_VERSION, metric_values
from analysis.store import AnalysisStore
from ai import (
    VERSION, clustering, cohort_ml, envelope, fuzzy, insights, llm, ollama, registry, semantic, settings, vlm,
)
from ai.jobs import Job, manager
from ai.store import AiStore, key_hash

SCOPES = ("gameType", "session", "all")
FEATURE_MODEL = {"labels": "clip", "description": "vision", "summary": "text"}


class Context:
    """What one request needs: the databases and the stores, with the metrics of a drawing read only once."""

    def __init__(self, db: sqlite3.Connection, store: Optional[AnalysisStore] = None, ai_store: Optional[AiStore] = None) -> None:
        self.db = db
        self.store = store or AnalysisStore()
        self.ai_store = ai_store or AiStore()
        self._analyses: dict[str, Optional[tuple[dict[str, Any], str]]] = {}
        self._metas: dict[str, Optional[dict[str, Any]]] = {}

    def analysis(self, drawing_id: str) -> Optional[tuple[dict[str, Any], str]]:
        if drawing_id not in self._analyses:
            self._analyses[drawing_id] = data.drawing_analysis(drawing_id, self.store)
        return self._analyses[drawing_id]

    def metrics(self, drawing_id: str) -> Optional[dict[str, Optional[float]]]:
        found = self.analysis(drawing_id)
        return metric_values(found[0]) if found else None

    def meta(self, drawing_id: str) -> Optional[dict[str, Any]]:
        if drawing_id not in self._metas:
            self._metas[drawing_id] = data.drawing_meta(self.db, drawing_id)
        return self._metas[drawing_id]

    def cohort_ids(self, meta: dict[str, Any], scope: str, age_min: Optional[int], age_max: Optional[int]) -> list[str]:
        return data.cohort_drawing_ids(
            self.db, meta["id"], meta["sessionID"] if scope == "session" else None,
            meta["gameType"] if scope == "gameType" else None, age_min, age_max)

    def cohort_metrics(self, ids: list[str]) -> list[dict[str, Any]]:
        """[{"id", "metrics"}] of the drawings that have an analysis."""

        return [{"id": i, "metrics": m} for i in ids if (m := self.metrics(i)) is not None]

    def directory(self, ids: set[str]) -> dict[str, dict[str, Any]]:
        out = {}
        for drawing_id in sorted(ids):
            meta = self.meta(drawing_id)
            if meta:
                out[drawing_id] = {"name": meta["name"], "playerId": meta["playerID"], "gameType": meta["gameType"]}
        return out


# --- Models of the envelopes ---

def _cohort_model() -> dict[str, str]:
    return envelope.model_info("cohort-ml", "Robust z-scores, isolation forest, PCA, nearest neighbours",
                               f"scikit-learn {registry._version('scikit-learn') or '-'}", "BSD-3-Clause", "statistical")


def _groups_model() -> dict[str, str]:
    return envelope.model_info("groups-dbscan", "DBSCAN stroke clustering", f"scikit-learn {registry._version('scikit-learn') or '-'}",
                               "BSD-3-Clause", "statistical")


def _rules_model(base: Optional[dict[str, Any]] = None) -> dict[str, str]:
    return envelope.model_info("fuzzy-rules", base["name"] if base else "Fuzzy rule base", base["version"] if base else "-",
                               "Own work of the project", "rules")


def _registry_model(model_id: str, models: Optional[list[dict[str, Any]]] = None) -> dict[str, str]:
    entry = next((m for m in models or registry.models() if m["id"] == model_id), None) or {}
    return envelope.model_info(model_id, entry.get("name", model_id), entry.get("version", "-"), entry.get("licence", "unknown"),
                               entry.get("kind", "neural"))


# --- Availability ---

def unavailable(feature: str, models: Optional[list[dict[str, Any]]] = None) -> Optional[tuple[str, str]]:
    """(reason id, detail) when the function cannot run now, None when it can."""

    if not settings.feature_enabled(feature):
        return "switchedOff", ""
    if feature in ("cohort", "groups") and not (settings.module_available("numpy") and settings.module_available("sklearn")):
        return "missingLibrary", "numpy and scikit-learn"
    if feature == "indicators":
        try:
            fuzzy.load_rule_base()
        except fuzzy.RuleBaseError as error:
            return "invalidRules", "; ".join(error.problems[:3])
    if feature in FEATURE_MODEL:
        if feature != "summary" and not settings.module_available("numpy"):
            return "missingLibrary", "numpy"
        if feature == "labels":
            try:
                semantic.load_vocabulary()
            except ValueError as error:
                return "invalidVocabulary", str(error)[:200]
        entry = next((m for m in models or registry.models() if m["id"] == FEATURE_MODEL[feature]), None)
        status = entry["status"] if entry else "notInstalled"
        if status == "notInstalled":
            return "missingLibrary", entry["name"] if entry else ""
        if status == "unreachable":
            return "serviceUnreachable", ""
        if status == "notDownloaded":
            return "missingModel", entry["name"]
    return None


def without_result(feature: str, reason: str, detail: str = "") -> dict[str, Any]:
    """The envelope of a function that cannot run: switched off, or unavailable with the reason."""

    result = envelope.without_result(feature, "disabled" if reason == "switchedOff" else "unavailable", reason,
                                     envelope.model_info(feature, "-", "-", "-", "-"))
    if detail:
        result["notes"] = [detail]
    return result


def status() -> dict[str, Any]:
    models = registry.models()
    features = []
    for feature, info in settings.FEATURES.items():
        reason = unavailable(feature, models)
        features.append({
            "id": feature, "requirement": info["requirement"], "kind": info["kind"],
            "enabled": settings.feature_enabled(feature), "evaluated": settings.feature_evaluated(feature),
            "available": reason is None, "reason": reason[0] if reason else None, "detail": reason[1] if reason else None,
            "models": info["models"]})
    return {"enabled": settings.enabled(), "label": envelope.LABEL, "disclaimer": envelope.DISCLAIMER,
            "moduleVersion": VERSION, "features": features}


def _not_enough(feature: str, have: int, need: int, model: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """No result, and how many drawings there are against how many are needed (in params)."""

    return envelope.make(feature, model=model, params={**params, "have": have, "need": need}, status="notEnoughData",
                         reason="notEnoughData")


# --- AI-10: cohort analytics ---

def drawing_cohort(ctx: Context, drawing_id: str, scope: str = "gameType", age_min: Optional[int] = None,
                   age_max: Optional[int] = None) -> Optional[dict[str, Any]]:
    off = unavailable("cohort")
    if off:
        return without_result("cohort", *off)
    meta, subject = ctx.meta(drawing_id), ctx.metrics(drawing_id)
    if meta is None or subject is None:
        return None
    minimum = int(settings.number("AI_MIN_COHORT", 8))
    params = {"scope": scope, "ageMin": age_min, "ageMax": age_max, "minCohort": minimum}
    entries = [{"id": drawing_id, "metrics": subject}, *ctx.cohort_metrics(ctx.cohort_ids(meta, scope, age_min, age_max))]
    if len(entries) < minimum:
        return _not_enough("cohort", len(entries), minimum, _cohort_model(), params)
    analysis = cohort_ml.analyze_cohort(entries, drawing_id, int(settings.number("AI_MIN_ISOLATION_FOREST", 12)))
    row = analysis["drawings"][0]
    shown = {p["id"] for p in (analysis["map"] or {"points": []})["points"]} | {s["id"] for s in row["similar"]}
    return envelope.make(
        "cohort", model=_cohort_model(), params={**params, "isolationForest": row["isolation"] is not None},
        result={"cohortSize": analysis["cohortSize"], "metrics": analysis["metrics"], "zLimit": analysis["zLimit"],
                "subject": row, "map": analysis["map"], "directory": ctx.directory(shown)},
        evidence=[{"kind": "metric", "id": m, "value": subject[m]} for m in analysis["metrics"]],
        notes=["Atypical only means far from the others on these measurements. It does not mean better, worse or abnormal."])


def session_cohort(ctx: Context, session_id: str, game_type: Optional[int] = None) -> Optional[dict[str, Any]]:
    off = unavailable("cohort")
    if off:
        return without_result("cohort", *off)
    if data.session_row(ctx.db, session_id) is None:
        return None
    minimum = int(settings.number("AI_MIN_COHORT", 8))
    params = {"scope": "session", "gameType": game_type, "minCohort": minimum}
    ids = [m["id"] for m in data.drawing_metas(ctx.db, session_id=session_id) if game_type is None or m["gameType"] == game_type]
    entries = ctx.cohort_metrics(ids)
    if len(entries) < minimum:
        return _not_enough("cohort", len(entries), minimum, _cohort_model(), params)
    analysis = cohort_ml.analyze_cohort(entries, None, int(settings.number("AI_MIN_ISOLATION_FOREST", 12)))
    return envelope.make(
        "cohort", model=_cohort_model(), params={**params, "isolationForest": any(d["isolation"] for d in analysis["drawings"])},
        result={**analysis, "directory": ctx.directory({e["id"] for e in entries})},
        evidence=[{"kind": "drawing", "id": e["id"]} for e in entries],
        notes=["Atypical only means far from the others on these measurements. It does not mean better, worse or abnormal."])


# --- AI-11: stroke groups ---

def drawing_groups(ctx: Context, drawing_id: str, space_eps: float = 0.10, time_eps: float = 20.0) -> Optional[dict[str, Any]]:
    off = unavailable("groups")
    if off:
        return without_result("groups", *off)
    found = ctx.analysis(drawing_id)
    if found is None:
        return None
    params = {"spaceEps": space_eps, "timeEps": time_eps, "minPoints": clustering.MIN_POINTS, "spacing": clustering.SPACING}
    key = {"fingerprint": found[1], "moduleVersion": VERSION, **params}
    cached = ctx.ai_store.get("groups", drawing_id, key)
    if cached:
        return cached
    document = data.load_drawing(drawing_id)
    if document is None:
        return None
    if not drawn_lines(document):
        return envelope.make("groups", model=_groups_model(), params=params, status="notEnoughData", reason="noStrokes")
    result = clustering.cluster_strokes(document, space_eps, time_eps)
    made = envelope.make(
        "groups", model=_groups_model(), params=params, result=result,
        evidence=[{"kind": "drawing", "id": drawing_id}],
        notes=["The groups describe how the lines lie in space and time. The method does not know what they depict, "
               "so it has no confidence value."])
    ctx.ai_store.put("groups", drawing_id, key, made)
    return made


# --- AI-14: fuzzy indicators ---

def indicator_results(ctx: Context, meta: dict[str, Any], subject: dict[str, Optional[float]], scope: str = "gameType",
                      age_min: Optional[int] = None, age_max: Optional[int] = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(rule base, evaluation). Raises fuzzy.RuleBaseError."""

    base = fuzzy.load_rule_base()
    cohort = [e["metrics"] for e in ctx.cohort_metrics(ctx.cohort_ids(meta, scope, age_min, age_max))]
    return base, fuzzy.evaluate(base, subject, cohort, int(settings.number("AI_MIN_FUZZY_COHORT", 8)))


def drawing_indicators(ctx: Context, drawing_id: str, scope: str = "gameType", age_min: Optional[int] = None,
                       age_max: Optional[int] = None) -> Optional[dict[str, Any]]:
    off = unavailable("indicators")
    if off:
        return without_result("indicators", *off)
    meta, subject = ctx.meta(drawing_id), ctx.metrics(drawing_id)
    if meta is None or subject is None:
        return None
    base, evaluation = indicator_results(ctx, meta, subject, scope, age_min, age_max)
    notes = ["The indicators say how this drawing compares with the others on measurements. They are not a diagnosis."]
    if base["status"] != "reviewed":
        notes.append("The rule base is a proposal that psychologists have not reviewed yet.")
    if evaluation["basis"] != "cohort":
        notes.append("The comparison group is small, so fixed ranges are used for some or all measurements.")
    used = {i["metric"] for ind in evaluation["indicators"] for i in ind["inputs"]}
    return envelope.make(
        "indicators", model=_rules_model(base), prompt_version=f"{base['name']} v{base['version']} ({base['status']})",
        params={"scope": scope, "ageMin": age_min, "ageMax": age_max, "minCohort": int(settings.number("AI_MIN_FUZZY_COHORT", 8))},
        result={"ruleBase": {k: base[k] for k in ("name", "version", "status", "reviewed")}, **evaluation},
        evidence=[{"kind": "metric", "id": m, "value": subject.get(m)} for m in sorted(used)], notes=notes)


# --- AI-12, AI-13, AI-16: the heavy functions ---

def _age_band(age: Optional[int]) -> str:
    if not age or age <= 0:
        return "unknown"
    return "child (12 or younger)" if age <= 12 else "teenager (13 to 17)" if age <= 17 else "adult (18 or older)"


def _summary_facts(ctx: Context, drawing_id: str) -> Optional[dict[str, Any]]:
    meta, found = ctx.meta(drawing_id), ctx.analysis(drawing_id)
    if meta is None or found is None:
        return None
    analysis, fingerprint = found
    metrics = metric_values(analysis)
    try:
        base, evaluation = indicator_results(ctx, meta, metrics)
        indicators = evaluation["indicators"] if settings.feature_enabled("indicators") else []
    except fuzzy.RuleBaseError:
        indicators = []
    groups = None
    if settings.feature_enabled("groups") and not unavailable("groups"):
        made = drawing_groups(ctx, drawing_id)
        groups = made["result"] if made and made["status"] == "ok" else None

    def stored(feature: str) -> Optional[dict[str, Any]]:
        saved = ctx.ai_store.peek(feature, drawing_id)
        ok = saved and saved["key"].get("fingerprint") == fingerprint and saved["envelope"]["status"] == "ok"
        return saved["envelope"]["result"] if ok else None

    owner = data.player_info(ctx.db, meta["playerID"]) or {}
    return llm.build_facts(meta["gameType"], _age_band(owner.get("age")), metrics, analysis["flags"], indicators,
                           groups, stored("labels"), stored("description"))


def _heavy_key(ctx: Context, feature: str, drawing_id: str) -> Optional[tuple[dict[str, Any], Optional[dict[str, Any]]]]:
    """(key, facts): what a stored result of this function has to match to count as up to date."""

    found = ctx.analysis(drawing_id)
    if found is None:
        return None
    base = {"fingerprint": found[1], "analyzer": ANALYZER_VERSION, "moduleVersion": VERSION}
    if feature == "labels":
        vocabulary = semantic.load_vocabulary()
        return {**base, "model": f"{settings.number('AI_CLIP_MODEL', 'ViT-B-32')}/{settings.number('AI_CLIP_PRETRAINED', 'laion2b_s34b_b79k')}",
                "vocabulary": vocabulary["version"], "groups": [clustering.SPACING, clustering.MIN_POINTS]}, None
    if feature == "description":
        return {**base, "model": settings.number("AI_VISION_MODEL", ""), "prompt": vlm.PROMPT_VERSION}, None
    facts = _summary_facts(ctx, drawing_id)
    if facts is None:
        return None
    return {**base, "model": settings.number("AI_TEXT_MODEL", ""), "prompt": llm.PROMPT_VERSION, "inputHash": key_hash(facts)}, facts


def _describe_failure(reason: str, detail: str) -> str:
    return " ".join(part for part in (envelope.REASONS.get(reason, reason), detail) if part)


def _make_labels(document: dict[str, Any], drawing_id: str, job: Job) -> dict[str, Any]:
    groups = clustering.cluster_strokes(document)
    vocabulary = semantic.load_vocabulary()
    result, evidence = semantic.label_drawing(document, groups["groups"], semantic.get_provider(), vocabulary, job.update)
    top = result["whole"]["top"][0]
    return envelope.make(
        "labels", model=_registry_model("clip"), prompt_version=f"{vocabulary['name']} v{vocabulary['version']}",
        params={"clipModel": settings.number("AI_CLIP_MODEL", "ViT-B-32"), "pretrained": settings.number("AI_CLIP_PRETRAINED", ""),
                "views": list(semantic.VIEWS), "groupViews": list(semantic.GROUP_VIEWS), "groupsLabelled": len(result["groups"])},
        result=result, confidence=top["score"], confidence_kind="vocabularyShare", evidence=evidence,
        notes=["The share is how the model spreads 100 % over the vocabulary. It is not the probability that the label is right.",
               "Models trained on photographs often fail on abstract 3D scribbles."])


def _make_description(document: dict[str, Any], drawing_id: str, job: Job) -> dict[str, Any]:
    result, evidence = vlm.describe_drawing(document, str(settings.number("AI_VISION_MODEL", "")), job.update)
    return envelope.make(
        "description", model=_registry_model("vision"), prompt_version=vlm.PROMPT_VERSION,
        params={"temperature": 0, "seed": 0, "views": list(vlm.VIEWS), "timeoutSeconds": ollama.timeout_seconds()},
        result=result, evidence=evidence,
        notes=["The certainty values are stated by the model itself and are not calibrated.",
               "The model only saw images of the lines, not the scene the participant was in."])


def _make_summary(facts: dict[str, Any], job: Job) -> dict[str, Any]:
    result = llm.write_conclusion(facts, str(settings.number("AI_TEXT_MODEL", "")), job.update)
    return envelope.make(
        "summary", model=_registry_model("text"), prompt_version=llm.PROMPT_VERSION,
        params={"temperature": 0, "seed": 0, "timeoutSeconds": ollama.timeout_seconds()},
        result=result, evidence=[{"kind": "facts", "facts": facts}],
        notes=["A language model can state things that are not in the facts. Check every statement against the data.",
               "The template summary of the drawing (REP-07) remains the default text."])


def _runner(feature: str, drawing_id: str, key: dict[str, Any], facts: Optional[dict[str, Any]]):
    def run(job: Job) -> None:
        try:
            if feature == "summary":
                made = _make_summary(facts or {}, job)
            else:
                document = data.load_drawing(drawing_id)
                if document is None:
                    raise RuntimeError("The drawing was not found.")
                made = (_make_labels if feature == "labels" else _make_description)(document, drawing_id, job)
        except ollama.OllamaError as error:
            raise RuntimeError(_describe_failure(error.reason, error.detail)) from error
        except semantic.ProviderUnavailable as error:
            raise RuntimeError(_describe_failure(error.reason, str(error))) from error
        AiStore().put(feature, drawing_id, key, made)

    return run


def heavy_state(ctx: Context, feature: str, drawing_id: str) -> Optional[dict[str, Any]]:
    """The stored result of a heavy function, whether it is up to date, whether it can run, and the latest job."""

    if ctx.analysis(drawing_id) is None:
        return None
    off = unavailable(feature)
    stored = ctx.ai_store.peek(feature, drawing_id)
    stale = False
    if stored:
        try:
            current = _heavy_key(ctx, feature, drawing_id)
        except ValueError:
            current = None
        stale = current is None or stored.get("key") != current[0]
    latest = manager.latest(feature, drawing_id)
    return {"feature": feature, "result": stored["envelope"] if stored else None, "stale": stale,
            "available": off is None, "reason": off[0] if off else None, "detail": off[1] if off else None,
            "job": latest.to_dict() if latest else None}


def start_heavy(ctx: Context, feature: str, drawing_id: str, regenerate: bool = False) -> Optional[dict[str, Any]]:
    """Starts the function as a background job unless an up to date result exists. Returns the new state."""

    if ctx.analysis(drawing_id) is None:
        return None
    if unavailable(feature) is None:
        try:
            keyed = _heavy_key(ctx, feature, drawing_id)
        except ValueError:
            keyed = None
        if keyed is not None:
            key, facts = keyed
            if regenerate or ctx.ai_store.get(feature, drawing_id, key) is None:
                manager.submit(feature, drawing_id, _runner(feature, drawing_id, key, facts))
    return heavy_state(ctx, feature, drawing_id)


def start_download(model_id: str) -> Optional[Job]:
    entry = registry.model_by_id(model_id)
    if entry is None:
        return None
    job, _ = manager.submit("download", model_id, lambda job: registry.download(model_id, job.update))
    return job


# --- AI-15: session insights ---

def session_insights(ctx: Context, session_id: str) -> Optional[dict[str, Any]]:
    off = unavailable("insights")
    if off:
        return without_result("insights", *off)
    if data.session_row(ctx.db, session_id) is None:
        return None
    metas = data.drawing_metas(ctx.db, session_id=session_id)
    rule_error = None
    drawings = []
    for meta in metas:
        found = ctx.analysis(meta["id"])
        if found is None:
            continue
        indicators: list[dict[str, Any]] = []
        if settings.feature_enabled("indicators"):
            try:
                indicators = indicator_results(ctx, meta, metric_values(found[0]))[1]["indicators"]
            except fuzzy.RuleBaseError as error:
                rule_error = "; ".join(error.problems[:3])
        saved = ctx.ai_store.peek("labels", meta["id"])
        current = bool(saved) and saved["key"].get("fingerprint") == found[1] and saved["envelope"]["status"] == "ok"
        drawings.append({"id": meta["id"], "name": meta["name"], "playerId": meta["playerID"], "indicators": indicators,
                         "labels": saved["envelope"]["result"] if current else None, "atypical": None})
    params = {"sessionId": session_id}
    if len(drawings) < insights.MIN_DRAWINGS:
        return _not_enough("insights", len(drawings), insights.MIN_DRAWINGS, _cohort_model(), params)

    atypical_note = None
    if settings.feature_enabled("cohort") and not unavailable("cohort"):
        cohort = session_cohort(ctx, session_id)
        if cohort and cohort["status"] == "ok":
            flagged = {row["id"] for row in cohort["result"]["drawings"] if row["atypical"]}
            for drawing in drawings:
                drawing["atypical"] = drawing["id"] in flagged
        else:
            atypical_note = "notEnoughData"
    insight_list = insights.aggregate(drawings)
    notes = ["Counting only: a recurring label means that a model named it for several drawings, not that they show it."]
    if rule_error:
        notes.append(f"The indicators are missing: {rule_error}")
    return envelope.make(
        "insights", model=_cohort_model(), params=params,
        result={"drawingCount": len(drawings), "insights": insight_list, "atypicalSkipped": atypical_note,
                "coverage": {"labels": sum(1 for d in drawings if d["labels"]), "indicators": sum(1 for d in drawings if d["indicators"])},
                "directory": ctx.directory({d["id"] for d in drawings})},
        evidence=[{"kind": "drawing", "id": d["id"]} for d in drawings], notes=notes)


# --- Reports ---

def report_sections(db: sqlite3.Connection, kind: str, target_id: str) -> list[dict[str, Any]]:
    """
    The AI results a report shows: the cheap ones are computed, the heavy ones are the stored ones. A stored result that
    is out of date (the drawing or the findings it was written from changed since) is still part of the report, marked as
    out of date, so a report never silently lacks a section.
    """

    if not settings.enabled():
        return []
    ctx = Context(db)
    sections: list[Optional[dict[str, Any]]] = []
    if kind == "drawing":
        sections = [drawing_indicators(ctx, target_id), drawing_cohort(ctx, target_id), drawing_groups(ctx, target_id)]
        for feature in settings.HEAVY_FEATURES:
            state = heavy_state(ctx, feature, target_id)
            if state and state["result"] and state["result"]["status"] == "ok":
                sections.append({**state["result"], "stale": True} if state["stale"] else state["result"])
    elif kind == "session":
        sections = [session_insights(ctx, target_id)]
    return [s for s in sections if s and s["status"] in ("ok", "notEnoughData")]


def report_signature(db: sqlite3.Connection, kind: str, target_id: str) -> str:
    """A short hash of the AI state a report was made with, so that a report notices later AI results (REP-04)."""

    store = AiStore()
    ids = [target_id] if kind == "drawing" else [m["id"] for m in data.drawing_metas(db, session_id=target_id)] if kind == "session" else []
    return key_hash({"enabled": settings.enabled(), "features": {f: settings.feature_enabled(f) for f in settings.FEATURES},
                     "stamps": {i: store.generated_stamps(i) for i in ids}})
