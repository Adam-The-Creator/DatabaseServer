"""
A conclusion written by a locally run language model (requirement AI-16, optional). It is given the facts the other
functions found about ONE drawing (metrics, indicators, stroke groups and, when they exist, the semantic labels and
the description) and writes a short, careful summary for the psychologist. It is clearly marked as AI-generated, and the
deterministic summary (REP-07) stays the default text.

The model gets no name of the participant, only the age band and the game type. Its answer is checked: a wrong format
or any clinical wording discards it.
"""

import json
from typing import Any, Callable

from analysis import catalog
from ai import answers, ollama

PROMPT_VERSION = "llm-2"  # 1: first version; 2: no guessing of feelings or meanings, neutral questions (seen with a real model)

SYSTEM_PROMPT = (
    "You support a psychologist who looks at three-dimensional drawings made in a virtual reality art therapy game. "
    "You get measurements and findings about ONE drawing as JSON. Write a short, careful summary for the "
    "psychologist. Rules: use only the facts that are given and never invent objects, events or feelings; repeat "
    "numbers exactly as they are given and never calculate new ones; say what the measurements show and never what "
    "they mean, so do not guess feelings, thoughts, motives, preferences or hidden meanings; make no diagnosis and use "
    "no medical or psychological labels; do not judge the participant; say where data is missing or weak; the "
    "psychologist decides what it means. 'overview' is two or three sentences, 'observations' are plain statements of "
    "what the facts show, 'thingsToExplore' are neutral, open questions about the drawing or about how it was made that "
    "the psychologist could ask the participant to answer in their own words (for example what they would say the "
    "drawing shows, or how they decided where to start), never questions about feelings or hidden meanings, "
    "'limitations' say what the facts cannot tell. Answer only with JSON that matches the given schema."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "observations": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        "thingsToExplore": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "limitations": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
    },
    "required": ["overview", "observations", "thingsToExplore", "limitations"],
}


def build_facts(
        game_type: int, age_band: str, metrics: dict[str, Any], flags: list[str], indicators: list[dict[str, Any]],
        groups: dict[str, Any] | None, labels: dict[str, Any] | None, description: dict[str, Any] | None,
) -> dict[str, Any]:
    """The facts the model is given, as plain data. They are also what the stored summary is keyed on."""

    facts: dict[str, Any] = {
        "gameType": catalog.game_type_name(game_type),
        "participantAgeBand": age_band,
        "measurements": [
            {"name": catalog.metric(m)["name"], "value": catalog.format_metric(m, v)}
            for m, v in metrics.items() if v is not None],
        "dataQualityNotes": [catalog.FLAGS[f] for f in flags],
        "indicators": [
            {"name": i["name"], "comparedWithOthers": i["text"], "degree": i["degree"]}
            for i in indicators if i["level"]],
    }
    if groups:
        facts["strokeGroups"] = {"count": len(groups["groups"]), "ungroupedStrokes": len(groups["unclustered"]),
                                 "sizesAsShareOfInk": [g["inkShare"] for g in groups["groups"][:6]]}
    if labels and labels.get("whole"):
        facts["semanticLabels"] = {
            "wholeDrawing": [{"label": t["name"], "share": t["score"]} for t in labels["whole"]["top"]],
            "unspecific": labels["whole"]["unspecific"]}
    if description:
        facts["visionModelDescription"] = {k: description[k] for k in ("depictedObjects", "scene", "abstract")}
    return facts


def validate(data: dict[str, Any]) -> dict[str, Any]:
    result = {
        "overview": answers.text(data, "overview", 700),
        "observations": answers.strings(data, "observations", 6, 300),
        "thingsToExplore": answers.strings(data, "thingsToExplore", 4, 300),
        "limitations": answers.strings(data, "limitations", 4, 300),
    }
    answers.reject_diagnostic_language(result)
    return result


def write_conclusion(facts: dict[str, Any], model: str, update: Callable[[float, str], None]) -> dict[str, Any]:
    update(0.2, "Waiting for the language model")
    content = ollama.chat(
        model,
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Facts:\n" + json.dumps(facts, ensure_ascii=False)}],
        schema=SCHEMA)
    update(0.95, "Checking the answer")
    return validate(answers.parse_object(content))
