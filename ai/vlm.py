"""
Description of a drawing by a locally served open-weight vision-language model (requirement AI-13): the rendered views
and a fixed, versioned prompt go in, a structured JSON description comes out, validated against a schema, with a
timeout. The model is asked what can be SEEN. It is told not to interpret the person, and any answer that uses clinical
wording is discarded.
"""

import base64
from typing import Any, Callable

from analysis.common import Drawing
from ai import answers, ollama
from ai.render import render_view

PROMPT_VERSION = "vlm-2"  # 1: first version; 2: "scene" has to be a sentence (a real model answered with the view's name)
VIEWS = ("iso", "front", "side")
MIN_SCENE_CHARS = 20

SYSTEM_PROMPT = (
    "You look at three images of ONE three-dimensional line drawing made in a virtual reality drawing game. The three "
    "images are the same drawing seen from three sides, they are not three drawings. Describe only what can be seen. "
    "Use plain, short words. 'scene' is one or two complete sentences that say what the drawing as a whole looks like; "
    "never answer with a single word and never name the side it is seen from. 'depictedObjects' lists the objects you "
    "can recognise, and is empty if you cannot recognise any. 'abstract' is true when the drawing is only lines and "
    "shapes without a recognisable object. Do not interpret feelings, motives or behaviour, and do not say anything "
    "about the person who drew it. Do not make a diagnosis and do not use medical or psychological terms. Answer only "
    "with JSON that matches the given schema."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "depictedObjects": {"type": "array", "maxItems": 8, "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "certainty": {"type": "string", "enum": ["low", "medium", "high"]}},
            "required": ["name", "certainty"]}},
        "scene": {"type": "string", "minLength": MIN_SCENE_CHARS,
                  "description": "One or two complete sentences about what the whole drawing looks like."},
        "notableFeatures": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        "colourMoodWords": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        "abstract": {"type": "boolean"},
    },
    "required": ["depictedObjects", "scene", "notableFeatures", "colourMoodWords", "abstract"],
}


def validate(data: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for item in data.get("depictedObjects") if isinstance(data.get("depictedObjects"), list) else []:
        if isinstance(item, dict) and answers.text(item, "name", 60, required=False):
            certainty = item.get("certainty") if item.get("certainty") in ("low", "medium", "high") else "low"
            objects.append({"name": answers.text(item, "name", 60), "certainty": certainty})
    if not isinstance(data.get("abstract"), bool):
        raise ollama.OllamaError("invalidAnswer", 'The field "abstract" is missing.')
    scene = answers.text(data, "scene", 400)
    if len(scene) < MIN_SCENE_CHARS:
        raise ollama.OllamaError("invalidAnswer", "The scene is not a sentence.")
    result = {
        "depictedObjects": objects[:8],
        "scene": scene,
        "notableFeatures": answers.strings(data, "notableFeatures", 6),
        "colourMoodWords": answers.strings(data, "colourMoodWords", 6, limit=30),
        "abstract": data["abstract"],
    }
    answers.reject_diagnostic_language(result)
    return result


def describe_drawing(doc: Drawing, model: str, update: Callable[[float, str], None]) -> tuple[dict[str, Any], list[dict]]:
    """(description, evidence). Raises OllamaError when the service or the model fails, or the answer is not acceptable."""

    update(0.1, "Rendering the views")
    images = [base64.b64encode(render_view(doc, view)).decode("ascii") for view in VIEWS]
    update(0.3, "Waiting for the vision model")
    content = ollama.chat(
        model,
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Describe this drawing.", "images": images}],
        schema=SCHEMA)
    update(0.95, "Checking the answer")
    return validate(answers.parse_object(content)), [{"kind": "view", "view": view} for view in VIEWS]
