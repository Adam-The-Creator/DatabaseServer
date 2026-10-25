"""
Zero-shot semantic labelling (requirement AI-12): canonical views of the whole drawing and of the biggest stroke groups
are rendered and classified by a free CLIP-like image-text model against a configurable vocabulary
(data/vocabulary.toml). The result is the top labels with their share of the vocabulary's probability.

The share is NOT the probability that the label is right: the model has to spread 100 % over the vocabulary, so it is
high even for a poor fit when the other entries fit worse. That is why the vocabulary has "other" entries (scribble,
pattern, almost empty), and why a result with such an entry on top, or with a small margin, is marked as unspecific.

The model is OpenCLIP, which needs PyTorch and the downloaded weights. They are not part of requirements.txt (see
requirements-ai.txt): the function says "missing library" or "missing model" until they are installed (AI-03, AI-06).
The weights are never downloaded silently: the model is loaded from the checkpoint file in the local cache, which needs no
network at all, and the download is an explicit action (ai/registry.py).
"""

import math
import tomllib
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from analysis.common import Drawing, number
from ai import registry, settings
from ai.render import VIEWS, render_view

DEFAULT_VOCABULARY = Path(__file__).parent / "data" / "vocabulary.toml"
GROUP_VIEWS = ("front", "iso")  # the views of a stroke group (the whole drawing gets all four)
MAX_GROUPS = 6
TOP = 3
UNSPECIFIC_SCORE = 0.25  # a top label below this share, or with less margin than MIN_MARGIN, is marked as unspecific
MIN_MARGIN = 0.05


class ProviderUnavailable(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason  # "missingLibrary" or "missingModel"


class Provider(Protocol):
    def score(self, images: list[bytes], texts: list[str]) -> list[list[float]]:
        """Image-text similarity logits: one row per image (PNG bytes), one column per text."""


# --- The vocabulary ---

def vocabulary_path() -> Path:
    configured = settings.number("AI_VOCABULARY_PATH", None)
    return Path(configured) if configured else DEFAULT_VOCABULARY


def load_vocabulary(path: Optional[Path] = None) -> dict[str, Any]:
    path = path or vocabulary_path()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"The vocabulary file cannot be read: {error}")
    problems = []
    templates = raw.get("templates") or []
    if not templates or any("{}" not in t for t in templates):
        problems.append('"templates" needs at least one template with {} in it.')
    labels, seen = [], set()
    for label in raw.get("label") or []:
        if not label.get("id") or label["id"] in seen or not label.get("name") or not label.get("phrases"):
            problems.append(f'label "{label.get("id", "?")}": id, name and phrases are required, ids must be unique.')
            continue
        seen.add(label["id"])
        labels.append({"id": label["id"], "name": label["name"], "other": label.get("kind") == "other",
                       "phrases": [str(p) for p in label["phrases"]]})
    if len(labels) < 2:
        problems.append("The vocabulary needs at least two labels.")
    if not raw.get("version"):
        problems.append('"version" is missing.')
    if problems:
        raise ValueError(" ".join(problems))
    return {"name": raw.get("name", "Vocabulary"), "version": str(raw["version"]), "templates": templates, "labels": labels}


# --- The model ---

class ClipProvider:
    """OpenCLIP on the CPU (or the GPU when there is one). Loaded on first use and then kept."""

    def __init__(self, model: str, pretrained: str) -> None:
        self.model_name, self.pretrained = model, pretrained
        self._loaded: Optional[tuple[Any, Any, Any, Any]] = None

    def _load(self):
        if self._loaded is None:
            if not (settings.module_available("torch") and settings.module_available("open_clip")):
                raise ProviderUnavailable("missingLibrary", "PyTorch and open_clip are not installed.")
            # The checkpoint FILE is loaded, never a pretrained tag: a tag would download what is missing (requirement AI-06)
            arguments = registry.clip_load_arguments()
            if arguments is None:
                raise ProviderUnavailable("missingModel", "The CLIP weights are not downloaded.")
            import open_clip
            import torch

            try:
                model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, **arguments)
            except Exception as error:
                raise ProviderUnavailable("missingModel", f"The CLIP weights cannot be loaded: {error}") from error
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._loaded = (model.to(device).eval(), preprocess, open_clip.get_tokenizer(self.model_name), device)
        return self._loaded

    def score(self, images: list[bytes], texts: list[str]) -> list[list[float]]:
        model, preprocess, tokenizer, device = self._load()  # first: it says "missing library" if PyTorch is not there

        import io

        import torch
        from PIL import Image

        with torch.no_grad():
            batch = torch.stack([preprocess(Image.open(io.BytesIO(png)).convert("RGB")) for png in images]).to(device)
            image_features = model.encode_image(batch)
            text_features = model.encode_text(tokenizer(texts).to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return (100.0 * image_features @ text_features.T).cpu().tolist()


_provider: Optional[Provider] = None


def get_provider() -> Provider:
    global _provider
    if _provider is None:
        _provider = ClipProvider(str(settings.number("AI_CLIP_MODEL", "ViT-B-32")),
                                 str(settings.number("AI_CLIP_PRETRAINED", "laion2b_s34b_b79k")))
    return _provider


# --- Labelling ---

def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _label_scores(provider: Provider, vocabulary: dict[str, Any], images: list[bytes]) -> list[list[float]]:
    """Per image, the share of every label: the logits of a label's phrases (in every template) are averaged, then softmaxed."""

    texts, spans = [], []
    for label in vocabulary["labels"]:
        start = len(texts)
        texts += [template.format(phrase) for phrase in label["phrases"] for template in vocabulary["templates"]]
        spans.append((start, len(texts)))
    rows = provider.score(images, texts)
    return [_softmax([sum(row[a:b]) / (b - a) for a, b in spans]) for row in rows]


def _top(vocabulary: dict[str, Any], shares: list[float]) -> list[dict[str, Any]]:
    ranked = sorted(range(len(shares)), key=lambda i: (-shares[i], vocabulary["labels"][i]["id"]))[:TOP]
    return [{"label": vocabulary["labels"][i]["id"], "name": vocabulary["labels"][i]["name"],
             "other": vocabulary["labels"][i]["other"], "score": number(shares[i], 3)} for i in ranked]


def _verdict(top: list[dict[str, Any]]) -> dict[str, Any]:
    margin = top[0]["score"] - top[1]["score"] if len(top) > 1 else top[0]["score"]
    return {"top": top, "margin": number(margin, 3),
            "unspecific": top[0]["other"] or top[0]["score"] < UNSPECIFIC_SCORE or margin < MIN_MARGIN}


def label_drawing(doc: Drawing, groups: list[dict[str, Any]], provider: Provider, vocabulary: dict[str, Any],
                  update: Callable[[float, str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(result, evidence) of the labelling of the whole drawing and of its biggest groups."""

    update(0.05, "Rendering the views")
    shown = groups[:MAX_GROUPS]
    whole_images = [render_view(doc, view) for view in VIEWS]
    group_images = [[render_view(doc, view, set(g["strokeIds"])) for view in GROUP_VIEWS] for g in shown]

    update(0.3, "Classifying the whole drawing")
    whole_shares = _label_scores(provider, vocabulary, whole_images)
    mean = [sum(column) / len(column) for column in zip(*whole_shares)]
    result: dict[str, Any] = {
        "vocabulary": {"name": vocabulary["name"], "version": vocabulary["version"]},
        "whole": {**_verdict(_top(vocabulary, mean)),
                  "views": [{"view": view, "top": _top(vocabulary, shares)} for view, shares in zip(VIEWS, whole_shares)]},
        "groups": [],
    }
    evidence = [{"kind": "view", "view": view} for view in VIEWS]
    for number_, (group, images) in enumerate(zip(shown, group_images), 1):
        update(0.3 + 0.65 * number_ / max(1, len(shown)), f"Classifying group {group['id']}")
        shares = _label_scores(provider, vocabulary, images)
        result["groups"].append({
            "groupId": group["id"], "strokeIds": group["strokeIds"],
            **_verdict(_top(vocabulary, [sum(c) / len(c) for c in zip(*shares)]))})
        evidence += [{"kind": "view", "view": view, "groupId": group["id"]} for view in GROUP_VIEWS]
    return result, evidence
