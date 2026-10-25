"""
The models and libraries the AI functions use, with version, licence and status (requirement AI-06), and the explicit
download of a model. Nothing is downloaded silently: a model only comes onto this machine when a psychologist starts
its download on the settings page, after seeing its licence.

The licences are recorded from the publishers' model cards and are NOT yet verified against the use in the study
(licenceChecked is False for every entry): that check is still to be done by the project before the study.
"""

import time
from importlib import metadata
from typing import Any, Callable, Optional

from ai import fuzzy, ollama, settings

# Licences as the publishers state them. Anything not listed is recorded as unknown, to be checked before use.
KNOWN_LICENCES = {
    "qwen2.5:7b-instruct": "Apache-2.0", "qwen2.5vl:7b": "Apache-2.0", "moondream": "Apache-2.0",
}
SIZE_HINTS = {"qwen2.5:7b-instruct": "about 4.7 GB", "qwen2.5vl:7b": "about 6 GB", "clip": "about 0.6 GB (plus PyTorch)"}
OLLAMA_HINT = "Install Ollama from https://ollama.com and start it, then download the model here."


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _library(model_id: str, package: str, name: str, licence: str, used_by: list[str]) -> dict[str, Any]:
    version = _version(package)
    return {"id": model_id, "name": name, "version": version or "-", "kind": "library", "licence": licence,
            "licenceChecked": False, "source": "pypi.org", "sizeHint": None, "usedBy": used_by,
            "status": "installed" if version else "notInstalled", "statusDetail": None, "downloadable": False,
            "hint": None if version else f"Install it with: pip install {package}"}


def _clip() -> dict[str, Any]:
    model, tag = str(settings.number("AI_CLIP_MODEL", "ViT-B-32")), str(settings.number("AI_CLIP_PRETRAINED", "laion2b_s34b_b79k"))
    libraries = settings.module_available("torch") and settings.module_available("open_clip")
    status, detail, hint = "notInstalled", None, "Install PyTorch and open_clip into the server's venv: pip install -r requirements-ai.txt"
    if libraries:
        cached = clip_checkpoint() is not None
        status = "installed" if cached else "notDownloaded"
        hint = None if cached else "Download the weights with the button on this page."
    return {"id": "clip", "name": f"OpenCLIP {model}", "version": tag, "kind": "neural",
            "licence": "MIT (OpenCLIP code and the LAION-2B weights)", "licenceChecked": False,
            "source": "github.com/mlfoundations/open_clip", "sizeHint": SIZE_HINTS["clip"], "usedBy": ["labels"],
            "status": status, "statusDetail": detail, "downloadable": libraries and status == "notDownloaded", "hint": hint}


# The file names OpenCLIP looks for in a Hugging Face repository, safetensors first (see open_clip.pretrained)
CLIP_CHECKPOINT_FILES = ("open_clip_model.safetensors", "open_clip_pytorch_model.bin")


def _clip_config() -> dict[str, Any]:
    import open_clip

    return dict(open_clip.pretrained.get_pretrained_cfg(
        str(settings.number("AI_CLIP_MODEL", "ViT-B-32")), str(settings.number("AI_CLIP_PRETRAINED", "laion2b_s34b_b79k"))) or {})


def clip_checkpoint() -> Optional[str]:
    """
    The path of the CLIP checkpoint in the local cache, None when it has not been downloaded (or the libraries are
    missing). Loading the model from this path needs no network at all, which is how nothing is ever downloaded
    silently (requirement AI-06).
    """

    try:
        from huggingface_hub import try_to_load_from_cache

        repo = str(_clip_config().get("hf_hub", "")).rstrip("/")
        if not repo:
            return None
        for name in CLIP_CHECKPOINT_FILES:
            found = try_to_load_from_cache(repo, name)
            if isinstance(found, str):
                return found
    except Exception:
        return None
    return None


def clip_load_arguments() -> Optional[dict[str, Any]]:
    """
    The arguments for open_clip.create_model_and_transforms that load the cached checkpoint, with the model details the
    pretrained tag would have given (a path carries none of them). None when the checkpoint is not there.
    """

    checkpoint = clip_checkpoint()
    if checkpoint is None:
        return None
    config = _clip_config()
    arguments: dict[str, Any] = {"pretrained": checkpoint}
    if config.get("quick_gelu"):
        arguments["force_quick_gelu"] = True
    if config.get("mean") and config.get("std"):
        arguments["image_mean"], arguments["image_std"] = tuple(config["mean"]), tuple(config["std"])
    return arguments


def _ollama_model(model_id: str, config_name: str, default: str, kind: str, used_by: str,
                  installed: Optional[list[dict[str, Any]]], reachable: bool) -> dict[str, Any]:
    name = str(settings.number(config_name, default))
    if not reachable:
        status, hint = "unreachable", OLLAMA_HINT
    elif ollama.has_model(name, installed):
        status, hint = "installed", None
    else:
        status, hint = "notDownloaded", "Download the model with the button on this page."
    return {"id": model_id, "name": name, "version": "Ollama tag", "kind": kind,
            "licence": KNOWN_LICENCES.get(name, "unknown: check the model card before use"), "licenceChecked": False,
            "source": "ollama.com/library", "sizeHint": SIZE_HINTS.get(name), "usedBy": [used_by],
            "status": status, "statusDetail": None, "downloadable": reachable and status == "notDownloaded", "hint": hint}


_cache: Optional[tuple[float, float, list[dict[str, Any]]]] = None
CACHE_SECONDS = 3  # the list is asked for several times per request, and asking Ollama is a network call
CACHE_SECONDS_UNREACHABLE = 20  # no Ollama: do not ask again at every request (the settings page asks afresh)


def clear_cache() -> None:
    global _cache
    _cache = None


def models() -> list[dict[str, Any]]:
    """Every library, rule base and model the AI functions use."""

    global _cache
    if _cache and time.monotonic() - _cache[0] < _cache[1]:
        return _cache[2]
    listed = _models()
    reachable = next(m for m in listed if m["id"] == "ollama")["status"] == "installed"
    _cache = (time.monotonic(), CACHE_SECONDS if reachable else CACHE_SECONDS_UNREACHABLE, listed)
    return listed


def _models() -> list[dict[str, Any]]:
    try:
        installed, reachable = ollama.installed_models(), True
    except ollama.OllamaError:
        installed, reachable = None, False
    ollama_version = ollama.server_version() if reachable else None
    return [
        _library("numpy", "numpy", "NumPy", "BSD-3-Clause", ["cohort", "groups", "labels"]),
        _library("scikit-learn", "scikit-learn", "scikit-learn", "BSD-3-Clause", ["cohort", "groups"]),
        fuzzy.rule_base_entry(),
        _clip(),
        {"id": "ollama", "name": "Ollama", "version": ollama_version or "-", "kind": "runtime", "licence": "MIT",
         "licenceChecked": False, "source": "ollama.com", "sizeHint": None, "usedBy": ["description", "summary"],
         "status": "installed" if reachable else "unreachable", "statusDetail": None, "downloadable": False,
         "hint": None if reachable else OLLAMA_HINT},
        _ollama_model("vision", "AI_VISION_MODEL", "qwen2.5vl:7b", "vision-language", "description", installed, reachable),
        _ollama_model("text", "AI_TEXT_MODEL", "qwen2.5:7b-instruct", "language", "summary", installed, reachable),
    ]


def model_by_id(model_id: str) -> Optional[dict[str, Any]]:
    return next((m for m in models() if m["id"] == model_id), None)


def download(model_id: str, update: Callable[[float, str], None]) -> None:
    """Downloads a model that is on the list (an explicit action, requirement AI-06). Runs inside a background job."""

    entry = model_by_id(model_id)
    if entry is None or not entry["downloadable"]:
        raise ollama.OllamaError("failed", "This model cannot be downloaded now.")
    if model_id == "clip":
        # The one place where the CLIP weights are fetched from the internet. Loading the model never does that.
        import open_clip

        update(0.05, "Downloading the CLIP weights (about 0.6 GB)")
        open_clip.pretrained.download_pretrained(_clip_config())
        if clip_checkpoint() is None:
            raise ollama.OllamaError("failed", "The download did not leave a usable checkpoint in the cache.")
    else:
        ollama.pull(entry["name"], update)
    clear_cache()  # the new status shows at once
