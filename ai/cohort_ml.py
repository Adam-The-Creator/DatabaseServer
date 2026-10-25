"""
Cohort analytics with classical machine learning (requirement AI-10, no pretrained model): which drawings are atypical
for their comparison group, which drawings are similar, and a 2D map of the group.

The robust z-scores, the nearest neighbours and the explanations are plain Python (features.py) and are mirrored by the
web app's demo mode. The isolation forest (a second opinion) and the PCA map use scikit-learn. "Atypical" only means
"far from the others on these measurements", never "worse" or "abnormal" in any clinical sense (SC-07).
"""

import math
from typing import Any, Optional

from analysis.common import number
from ai.features import Z_LIMIT, z_matrix

NOTABLE_Z = 2.0  # metrics further than this from the group's median are named as notable
NEIGHBOURS = 3


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _normalise_sign(component: Any) -> Any:
    """The sign of a principal component is arbitrary; make the loading with the largest magnitude positive."""

    import numpy as np

    return component if component[int(np.argmax(np.abs(component)))] >= 0 else -component


def _map(z: list[list[float]], ids: list[str], metrics: list[str]) -> Optional[dict[str, Any]]:
    if len(ids) < 4 or len(metrics) < 2:
        return None
    import numpy as np
    from sklearn.decomposition import PCA

    matrix = np.array(z, dtype=float)
    pca = PCA(n_components=2, svd_solver="full").fit(matrix)
    components = np.array([_normalise_sign(c) for c in pca.components_])
    scores = (matrix - pca.mean_) @ components.T
    axes = []
    for axis in range(2):
        order = np.argsort(-np.abs(components[axis]))[:3]
        axes.append({"explained": number(float(pca.explained_variance_ratio_[axis]), 3),
                     "top": [{"metric": metrics[i], "loading": number(float(components[axis][i]), 3)} for i in order]})
    return {"points": [{"id": i, "x": number(float(s[0]), 3), "y": number(float(s[1]), 3)} for i, s in zip(ids, scores)],
            "axes": axes}


def _isolation(z: list[list[float]], minimum: int) -> Optional[list[float]]:
    if len(z) < minimum:
        return None
    import numpy as np
    from sklearn.ensemble import IsolationForest

    forest = IsolationForest(n_estimators=200, random_state=0, contamination="auto").fit(np.array(z, dtype=float))
    return [float(-s) for s in forest.score_samples(np.array(z, dtype=float))]


def analyze_cohort(entries: list[dict[str, Any]], subject_id: Optional[str] = None,
                   min_isolation_forest: int = 12) -> dict[str, Any]:
    """
    entries: [{"id", "metrics": {metric id: value or None}}] of the whole group. With a subject_id only that drawing's
    row and neighbours are returned (the map always shows the whole group).
    """

    entries = sorted(entries, key=lambda e: e["id"])
    ids = [e["id"] for e in entries]
    metrics, z, imputed, tied = z_matrix([e["metrics"] for e in entries])
    isolation = _isolation(z, min_isolation_forest) if metrics else None
    # 1 = the drawing the forest finds easiest to isolate (the most atypical one)
    rank_of = {ids[k]: rank for rank, k in enumerate(sorted(range(len(ids)), key=lambda k: (-isolation[k], ids[k])), 1)} \
        if isolation else {}

    rows = []
    for index, entry in enumerate(entries):
        if subject_id is not None and entry["id"] != subject_id:
            continue
        found = []
        for metric_id, value, was_imputed, is_tied in zip(metrics, z[index], imputed[index], tied):
            if not was_imputed and abs(value) >= NOTABLE_Z:
                found.append({"metric": metric_id, "z": number(value, 2), "value": entry["metrics"][metric_id],
                              "direction": "high" if value > 0 else "low",
                              "beyondLimit": abs(value) >= Z_LIMIT and not is_tied})
        found.sort(key=lambda f: (-abs(f["z"]), f["metric"]))
        peers = sorted(((_distance(z[index], z[other]), ids[other]) for other in range(len(ids)) if other != index))
        rows.append({
            "id": entry["id"],
            "maxAbsZ": number(max((abs(f["z"]) for f in found), default=0.0), 2),
            "atypical": any(f["beyondLimit"] for f in found),
            "notable": found[:3],
            "isolation": None if isolation is None else {
                "score": number(isolation[index], 3), "rank": rank_of[entry["id"]], "of": len(ids)},
            "similar": [{"id": peer, "distance": number(d, 3), "similarity": number(1 / (1 + d), 3)}
                        for d, peer in peers[:NEIGHBOURS]],
        })
    return {"cohortSize": len(entries), "metrics": metrics, "zLimit": Z_LIMIT, "drawings": rows,
            "map": _map(z, ids, metrics)}
