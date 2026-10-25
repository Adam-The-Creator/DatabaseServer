"""
The metric vector of a drawing and the robust statistics on top of it, shared by the cohort analytics (AI-10) and the
fuzzy indicators (AI-14). Pure Python, no numpy: the numbers have to be reproducible by the web app's demo mode
(src/api/mock/ai.ts), which is a port of this file.
"""

import math
from typing import Optional

from analysis.common import median

# The metrics of the vector. Left out on purpose: the point count (depends on the sampling), the pause time (derived from
# the time span), and the metrics that older drawings never have (pressure, figure distance, menu and tool events).
VECTOR_METRICS = [
    "strokesDrawn", "totalLength", "meanStrokeLength", "boundingVolume", "heightRange", "activeTime", "meanSpeed",
    "directionalChange", "leftHandShare", "warmShare", "distinctColours", "eraseRatio",
]
# Quantities that are right-skewed (a few big drawings among many small ones) are compared on a log scale
LOG_METRICS = {"strokesDrawn", "totalLength", "meanStrokeLength", "boundingVolume", "activeTime", "meanSpeed"}

# The smallest spread that counts, in the units of the (transformed) metric. Without it a group of drawings that are all
# nearly alike would make a tiny difference look far away: 0.1 on a log scale is a difference of 10 %.
MIN_SCALE = {
    "strokesDrawn": 0.1, "totalLength": 0.1, "meanStrokeLength": 0.1, "boundingVolume": 0.1, "activeTime": 0.1,
    "meanSpeed": 0.1, "heightRange": 0.05, "directionalChange": 0.01, "leftHandShare": 5.0, "warmShare": 5.0,
    "eraseRatio": 5.0, "distinctColours": 0.5,
}

# Modified z-score of Iglewicz and Hoaglin: |z| above 3.5 is the usual limit for an outlier
Z_LIMIT = 3.5
Z_CLIP = 6.0
MIN_COVERAGE = 0.5  # a metric that fewer than this share of the drawings have is not used


def transform(metric_id: str, value: float) -> float:
    return math.log1p(max(value, 0.0)) if metric_id in LOG_METRICS else value


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between the closest ranks (numpy's default), q from 0 to 100."""

    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def robust_scale(values: list[float]) -> tuple[float, float, bool]:
    """
    (median, scale, tied). The scale is the MAD divided by 0.6745. When the MAD is 0 (more than half of the values are
    equal to the median) it is 1.253314 times the mean absolute deviation, as Iglewicz and Hoaglin recommend, and
    tied is True: such a scale is shaky, so a metric like that can describe a drawing but never make it "atypical".
    """

    centre = median(values)
    deviations = [abs(v - centre) for v in values]
    mad = median(deviations)
    if mad > 0:
        return centre, mad / 0.6745, False
    mean_deviation = sum(deviations) / len(deviations)
    return centre, mean_deviation / 1.253314 if mean_deviation > 0 else 0.0, True


def usable_metrics(rows: list[dict[str, Optional[float]]]) -> list[str]:
    """The vector metrics that enough drawings have a value for and that are not constant."""

    needed = max(2, math.ceil(len(rows) * MIN_COVERAGE))
    usable = []
    for metric_id in VECTOR_METRICS:
        values = [transform(metric_id, r[metric_id]) for r in rows if r.get(metric_id) is not None]
        if len(values) >= needed and robust_scale(values)[1] > 0:
            usable.append(metric_id)
    return usable


def z_matrix(rows: list[dict[str, Optional[float]]]) -> tuple[list[str], list[list[float]], list[list[bool]], list[bool]]:
    """
    Robust z-scores of every drawing on every usable metric: (metric ids, z[row][metric], imputed[row][metric],
    tied[metric]). A missing value gets z 0 (the median) and is marked as imputed, so it never makes a drawing look
    atypical. Values are clipped to +-Z_CLIP so that one extreme metric cannot dominate the distances. tied marks the
    metrics whose scale is shaky (see robust_scale).
    """

    metrics = usable_metrics(rows)
    columns = []
    for metric_id in metrics:
        present = [transform(metric_id, r[metric_id]) for r in rows if r.get(metric_id) is not None]
        columns.append(robust_scale(present))
    z: list[list[float]] = []
    imputed: list[list[bool]] = []
    for row in rows:
        z_row, missing_row = [], []
        for metric_id, (centre, scale, _) in zip(metrics, columns):
            value = row.get(metric_id)
            if value is None:
                z_row.append(0.0)
                missing_row.append(True)
            else:
                z_row.append(max(-Z_CLIP, min(Z_CLIP, (transform(metric_id, value) - centre) / max(scale, MIN_SCALE[metric_id]))))
                missing_row.append(False)
        z.append(z_row)
        imputed.append(missing_row)
    return metrics, z, imputed, [tied for _, _, tied in columns]
