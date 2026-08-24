"""Summary statistics for latency and monitor samples.

Percentiles use linear interpolation between order statistics (numpy's
default, same as ``numpy.percentile``) so small sample counts degrade
gracefully instead of snapping to the nearest observation.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

DEFAULT_PERCENTILES = (50.0, 90.0, 95.0, 99.0)


def percentile(values: Sequence[float], q: float) -> float:
    if not len(values):
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(
    values: Sequence[float],
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    unit: Optional[str] = None,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    summary: Dict[str, float] = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for q in percentiles:
        key = f"p{q:g}".replace(".", "_")
        summary[key] = float(np.percentile(arr, q))
    if unit:
        summary["unit"] = unit
    return summary
