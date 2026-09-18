"""Split-conformal prediction intervals.

Distribution-free: hold out a calibration slice, take the empirical
quantile of absolute residuals, and emit a symmetric interval. Coverage is
guaranteed under exchangeability, which is why empirical coverage is
reported rather than assumed.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(residuals, confidence: float = 0.90) -> float:
    """The conformal width for a given confidence level.

    Uses the finite-sample corrected rank ceil((n+1)*confidence)/n, which is
    what gives the coverage guarantee rather than the plain quantile.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    residuals = np.abs(np.asarray(residuals, dtype=float))
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * confidence) / n)
    return float(np.quantile(residuals, level, method="higher"))


def make_interval(predictions, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric interval of half-width ``q`` around each prediction."""
    predictions = np.asarray(predictions, dtype=float)
    return predictions - q, predictions + q
