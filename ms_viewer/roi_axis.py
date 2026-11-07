"""Convert ROI bounds between m/z and a reversible display axis."""

import numpy as np


class ROIAxisMapping:
    """Piecewise-linear mapping between strictly monotonic sampled axes."""

    def __init__(self, mz: np.ndarray, display: np.ndarray):
        if mz.ndim != 1 or mz.shape != display.shape or mz.size < 2:
            raise ValueError(
                "ROI axis mapping requires matching arrays with at least two points"
            )
        if not np.isfinite(mz).all() or not np.isfinite(display).all():
            raise ValueError("ROI axes must contain only finite values")

        self._mz_to_display = self._ordered_pair(mz, display)
        self._display_to_mz = self._ordered_pair(display, mz)

    @staticmethod
    def _ordered_pair(
        source: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.all(source[1:] > source[:-1]):
            return source, target
        if np.all(source[1:] < source[:-1]):
            return source[::-1], target[::-1]
        raise ValueError("ROI axes must be strictly monotonic")

    @staticmethod
    def _convert(
        bounds: tuple[float, float], pair: tuple[np.ndarray, np.ndarray]
    ) -> tuple[float, float]:
        """Interpolate bounds, clamping to the sampled range."""
        first, second = np.interp(bounds, *pair)
        return float(min(first, second)), float(max(first, second))

    def to_display(self, bounds: tuple[float, float]) -> tuple[float, float]:
        return self._convert(bounds, self._mz_to_display)

    def to_mz(self, bounds: tuple[float, float]) -> tuple[float, float]:
        return self._convert(bounds, self._display_to_mz)
