"""
Peak area Plugin.

Calculates the absolute area under the peak.
Background subtraction is performed
using the linear fit of the amplitudes at the edges of the ROI.

"""

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ms_viewer.roi_statistics import CustomStatisticPlugin
else:
    try:
        from ms_viewer.roi_statistics import CustomStatisticPlugin
    except ImportError:
        from abc import ABC, abstractmethod

        class CustomStatisticPlugin(ABC):
            @abstractmethod
            def calculate(
                self,
                times: np.ndarray,
                amplitudes: np.ndarray,
                roi_start: float,
                roi_end: float,
            ) -> Dict[str, float]: ...
            @abstractmethod
            def get_metrics(self) -> List[str]: ...


class PeakArea(CustomStatisticPlugin):
    """
    Calculate the area under the peak within the ROI.
    """

    def calculate(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        roi_start: float,
        roi_end: float,
        background_edge_percent: float = 5.0,
    ) -> Dict[str, float]:
        """
        Calculate the peak area.

        Args:
            times: Time values within the ROI
            amplitudes: Amplitude values within the ROI
            roi_start: Start time of ROI
            roi_end: End time of ROI
            background_edge_percent: Percentage of ROI edges to use for background estimation
        Returns:
            Dictionary with peak area metric
        """
        edge_percent = background_edge_percent / 100.0
        n_points = len(times)
        edge_points = int(n_points * edge_percent)

        # Calculate background as linear fit of amplitudes at the edges
        left_edge_times = times[:edge_points]
        left_edge_amps = amplitudes[:edge_points]
        right_edge_times = times[-edge_points:]
        right_edge_amps = amplitudes[-edge_points:]
        edge_times = np.concatenate((left_edge_times, right_edge_times))
        edge_amps = np.concatenate((left_edge_amps, right_edge_amps))
        coeffs = np.polyfit(edge_times, edge_amps, 1)
        background = np.polyval(coeffs, times)

        # Subtract background from amplitudes
        corrected_amplitudes = amplitudes - background

        # Calculate area under the peak using trapezoidal integration
        area = np.trapezoid(corrected_amplitudes, times)

        return {"peak_area": float(area)}

    def get_metrics(self) -> List[str]:
        """
        Get the list of metrics provided by this plugin.

        Returns:
            List of metric names
        """
        return ["peak_area"]
