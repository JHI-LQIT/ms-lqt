"""
Pulse Analysis Plugin

Analyzes pulse characteristics such as rise time, fall time,
pulse width, and overshoot.

Useful for digital signal analysis and pulse waveform characterization.
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


class PulseAnalysis(CustomStatisticPlugin):
    """
    Analyze pulse characteristics in waveform data.

    Measures rise time, fall time, pulse width, and overshoot.
    """

    def calculate(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        roi_start: float,
        roi_end: float,
    ) -> Dict[str, float]:
        """
        Calculate pulse metrics.

        Args:
            times: Time values within the ROI
            amplitudes: Amplitude values within the ROI
            roi_start: Start time of ROI
            roi_end: End time of ROI

        Returns:
            Dictionary with pulse analysis metrics
        """
        if len(amplitudes) < 3:
            return {
                "rise_time": 0.0,
                "fall_time": 0.0,
                "pulse_width": 0.0,
                "overshoot_percent": 0.0,
            }

        min_val = np.min(amplitudes)
        max_val = np.max(amplitudes)
        amplitude_range = max_val - min_val

        if amplitude_range == 0:
            return {
                "rise_time": 0.0,
                "fall_time": 0.0,
                "pulse_width": 0.0,
                "overshoot_percent": 0.0,
            }

        # Calculate threshold levels
        level_10 = min_val + 0.1 * amplitude_range
        level_50 = min_val + 0.5 * amplitude_range
        level_90 = min_val + 0.9 * amplitude_range

        # Find rising edge (10% to 90%)
        rise_start_idx = self._find_first_crossing(amplitudes, level_10, rising=True)
        rise_end_idx = self._find_first_crossing(amplitudes, level_90, rising=True)

        if (
            rise_start_idx is not None
            and rise_end_idx is not None
            and rise_end_idx > rise_start_idx
        ):
            rise_time = times[rise_end_idx] - times[rise_start_idx]
        else:
            rise_time = 0.0

        # Find falling edge (90% to 10%)
        fall_start_idx = self._find_first_crossing(amplitudes, level_90, rising=False)
        fall_end_idx = self._find_first_crossing(amplitudes, level_10, rising=False)

        if (
            fall_start_idx is not None
            and fall_end_idx is not None
            and fall_end_idx > fall_start_idx
        ):
            fall_time = times[fall_end_idx] - times[fall_start_idx]
        else:
            fall_time = 0.0

        # Calculate pulse width at 50% level
        pulse_width = self._calculate_pulse_width(amplitudes, times, level_50)

        # Calculate overshoot
        # Overshoot is the amount signal exceeds final steady-state value
        # Estimate steady state as mean of last 10% of data
        steady_state_samples = max(1, len(amplitudes) // 10)
        steady_state = np.mean(amplitudes[-steady_state_samples:])

        if steady_state != min_val:
            overshoot = max_val - steady_state
            overshoot_percent = (overshoot / abs(steady_state - min_val)) * 100.0
        else:
            overshoot_percent = 0.0

        return {
            "rise_time": float(rise_time),
            "fall_time": float(fall_time),
            "pulse_width": float(pulse_width),
            "overshoot_percent": float(overshoot_percent),
        }

    @staticmethod
    def _find_first_crossing(
        data: np.ndarray, threshold: float, rising: bool = True
    ) -> Optional[int]:
        """
        Find first index where data crosses threshold.

        Args:
            data: Array of values
            threshold: Threshold value
            rising: True for rising edge, False for falling edge

        Returns:
            Index of first crossing or None if not found
        """
        if rising:
            crossings = np.where(data >= threshold)[0]
        else:
            crossings = np.where(data <= threshold)[0]

        return int(crossings[0]) if len(crossings) > 0 else None

    @staticmethod
    def _calculate_pulse_width(
        amplitudes: np.ndarray, times: np.ndarray, threshold: float
    ) -> float:
        """
        Calculate pulse width at given threshold level.

        Args:
            amplitudes: Amplitude values
            times: Time values
            threshold: Threshold level

        Returns:
            Pulse width in seconds
        """
        # Find all regions above threshold
        above_threshold = amplitudes >= threshold

        # Find rising and falling edges
        edges = np.diff(above_threshold.astype(int))
        rising_edges = np.where(edges == 1)[0]
        falling_edges = np.where(edges == -1)[0]

        if len(rising_edges) > 0 and len(falling_edges) > 0:
            # Use first complete pulse
            if rising_edges[0] < falling_edges[0]:
                pulse_width = times[falling_edges[0]] - times[rising_edges[0]]
                return float(pulse_width)

        return 0.0

    def get_metrics(self) -> List[str]:
        """Return list of metrics this plugin provides."""
        return ["rise_time", "fall_time", "pulse_width", "overshoot_percent"]

    def get_info(self) -> dict:
        """Return plugin metadata."""
        return {
            "name": "Pulse Analysis",
            "version": "1.0",
            "author": "MQTT Oscilloscope Team",
            "description": "Analyze pulse characteristics (rise/fall time, width, overshoot)",
            "requires": [],
        }


# Legacy format support
PLUGIN_INFO = {
    "name": "Pulse Analysis",
    "version": "1.0",
    "metrics": ["rise_time", "fall_time", "pulse_width", "overshoot_percent"],
}


def calculate(times, amplitudes, roi_start, roi_end):
    """Legacy plugin interface."""
    plugin = PulseAnalysis()
    return plugin.calculate(times, amplitudes, roi_start, roi_end)
