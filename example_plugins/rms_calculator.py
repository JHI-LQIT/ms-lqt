"""
RMS Calculator Plugin

Calculates Root Mean Square (RMS) value and RMS in decibels.

RMS is useful for measuring the effective value of an AC signal.
"""

from typing import TYPE_CHECKING, Dict, List

import numpy as np

# Import base class - this import path works when plugin is loaded
if TYPE_CHECKING:
    from waveform_viewer.roi_statistics import CustomStatisticPlugin
else:
    try:
        from waveform_viewer.roi_statistics import CustomStatisticPlugin
    except ImportError:
        # Fallback for standalone testing
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


class RMSCalculator(CustomStatisticPlugin):
    """
    Calculate RMS (Root Mean Square) statistics.

    The RMS value represents the effective or DC equivalent value
    of an AC signal.
    """

    def calculate(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        roi_start: float,
        roi_end: float,
    ) -> Dict[str, float]:
        """
        Calculate RMS value and RMS in decibels.

        Args:
            times: Time values within the ROI
            amplitudes: Amplitude values within the ROI
            roi_start: Start time of ROI
            roi_end: End time of ROI

        Returns:
            Dictionary with 'rms' and 'rms_db' values
        """
        # Calculate RMS: sqrt(mean(x^2))
        rms = np.sqrt(np.mean(amplitudes**2))

        # Calculate RMS in decibels (20*log10 for voltage/amplitude)
        # Avoid log(0) by using a small epsilon
        rms_db = 20 * np.log10(rms + 1e-20)

        return {"rms": float(rms), "rms_db": float(rms_db)}

    def get_metrics(self) -> List[str]:
        """Return list of metrics this plugin provides."""
        return ["rms", "rms_db"]

    def get_info(self) -> dict:
        """Return plugin metadata."""
        return {
            "name": "RMS Calculator",
            "version": "1.0",
            "author": "MQTT Oscilloscope Team",
            "description": "Calculate RMS value and RMS in decibels",
            "requires": [],
        }


# Legacy plugin format support (for backwards compatibility)
PLUGIN_INFO = {"name": "RMS Calculator", "version": "1.0", "metrics": ["rms", "rms_db"]}


def calculate(times, amplitudes, roi_start, roi_end):
    """Legacy plugin interface."""
    plugin = RMSCalculator()
    return plugin.calculate(times, amplitudes, roi_start, roi_end)
