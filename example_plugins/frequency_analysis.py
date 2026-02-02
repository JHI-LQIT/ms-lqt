"""
Frequency Analysis Plugin

Performs FFT-based frequency domain analysis on waveform data.

Provides metrics like dominant frequency, total harmonic distortion,
and spectral centroid.

Requirements: scipy
"""

from typing import TYPE_CHECKING, Dict, List

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


class FrequencyAnalysis(CustomStatisticPlugin):
    """
    Analyze frequency domain characteristics of waveform.

    Uses FFT (Fast Fourier Transform) to extract frequency information.
    """

    def calculate(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        roi_start: float,
        roi_end: float,
    ) -> Dict[str, float]:
        """
        Calculate frequency domain metrics.

        Args:
            times: Time values within the ROI
            amplitudes: Amplitude values within the ROI
            roi_start: Start time of ROI
            roi_end: End time of ROI

        Returns:
            Dictionary with frequency metrics
        """
        # Calculate sampling rate
        if len(times) < 2:
            return {
                "dominant_frequency": 0.0,
                "thd_percent": 0.0,
                "spectral_centroid": 0.0,
            }

        dt = np.mean(np.diff(times))
        fs = 1.0 / dt  # Sampling frequency

        # Perform FFT
        fft_vals = np.fft.rfft(amplitudes)
        fft_freq = np.fft.rfftfreq(len(amplitudes), dt)

        # Get magnitude spectrum
        magnitude = np.abs(fft_vals)

        # Find dominant frequency (skip DC component at index 0)
        dominant_idx = 0
        if len(magnitude) > 1:
            dominant_idx = np.argmax(magnitude[1:]) + 1
            dominant_freq = fft_freq[dominant_idx]
        else:
            dominant_freq = 0.0

        # Calculate Total Harmonic Distortion (THD)
        # THD = sqrt(sum of harmonic powers / fundamental power)
        if dominant_idx > 0 and len(magnitude) > dominant_idx:
            fundamental_power = magnitude[dominant_idx] ** 2
            total_power = np.sum(magnitude[1:] ** 2)  # Exclude DC

            if fundamental_power > 0:
                harmonic_power = total_power - fundamental_power
                thd = np.sqrt(harmonic_power / fundamental_power)
                thd_percent = thd * 100.0
            else:
                thd_percent = 0.0
        else:
            thd_percent = 0.0

        # Calculate spectral centroid (weighted mean frequency)
        # Centroid = sum(freq * magnitude) / sum(magnitude)
        total_magnitude = np.sum(magnitude[1:])  # Exclude DC
        if total_magnitude > 0:
            spectral_centroid = np.sum(fft_freq[1:] * magnitude[1:]) / total_magnitude
        else:
            spectral_centroid = 0.0

        return {
            "dominant_frequency": float(dominant_freq),
            "thd_percent": float(thd_percent),
            "spectral_centroid": float(spectral_centroid),
        }

    def get_metrics(self) -> List[str]:
        """Return list of metrics this plugin provides."""
        return ["dominant_frequency", "thd_percent", "spectral_centroid"]

    def get_info(self) -> dict:
        """Return plugin metadata."""
        return {
            "name": "Frequency Analysis",
            "version": "1.0",
            "author": "MQTT Oscilloscope Team",
            "description": "FFT-based frequency domain analysis",
            "requires": [],
        }


# Legacy format support
PLUGIN_INFO = {
    "name": "Frequency Analysis",
    "version": "1.0",
    "metrics": ["dominant_frequency", "thd_percent", "spectral_centroid"],
}


def calculate(times, amplitudes, roi_start, roi_end):
    """Legacy plugin interface."""
    plugin = FrequencyAnalysis()
    return plugin.calculate(times, amplitudes, roi_start, roi_end)
