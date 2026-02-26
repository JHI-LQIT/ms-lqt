from datetime import datetime
from typing import Dict

import numpy as np

from .trapping_parameters import TrapControls


class WaveformData:
    """
    Container for oscilloscope waveform data.

    Attributes:
        t0 (float): Start time in seconds
        dt (float): Time step between samples in seconds
        times (np.ndarray): Computed time values for each sample
        channels (Dict[str, np.ndarray]): Dictionary of channel names to amplitude arrays
        timestamp (datetime): Timestamp when the waveform was captured
        trapping_parameters (TrapControls): Trapping control parameters
        selected_excitation_step (int): Index of the excitation step used for m/z calculation
    """

    def __init__(
        self,
        t0: float,
        dt: float,
        channels: Dict[str, np.ndarray],
        timestamp: datetime,
        trapping_parameters: TrapControls,
        selected_excitation_step: int,
        collector_metadata: str | None = None,
    ):
        """
        Initialize WaveformData.

        Args:
            t0: Start time in seconds
            dt: Time step between samples in seconds
            channels: Dictionary mapping channel names to amplitude arrays
            timestamp: Capture timestamp
            trapping_parameters: TrapControls with basic controls and excitation steps
            selected_excitation_step: Index of the excitation step to use for m/z calculation (default: 0)
            collector_metadata: Optional JSON metadata from collector parameters topic
        Raises:
            ValueError: If parameters are invalid
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        if not channels:
            raise ValueError("At least one channel must be provided")

        # Validate all channels have the same length
        lengths = [len(data) for data in channels.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All channels must have equal length. Got lengths: {lengths}"
            )

        self.t0 = t0
        self.dt = dt
        self.channels = channels
        self.timestamp = timestamp
        self.trapping_parameters = trapping_parameters
        self.selected_excitation_step = selected_excitation_step
        self.collector_metadata = collector_metadata

        # Compute time axis
        N = lengths[0]
        self.mz, self.f_AC_kHz, self.rf_amp_V0p, start_index, n_samples = (
            self.calc_mz_values(dt, N, selected_excitation_step)
        )

        # Cut channels to match m/z length and sort by m/z
        sorted_indices = np.argsort(self.mz)
        self.mz = self.mz[sorted_indices]
        self.f_AC_kHz = self.f_AC_kHz[sorted_indices]
        self.rf_amp_V0p = self.rf_amp_V0p[sorted_indices]
        for key in self.channels:
            self.channels[key] = self.channels[key][
                start_index : start_index + n_samples
            ][sorted_indices]

    def __repr__(self):
        return (
            f"WaveformData(t0={self.t0}, dt={self.dt}, "
            f"channels={list(self.channels.keys())}, "
            f"n_samples={len(self.mz)}, "
            f"timestamp={self.timestamp.isoformat()}, "
            f"trapping_parameters={self.trapping_parameters}, "
            f"selected_excitation_step={self.selected_excitation_step})"
        )

    @property
    def _multiplier(self) -> float:
        """Calculate the m/z conversion multiplier from trap parameters.

        The multiplier is used in the formula: m/z = multiplier * V_RF / f_secular
        where multiplier = 34.45 / (r0^2 * f_RF)
        """
        return 34.45 / (
            self.trapping_parameters.basic_controls.r0_cm**2
            * self.trapping_parameters.basic_controls.f_RF_MHz
        )

    def calc_mz_values(
        self, dt: float, N: int, step_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Calculate m/z values from trapping parameters and time data.

        Args:
            dt (float): The time interval between samples.
            N (int): The number of samples.
            step_index (int): Index of the excitation step to use (default: 0).

        Returns:
            np.ndarray: The calculated m/z values.
            np.ndarray: The frequency array used in the calculation.
            np.ndarray: The RF amplitude array used in the calculation.
            int: The start index used in the calculation.
            int: The number of samples used in the calculation.
        """
        # Validate step index
        if step_index < 0 or step_index >= len(
            self.trapping_parameters.excitation_steps
        ):
            step_index = 0

        excitation_step = self.trapping_parameters.excitation_steps[step_index]

        # Calculate start and end time for this excitation step
        start_time_ms = self.trapping_parameters.calculate_step_start_time(step_index)
        end_time_ms = start_time_ms + excitation_step.step_duration_ms

        start_idx = int(np.ceil(start_time_ms / (1000.0 * dt)))
        end_idx = int(np.floor(end_time_ms / (1000.0 * dt)))
        if end_idx > N:
            end_idx = N
        if start_idx > end_idx:
            start_idx = end_idx
        n_samples = end_idx - start_idx

        # Get frequency values
        f_begin = excitation_step.frequency_begin_kHz
        f_AC = (
            np.linspace(f_begin, excitation_step.frequency_end_kHz, n_samples)
            if excitation_step.enable_frequency_sweep
            else np.full(n_samples, f_begin)
        )

        # Get RF amplitude values
        rf_begin = excitation_step.rf_amplitude_begin_V0p
        rf_amp = (
            np.linspace(rf_begin, excitation_step.rf_amplitude_end_V0p, n_samples)
            if excitation_step.enable_rf_amplitude_sweep
            else np.full(n_samples, rf_begin)
        )

        return self._multiplier * rf_amp / f_AC, f_AC, rf_amp, start_idx, n_samples

    @classmethod
    def from_preprocessed(
        cls,
        source: "WaveformData",
        channels: Dict[str, np.ndarray],
        timestamp=None,
    ) -> "WaveformData":
        """Create a WaveformData from pre-sliced channels, reusing the source m/z data.

        Use this instead of the normal constructor when channels are already sliced
        and sorted (e.g. after averaging or scaling).  Re-running the constructor on
        already-sliced data re-applies ``calc_mz_values``, which computes a
        ``start_idx`` relative to wall-clock time zero; for excitation steps beyond
        the first that offset can exceed the (already short) channel length, resulting
        in an empty m/z array and broken statistics.

        Args:
            source: The WaveformData whose m/z arrays and metadata should be reused.
            channels: Pre-sliced and sorted channel arrays (same shape as source.channels).
            timestamp: Optional override for the timestamp (defaults to source.timestamp).

        Returns:
            A new WaveformData with the given channels but the source's m/z data.
        """
        instance: "WaveformData" = object.__new__(cls)
        instance.t0 = source.t0
        instance.dt = source.dt
        instance.channels = channels
        instance.timestamp = timestamp if timestamp is not None else source.timestamp
        instance.trapping_parameters = source.trapping_parameters
        instance.selected_excitation_step = source.selected_excitation_step
        instance.collector_metadata = source.collector_metadata
        instance.mz = source.mz.copy()
        instance.f_AC_kHz = source.f_AC_kHz.copy()
        instance.rf_amp_V0p = source.rf_amp_V0p.copy()
        return instance

    def mz_from_rf_and_freq(self, rf_amp_V0p: float, f_AC_kHz: float) -> float:
        """Calculate m/z from RF amplitude and frequency."""
        return self._multiplier * rf_amp_V0p / f_AC_kHz

    def freq_from_mz_and_rf(self, mz: float, rf_amp_V0p: float) -> float:
        """Calculate frequency from m/z and RF amplitude."""
        return self._multiplier * rf_amp_V0p / mz

    def rf_from_mz_and_freq(self, mz: float, f_AC_kHz: float) -> float:
        """Calculate RF amplitude from m/z and frequency."""
        return mz * f_AC_kHz / self._multiplier
