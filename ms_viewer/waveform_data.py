from datetime import datetime
from typing import Dict

import numpy as np

from .trapping_parameters import TrappingParameters


class WaveformData:
    """
    Container for oscilloscope waveform data.

    Attributes:
        t0 (float): Start time in seconds
        dt (float): Time step between samples in seconds
        times (np.ndarray): Computed time values for each sample
        channels (Dict[str, np.ndarray]): Dictionary of channel names to amplitude arrays
        timestamp (datetime): Timestamp when the waveform was captured
    """

    def __init__(
        self,
        t0: float,
        dt: float,
        channels: Dict[str, np.ndarray],
        timestamp: datetime,
        trapping_parameters: TrappingParameters,
    ):
        """
        Initialize WaveformData.

        Args:
            t0: Start time in seconds
            dt: Time step between samples in seconds
            channels: Dictionary mapping channel names to amplitude arrays
            timestamp: Capture timestamp
            trapping_parameters: TrappingParameters
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
        # Compute time axis
        N = lengths[0]
        self.mz, start_index, n_samples = self.calc_mz_values(trapping_parameters, dt, N)

        # Cut channels to match m/z length and sort by m/z
        sorted_indices = np.argsort(self.mz)
        self.mz = self.mz[sorted_indices]
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
            f"trapping_parameters={self.trapping_parameters})"
        )

    def calc_mz_values(
        self, trapping_params: TrappingParameters, dt: float, N: int
    ) -> tuple[np.ndarray, int, int]:
        """
        Calculate m/z values from trapping parameters and time data.

        Args:
            trapping_params (TrappingParameters): The trapping parameters.
            dt (float): The time interval between samples.
            N (int): The number of samples.

        Returns:
            np.ndarray: The calculated m/z values.
            int: The start index used in the calculation.
            int: The number of samples used in the calculation.
        """

        start_idx = int(
            np.ceil(trapping_params.secular_excitation_begin_ms / (1000.0 * dt))
        )
        end_idx = int(
            np.floor(trapping_params.secular_excitation_end_ms / (1000.0 * dt))
        )
        if end_idx > N:
            end_idx = N
        n_samples = end_idx - start_idx

        f_begin = trapping_params.frequency_begin_khz
        f_AC = (
            np.linspace(f_begin, trapping_params.frequency_end_khz, n_samples)
            if trapping_params.frequency_sweep
            else np.full(n_samples, f_begin)
        )

        rf_begin = trapping_params.rf_amplitude_begin_v0p
        rf_amp = (
            np.linspace(rf_begin, trapping_params.rf_amplitude_end_v0p, n_samples)
            if trapping_params.rf_amplitude_sweep
            else np.full(n_samples, rf_begin)
        )

        multiplier = 34.45 / (trapping_params.r0_cm**2 * trapping_params.f_RF_MHz)

        return multiplier * rf_amp / f_AC, start_idx, n_samples
