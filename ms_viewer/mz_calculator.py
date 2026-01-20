import numpy as np

from .trapping_parameters import TrappingParameters


def calc_mz_values(
    trapping_params: TrappingParameters, dt: float, N: int
) -> np.ndarray:
    """
    Calculate m/z values from trapping parameters and time data.

    Args:
        trapping_params (TrappingParameters): The trapping parameters.
        dt (float): The time interval between samples.
        N (int): The number of samples.

    Returns:
        np.ndarray: The calculated m/z values.
    """

    start_idx = int(
        np.ceil(trapping_params.secular_excitation_begin_ms / (1000.0 * dt))
    )
    n_samples = N - start_idx

    f_begin = trapping_params.frequency_begin_khz * 1e3
    f_AC = (
        np.linspace(f_begin, trapping_params.frequency_end_khz * 1e3, n_samples)
        if trapping_params.frequency_sweep
        else np.full(n_samples, f_begin)
    )

    rf_begin = trapping_params.rf_amplitude_begin_v0p
    rf_amp = (
        np.linspace(rf_begin, trapping_params.rf_amplitude_end_v0p, n_samples)
        if trapping_params.rf_amplitude_sweep
        else np.full(n_samples, rf_begin)
    )

    multiplier = 34.45 / trapping_params.r0_cm**2 * trapping_params.f_RF_MHz

    return multiplier * rf_amp / f_AC
