from dataclasses import dataclass


@dataclass
class TrappingParameters:
    """Data class for trapping parameters."""

    secular_excitation_begin_ms: float
    secular_excitation_end_ms: float
    frequency_begin_khz: float
    frequency_end_khz: float
    frequency_sweep: bool
    rf_amplitude_begin_v0p: float
    rf_amplitude_end_v0p: float
    rf_amplitude_sweep: bool
    r0_cm: float
    f_RF_MHz: float
