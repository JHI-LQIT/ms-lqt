"""Average scans on a common acquisition grid for ROI statistics."""

from collections.abc import Sequence
from datetime import timezone

import numpy as np

from .waveform_data import WaveformData


def _same_acquisition(reference: WaveformData, waveform: WaveformData) -> bool:
    if (
        waveform.selected_excitation_step != reference.selected_excitation_step
        or waveform.t0 != reference.t0
        or waveform.dt != reference.dt
        or waveform.sample_unit != reference.sample_unit
        or waveform.channels.keys() != reference.channels.keys()
    ):
        return False

    if any(
        waveform.channels[name].shape != samples.shape
        for name, samples in reference.channels.items()
    ):
        return False

    step = reference.selected_excitation_step
    return (
        waveform.trapping_parameters.excitation_steps[step]
        == reference.trapping_parameters.excitation_steps[step]
        and np.array_equal(waveform.time_ms, reference.time_ms)
        and np.array_equal(waveform.mz, reference.mz)
        and np.array_equal(waveform.f_AC_kHz, reference.f_AC_kHz)
        and np.array_equal(waveform.rf_amp_V0p, reference.rf_amp_V0p)
    )


def average_latest_batch(
    waveforms: Sequence[WaveformData],
) -> tuple[WaveformData, int]:
    """Average the latest contiguous group with matching settings and axes.

    Stop at the most recent acquisition change, even if older scans happen to
    match again. Return the waveform and the number of scans used.
    """
    if not waveforms:
        raise ValueError("Cannot average an empty waveform batch")

    reference = waveforms[-1]
    start = len(waveforms) - 1
    while start > 0 and _same_acquisition(reference, waveforms[start - 1]):
        start -= 1

    batch = waveforms[start:]
    if len(batch) == 1:
        return reference, 1

    averaged_channels = {
        name: np.mean([waveform.channels[name] for waveform in batch], axis=0)
        for name in reference.channels
    }

    # Legacy JSON timestamps can be naive UTC, while binary timestamps are aware.
    source = max(
        reversed(batch),
        key=lambda waveform: (
            waveform.timestamp
            if waveform.timestamp.tzinfo is not None
            else waveform.timestamp.replace(tzinfo=timezone.utc)
        ),
    )
    return WaveformData.from_preprocessed(source, averaged_channels), len(batch)
