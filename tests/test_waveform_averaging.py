from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest

from ms_viewer.gui import StatisticsWorker
from ms_viewer.roi_statistics import ROIDefinition
from ms_viewer.trapping_parameters import ExcitationStep, TrapControls
from ms_viewer.waveform_averaging import average_latest_batch
from ms_viewer.waveform_data import WaveformData


def make_waveform(value, *, dt=1e-3, t0=0.0, step=0, rf_frequency=2.112, ac_mv=10):
    trap = TrapControls()
    trap.basic_controls.injection_duration_ms = 0
    trap.basic_controls.f_RF_MHz = rf_frequency
    trap.excitation_steps = [
        ExcitationStep(
            step_duration_ms=4,
            enable_frequency_sweep=True,
            rf_amplitude_begin_V0p=100,
            ac_amplitude_begin_mV=ac_mv,
        )
        for _ in range(2)
    ]
    return WaveformData(
        t0=t0,
        dt=dt,
        channels={"A": np.full(12, value)},
        timestamp=datetime(2026, 1, 1, 0, 0, value, tzinfo=timezone.utc),
        trapping_parameters=trap,
        selected_excitation_step=step,
    )


@pytest.mark.parametrize(
    "changed_settings",
    [
        {"dt": 0.95e-3},
        {"t0": 0.001},
        {"step": 1},
        {"rf_frequency": 2.2},
        {"ac_mv": 20},
    ],
)
def test_equal_shape_scans_are_split_at_acquisition_change(changed_settings):
    older = make_waveform(20)
    changed = make_waveform(30, **changed_settings)
    first = make_waveform(2)
    latest = make_waveform(4)
    assert changed.channels["A"].shape == latest.channels["A"].shape

    averaged, count = average_latest_batch([older, changed, first, latest])

    assert count == 2
    np.testing.assert_array_equal(averaged.channels["A"], 3)
    np.testing.assert_array_equal(averaged.mz, latest.mz)
    assert averaged.timestamp == latest.timestamp


def test_latest_capture_timestamp_accepts_legacy_json_utc():
    first = make_waveform(2)
    latest = make_waveform(4)
    latest.timestamp = latest.timestamp.replace(tzinfo=None)

    averaged, count = average_latest_batch([first, latest])

    assert count == 2
    assert averaged.timestamp == latest.timestamp


def test_statistics_worker_reports_actual_number_of_scans_used():
    old = make_waveform(10, rf_frequency=2.2)
    latest = make_waveform(4)
    calculator = Mock()
    calculator.calculate_all_statistics.return_value = []
    worker = StatisticsWorker()
    results = []
    worker.calculation_complete.connect(lambda stats, count: results.append(count))
    worker.setup_calculation(
        [old, latest], [ROIDefinition("peak", 10, 20)], {}, calculator
    )

    worker.run()

    assert results == [1]
    calculated_waveform = calculator.calculate_all_statistics.call_args.args[0]
    np.testing.assert_array_equal(calculated_waveform.channels["A"], 4)
