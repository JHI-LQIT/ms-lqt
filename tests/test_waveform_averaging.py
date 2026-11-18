import json
from datetime import datetime, timezone
from threading import Event
from time import monotonic, sleep
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from ms_viewer.gui import MainWindow
from ms_viewer.live_acquisition import StatisticsWorker
from ms_viewer.mqtt_client import MqttClient
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
    worker.calculation_complete.connect(
        lambda stats, count, capture_time: results.append((count, capture_time))
    )
    worker.setup_calculation(
        [old, latest], [ROIDefinition("peak", 10, 20)], {}, calculator
    )

    worker.run()

    assert results == [(1, latest.timestamp)]
    calculated_waveform = calculator.calculate_all_statistics.call_args.args[0]
    np.testing.assert_array_equal(calculated_waveform.channels["A"], 4)


def test_statistics_outputs_use_batch_timestamp_when_newer_waveform_arrives(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    config = tmp_path / "viewer.yaml"
    config.write_text(
        "preferences:\n"
        "  statistics_update_interval: 100\n"
        "  statistics_update_min_interval_ms: 60000\n",
        encoding="utf-8",
    )
    client = MqttClient()
    window = MainWindow(client, str(config))
    publish = Mock()
    monkeypatch.setattr(client, "is_connected", lambda: True)
    monkeypatch.setattr(client, "publish", publish)

    calculating = Event()
    release = Event()
    calculate = window.statistics_calculator.calculate_all_statistics

    def delayed_calculate(*args):
        calculating.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Test did not release the statistics worker")
        return calculate(*args)

    monkeypatch.setattr(
        window.statistics_calculator, "calculate_all_statistics", delayed_calculate
    )
    first, last_in_batch, newer = [make_waveform(value) for value in (2, 4, 10)]
    capture_time = last_in_batch.timestamp
    last_in_batch.timestamp = capture_time.replace(tzinfo=None)
    try:
        window.stats_checkboxes["mean"].setChecked(True)
        window._add_region_visual(first.mz[0], first.mz[-1], "peak")
        window.on_waveform_received(first)
        window.on_waveform_received(last_in_batch)
        window._calculate_and_display_statistics()
        assert calculating.wait(timeout=5)

        window.on_waveform_received(newer)
        release.set()
        deadline = monotonic() + 5
        while not publish.called and monotonic() < deadline:
            app.processEvents()
            sleep(0.01)

        publish.assert_called_once()
        topic, raw_payload = publish.call_args.args
        payload = json.loads(raw_payload)
        assert topic == "lqt/lqt_ms/statistics"
        assert payload["timestamp"] == capture_time.isoformat()
        assert payload["num_averaged_waveforms"] == 2
        assert window.statistics_history["mean"]["A:peak"] == [(capture_time, 3)]
        curve = window.stats_plot_curves["mean:A:peak"]
        np.testing.assert_array_equal(curve.xData, [capture_time.timestamp()])
        np.testing.assert_array_equal(curve.yData, [3])
        assert window.current_waveform is newer
    finally:
        release.set()
        window.close()
