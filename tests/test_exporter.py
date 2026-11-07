from datetime import datetime, timezone

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication

from ms_viewer.exporter import export_waveform_to_csv
from ms_viewer.gui import ExportWorker
from ms_viewer.trapping_parameters import TrapControls
from ms_viewer.waveform_data import WaveformData


@pytest.fixture
def large_waveform():
    trap = TrapControls()
    step = trap.excitation_steps[0]
    step.step_name = "Sweep"
    step.step_duration_ms = 110
    step.enable_frequency_sweep = True
    step.rf_amplitude_begin_V0p = 100

    samples = np.arange(120_000, dtype=float)
    return WaveformData(
        t0=0.0,
        dt=1e-6,
        channels={"B": samples, "A": -samples},
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        trapping_parameters=trap,
        selected_excitation_step=0,
        collector_metadata='{"run":42}',
    )


def test_background_csv_matches_synchronous_export(tmp_path, large_waveform):
    app = QCoreApplication.instance() or QCoreApplication([])
    assert len(large_waveform.mz) > 100_000
    assert export_waveform_to_csv(str(tmp_path / "sync"), large_waveform)

    worker = ExportWorker()
    results = []
    worker.export_complete.connect(lambda success, message: results.append(success))
    worker.setup_export(str(tmp_path / "background.csv"), large_waveform)
    worker.start()
    assert worker.wait()
    app.processEvents()

    assert results == [True]
    contents = (tmp_path / "background.csv").read_bytes()
    assert contents == (tmp_path / "sync.csv").read_bytes()
    assert b"# Timestamp: 2026-01-01T00:00:00+00:00" in contents
    assert b"# Step Name: Sweep" in contents
    assert b'# Collected metadata: {"run":42}' in contents
    assert contents.splitlines()[2] == b"mz,A,B"


@pytest.mark.parametrize("destination_exists", [False, True])
def test_cancelled_export_leaves_destination_untouched(
    tmp_path, large_waveform, destination_exists
):
    destination = tmp_path / "capture.csv"
    if destination_exists:
        destination.write_bytes(b"previous capture")
    checks = iter([False, True])  # Cancel after the first batch is written.

    assert not export_waveform_to_csv(
        str(destination), large_waveform, is_cancelled=lambda: next(checks)
    )

    if destination_exists:
        assert destination.read_bytes() == b"previous capture"
        assert list(tmp_path.iterdir()) == [destination]
    else:
        assert list(tmp_path.iterdir()) == []
