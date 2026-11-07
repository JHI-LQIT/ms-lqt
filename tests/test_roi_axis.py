from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from ms_viewer.gui import MainWindow
from ms_viewer.mqtt_client import MqttClient
from ms_viewer.roi_axis import ROIAxisMapping


@pytest.mark.parametrize("reverse_samples", [False, True])
@pytest.mark.parametrize("descending_display", [False, True])
def test_roi_mapping_round_trip(reverse_samples, descending_display):
    mz = np.array([100, 200, 400])
    display = np.array([10, 20, 60])
    if descending_display:
        display = display[::-1]
    if reverse_samples:
        mz, display = mz[::-1], display[::-1]

    mapping = ROIAxisMapping(mz, display)

    expected = (15, 30) if not descending_display else (17.5, 40)
    assert mapping.to_display((150, 250)) == expected
    assert mapping.to_mz(expected) == (150, 250)
    assert mapping.to_display((0, 500)) == (10, 60)


@pytest.mark.parametrize(
    ("mz", "display"),
    [
        ([], []),
        ([1], [2]),
        ([1, 2], [1]),
        ([1, 1, 2], [1, 2, 3]),
        ([1, 2, 3], [7, 7, 7]),
        ([1, 2, 3], [3, 1, 2]),
        ([1, 2, 3], [1, np.nan, 3]),
    ],
)
def test_rejects_ambiguous_or_unusable_axes(mz, display):
    with pytest.raises(ValueError):
        ROIAxisMapping(np.array(mz), np.array(display))


def test_roi_markers_follow_rendered_axis_and_preserve_mz(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(MqttClient(), str(tmp_path / "viewer.yaml"))
    monkeypatch.setattr(window, "_calculate_and_display_statistics_throttled", lambda: None)

    waveform = SimpleNamespace(
        mz=np.array([100, 200, 400]),
        f_AC_kHz=np.array([60, 20, 10]),
        rf_amp_V0p=np.full(3, 100),
        time_ms=np.array([1, 2, 3]),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    channels = {"A": np.array([1, 2, 3])}
    try:
        window._add_region_visual(150, 300)
        region = window.regions[0]
        window.state["gui"]["x_axis_mode"] = "frequency"
        window._on_plot_data_ready(waveform, channels, "frequency")
        assert region.getRegion() == (15, 40)

        region.setRegion((10, 20))
        assert window.region_mz_bounds[region] == (200, 400)

        waveform.f_AC_kHz = np.array([120, 40, 20])
        window._on_plot_data_ready(waveform, channels, "frequency")
        assert region.getRegion() == (20, 40)
        assert window.region_mz_bounds[region] == (200, 400)

        window.state["gui"]["x_axis_mode"] = "rf_amplitude"
        window._on_plot_data_ready(waveform, channels, "rf_amplitude")
        assert not region.isVisible()
        window._on_plot_data_ready(waveform, channels, "frequency")
        assert not region.isVisible()
        assert window.region_mz_bounds[region] == (200, 400)
        window.region_table.item(0, 1).setText("250")
        assert window.region_mz_bounds[region] == (250, 400)

        window.state["gui"]["x_axis_mode"] = "mz"
        window._on_plot_data_ready(waveform, channels, "mz")
        assert region.isVisible()
        assert region.getRegion() == (250, 400)
    finally:
        window.close()
        app.processEvents()
