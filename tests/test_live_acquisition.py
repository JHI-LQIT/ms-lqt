from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from ms_viewer.live_acquisition import LiveAcquisitionController
from ms_viewer.roi_statistics import ROIDefinition


def waveform(samples=3):
    values = np.ones(samples)
    return SimpleNamespace(mz=values, channels={"A": values})


def test_plot_worker_keeps_only_latest_pending_frame(monkeypatch):
    controller = LiveAcquisitionController(Mock())
    worker = controller._plot_worker
    started = []
    monkeypatch.setattr(worker, "start", lambda: started.append(worker.request.waveform))
    rendered = []
    controller.plot_ready.connect(lambda frame, channels, mode: rendered.append(frame))
    first, skipped, latest = [waveform(100_001) for _ in range(3)]

    for frame in (first, skipped, latest):
        controller.request_plot(frame, {}, "mz")

    assert [id(frame) for frame in started] == [id(first)]
    worker.run()
    worker.finished.emit()
    worker.run()
    worker.finished.emit()

    assert [id(frame) for frame in started] == [id(first), id(latest)]
    assert [id(frame) for frame in rendered] == [id(first), id(latest)]


def test_small_frame_supersedes_inflight_and_pending_plot_requests(monkeypatch):
    controller = LiveAcquisitionController(Mock())
    worker = controller._plot_worker
    start = Mock()
    monkeypatch.setattr(worker, "start", start)
    rendered = []
    controller.plot_ready.connect(lambda frame, channels, mode: rendered.append(channels))
    controller.request_plot(waveform(100_001), {}, "mz")
    controller.request_plot(waveform(100_001), {}, "mz")
    controller.request_plot(waveform(), {"A": (2, 0.5)}, "time")

    worker.run()
    worker.finished.emit()

    start.assert_called_once()
    assert len(rendered) == 1
    np.testing.assert_array_equal(rendered[0]["A"], [1, 1, 1])


def test_statistics_queue_drops_oldest_scans_and_reports_overflow(monkeypatch):
    controller = LiveAcquisitionController(Mock(), max_pending_scans=2)
    worker = controller._statistics_worker
    batches = []
    monkeypatch.setattr(
        worker, "start", lambda: batches.append(worker.waveforms_to_process)
    )
    drops = []
    controller.statistics_dropped.connect(drops.append)
    rois = [ROIDefinition("peak", 10, 20)]
    first, second, third, fourth = [waveform() for _ in range(4)]

    controller.record_waveform(first, has_regions=True)
    controller.request_statistics(first, rois, {})
    for frame in (second, third, fourth):
        controller.record_waveform(frame, has_regions=True)
        controller.request_statistics(frame, rois, {})

    assert drops == [1]
    assert len(batches) == 1
    assert batches[0][0] is first
    worker.finished.emit()
    assert [id(frame) for frame in batches[1]] == [id(third), id(fourth)]

    controller.close()
    worker.finished.emit()
    assert len(batches) == 2
