"""Schedule plot preparation and ROI statistics for live waveforms."""

import logging
from collections import deque
from dataclasses import dataclass
from time import monotonic

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from .roi_statistics import ROIDefinition, ROIStatisticsCalculator
from .waveform_averaging import average_latest_batch
from .waveform_data import WaveformData


ChannelScaling = dict[str, tuple[float, float]]


def _scale_channels(
    waveform: WaveformData, scaling: ChannelScaling
) -> dict[str, np.ndarray]:
    channels = waveform.channels.copy()
    for name, (gain, offset) in scaling.items():
        if name in channels:
            channels[name] = gain * (channels[name] - offset)
    return channels


class StatisticsWorker(QThread):
    """Worker thread for calculating statistics to avoid blocking the GUI."""

    # Signal emitted when calculation is complete
    calculation_complete = pyqtSignal(list, int)  # (statistics, num_averaged_waveforms)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveforms_to_process = None
        self.rois = None
        self.scaling: ChannelScaling = {}
        self.statistics_calculator = None
        self._should_stop = False

    def setup_calculation(
        self,
        waveforms: list[WaveformData],
        rois: list[ROIDefinition],
        scaling: ChannelScaling,
        statistics_calculator: ROIStatisticsCalculator,
    ):
        """Set up the data for the next calculation."""
        self.waveforms_to_process = waveforms
        self.rois = rois
        self.scaling = scaling
        self.statistics_calculator = statistics_calculator
        self._should_stop = False

    def stop(self):
        """Signal the worker to stop processing."""
        self._should_stop = True

    def run(self):
        """Execute the statistics calculation in the background."""
        if not self.waveforms_to_process or not self.rois:
            return

        try:
            if self._should_stop:
                return

            averaged_waveform, num_averaged = average_latest_batch(
                self.waveforms_to_process
            )

            if self._should_stop:
                return

            scaled_waveform = WaveformData.from_preprocessed(
                source=averaged_waveform,
                channels=_scale_channels(averaged_waveform, self.scaling),
            )

            if self._should_stop:
                return

            # Calculate statistics
            if not self.statistics_calculator:
                return

            statistics = self.statistics_calculator.calculate_all_statistics(
                scaled_waveform, self.rois
            )

            # Emit result
            self.calculation_complete.emit(statistics, num_averaged)

        except Exception as e:
            logging.error(f"Error in statistics calculation thread: {e}", exc_info=True)


@dataclass
class PlotRequest:
    waveform: WaveformData
    scaling: ChannelScaling
    axis_mode: str


@dataclass
class StatisticsRequest:
    waveform: WaveformData
    rois: list[ROIDefinition]
    scaling: ChannelScaling


class PlotDataWorker(QThread):
    data_ready = pyqtSignal(object, dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.request: PlotRequest | None = None

    def run(self):
        if self.request is None:
            return
        request = self.request
        try:
            scaled = _scale_channels(request.waveform, request.scaling)
            self.data_ready.emit(request.waveform, scaled, request.axis_mode)
        except Exception:
            logging.exception("Error preparing plot data")


class LiveAcquisitionController(QObject):
    """Own live worker scheduling on the GUI thread."""

    plot_ready = pyqtSignal(object, dict, str)
    statistics_ready = pyqtSignal(list, int)
    statistics_dropped = pyqtSignal(int)

    def __init__(
        self,
        calculator: ROIStatisticsCalculator,
        *,
        statistics_interval: int = 10,
        minimum_interval_ms: int = 100,
        max_pending_scans: int = 64,
        parent=None,
    ):
        super().__init__(parent)
        self._calculator = calculator
        self._statistics_interval = max(1, statistics_interval)
        self._minimum_interval = minimum_interval_ms / 1000
        self._last_statistics_request = monotonic()
        self._waveform_count = 0
        self._scans: deque[WaveformData] = deque(maxlen=max(1, max_pending_scans))
        self._drop_count = 0

        self._statistics_worker = StatisticsWorker(self)
        self._statistics_worker.calculation_complete.connect(self._on_statistics_ready)
        self._statistics_worker.finished.connect(self._finish_statistics)
        self._statistics_busy = False
        self._accept_statistics_result = False
        self._pending_statistics: StatisticsRequest | None = None

        self._plot_worker = PlotDataWorker(self)
        self._plot_worker.data_ready.connect(self._on_plot_ready)
        self._plot_worker.finished.connect(self._finish_plot)
        self._plot_busy = False
        self._accept_plot_result = False
        self._pending_plot: PlotRequest | None = None
        self._closed = False

    def request_plot(
        self,
        waveform: WaveformData,
        scaling: ChannelScaling,
        axis_mode: str,
        *,
        redraw: bool = False,
    ):
        if self._closed:
            return
        if redraw:
            self._accept_plot_result = False

        request = PlotRequest(waveform, scaling, axis_mode)
        if len(waveform.mz) <= 100_000:
            self._pending_plot = None
            self._accept_plot_result = False
            self.plot_ready.emit(waveform, _scale_channels(waveform, scaling), axis_mode)
        elif self._plot_busy:
            self._pending_plot = request
        else:
            self._start_plot(request)

    def _start_plot(self, request: PlotRequest):
        self._plot_busy = self._accept_plot_result = True
        self._plot_worker.request = request
        self._plot_worker.start()

    def _on_plot_ready(self, waveform, scaled_channels, axis_mode):
        if self._accept_plot_result and not self._closed:
            self.plot_ready.emit(waveform, scaled_channels, axis_mode)

    def _finish_plot(self):
        self._plot_busy = False
        self._plot_worker.request = None
        request, self._pending_plot = self._pending_plot, None
        if request is not None and not self._closed:
            self._start_plot(request)

    def record_waveform(self, waveform: WaveformData, *, has_regions: bool) -> bool:
        """Buffer a scan and report whether the automatic ROI update is due."""
        if self._closed:
            return False
        self._waveform_count += 1
        if not has_regions:
            return False
        if len(self._scans) == self._scans.maxlen:
            self._drop_count += 1
            self.statistics_dropped.emit(self._drop_count)
        self._scans.append(waveform)
        return (
            self._waveform_count % self._statistics_interval == 0
            or self.statistics_refresh_due()
        )

    def statistics_refresh_due(self) -> bool:
        return monotonic() - self._last_statistics_request >= self._minimum_interval

    def request_statistics(
        self, waveform: WaveformData, rois: list[ROIDefinition], scaling: ChannelScaling
    ):
        if self._closed:
            return
        self._last_statistics_request = monotonic()
        request = StatisticsRequest(waveform, rois, scaling)
        if self._statistics_busy:
            self._pending_statistics = request
        else:
            self._start_statistics(request)

    def clear_statistics(self):
        self._scans.clear()
        self._pending_statistics = None
        self._accept_statistics_result = False

    def _start_statistics(self, request: StatisticsRequest):
        scans = list(self._scans) or [request.waveform]
        self._scans.clear()
        self._statistics_busy = self._accept_statistics_result = True
        self._statistics_worker.setup_calculation(
            scans, request.rois, request.scaling, self._calculator
        )
        self._statistics_worker.start()

    def _on_statistics_ready(self, statistics, scan_count):
        if self._accept_statistics_result and not self._closed:
            self.statistics_ready.emit(statistics, scan_count)

    def _finish_statistics(self):
        self._statistics_busy = False
        self._statistics_worker.waveforms_to_process = None
        request, self._pending_statistics = self._pending_statistics, None
        if request is not None and not self._closed:
            self._start_statistics(request)

    def close(self):
        """Stop accepting work and join both workers before Qt destroys them."""
        self._closed = True
        self._pending_plot = None
        self.clear_statistics()
        self._statistics_worker.stop()
        self._plot_worker.wait()
        self._statistics_worker.wait()
        self._plot_worker.request = None
        self._statistics_worker.waveforms_to_process = None
