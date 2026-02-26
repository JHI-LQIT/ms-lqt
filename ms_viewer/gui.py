"""
Main GUI for the waveform viewer application.

This module provides the main window with PyQtGraph plotting, control panels,
region management, and state persistence.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt as QtCore
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QLineEdit, QMainWindow, QMessageBox,
                             QPushButton, QRadioButton, QScrollArea, QSpinBox,
                             QSplitter, QStatusBar, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from . import state_manager
from .data_model import WaveformData
from .exporter import build_payload, export_waveform_to_csv
from .mqtt_client import MqttClient
from .roi_statistics import (ROIDefinition, ROIStatistics,
                             ROIStatisticsCalculator)
from .trapping_parameters import ExcitationStep, TrapControls

logger = logging.getLogger(__name__)


class MqttConfigDialog(QDialog):
    """Separate dialog window for MQTT configuration."""

    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.main_window = parent
        self.setWindowTitle("MQTT Configuration")
        self.setMinimumSize(500, 600)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the MQTT configuration UI."""
        layout = QVBoxLayout()

        # Primary broker group
        primary_group = QGroupBox("Primary Broker")
        primary_layout = QVBoxLayout()

        # Broker host
        host_layout = QHBoxLayout()
        host_layout.addWidget(QLabel("Broker:"))
        self.main_window.broker_host_input = QLineEdit()
        self.main_window.broker_host_input.setText(
            self.main_window.state["mqtt"]["broker_host"]
        )
        host_layout.addWidget(self.main_window.broker_host_input)
        primary_layout.addLayout(host_layout)

        # Broker port
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.main_window.broker_port_input = QSpinBox()
        self.main_window.broker_port_input.setRange(1, 65535)
        self.main_window.broker_port_input.setValue(
            self.main_window.state["mqtt"]["broker_port"]
        )
        port_layout.addWidget(self.main_window.broker_port_input)
        primary_layout.addLayout(port_layout)

        # Input topic
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Topic:"))
        self.main_window.input_topic_input = QLineEdit()
        self.main_window.input_topic_input.setText(
            self.main_window.state["mqtt"]["input_topic"]
        )
        input_layout.addWidget(self.main_window.input_topic_input)
        primary_layout.addLayout(input_layout)

        # Trapping parameters topic
        trapping_layout = QHBoxLayout()
        trapping_layout.addWidget(QLabel("Trapping Parameters Topic:"))
        self.main_window.trapping_parameters_topic_input = QLineEdit()
        self.main_window.trapping_parameters_topic_input.setText(
            self.main_window.state["mqtt"]["trapping_parameters_topic"]
        )
        trapping_layout.addWidget(self.main_window.trapping_parameters_topic_input)
        primary_layout.addLayout(trapping_layout)

        # Collector parameters topic
        collector_layout = QHBoxLayout()
        collector_layout.addWidget(QLabel("Collector Parameters Topic:"))
        self.main_window.collector_parameters_topic_input = QLineEdit()
        self.main_window.collector_parameters_topic_input.setText(
            self.main_window.state["mqtt"].get(
                "collector_parameters_topic", "lqt/collector/parameters"
            )
        )
        collector_layout.addWidget(self.main_window.collector_parameters_topic_input)
        primary_layout.addLayout(collector_layout)

        # Output topic
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Topic:"))
        self.main_window.output_topic_input = QLineEdit()
        self.main_window.output_topic_input.setText(
            self.main_window.state["mqtt"]["output_topic"]
        )
        output_layout.addWidget(self.main_window.output_topic_input)
        primary_layout.addLayout(output_layout)

        # Statistics topic
        stats_topic_layout = QHBoxLayout()
        stats_topic_layout.addWidget(QLabel("Statistics Topic:"))
        self.main_window.stats_topic_input = QLineEdit()
        self.main_window.stats_topic_input.setText(
            self.main_window.state["mqtt"]["statistics_topic"]
        )

        stats_topic_layout.addWidget(self.main_window.stats_topic_input)
        primary_layout.addLayout(stats_topic_layout)

        # Apply button
        self.main_window.apply_mqtt_btn = QPushButton("Apply & Connect")
        self.main_window.apply_mqtt_btn.clicked.connect(
            self.main_window.apply_mqtt_config
        )
        primary_layout.addWidget(self.main_window.apply_mqtt_btn)

        # Disconnect button
        self.main_window.disconnect_btn = QPushButton("Disconnect")
        self.main_window.disconnect_btn.clicked.connect(
            self.main_window.disconnect_mqtt
        )
        self.main_window.disconnect_btn.setEnabled(False)
        primary_layout.addWidget(self.main_window.disconnect_btn)

        primary_group.setLayout(primary_layout)
        layout.addWidget(primary_group)

        # Separator
        layout.addSpacing(10)

        # Secondary broker group
        secondary_group = QGroupBox("Secondary Broker for Output/Statistics")
        secondary_layout = QVBoxLayout()

        # Enable checkbox
        enable_output_layout = QHBoxLayout()
        self.main_window.output_broker_enabled = QCheckBox("Enable secondary broker")
        self.main_window.output_broker_enabled.setChecked(
            self.main_window.state["mqtt"].get("output_broker_enabled", False)
        )
        self.main_window.output_broker_enabled.stateChanged.connect(
            self.main_window._on_output_broker_enabled_changed
        )
        enable_output_layout.addWidget(self.main_window.output_broker_enabled)
        secondary_layout.addLayout(enable_output_layout)

        # Output broker host
        output_host_layout = QHBoxLayout()
        output_host_layout.addWidget(QLabel("Output Broker:"))
        self.main_window.output_broker_host_input = QLineEdit()
        self.main_window.output_broker_host_input.setText(
            self.main_window.state["mqtt"].get("output_broker_host", "localhost")
        )
        output_host_layout.addWidget(self.main_window.output_broker_host_input)
        secondary_layout.addLayout(output_host_layout)

        # Output broker port
        output_port_layout = QHBoxLayout()
        output_port_layout.addWidget(QLabel("Output Port:"))
        self.main_window.output_broker_port_input = QSpinBox()
        self.main_window.output_broker_port_input.setRange(1, 65535)
        self.main_window.output_broker_port_input.setValue(
            self.main_window.state["mqtt"].get("output_broker_port", 1883)
        )
        output_port_layout.addWidget(self.main_window.output_broker_port_input)
        secondary_layout.addLayout(output_port_layout)

        # Output broker topic
        output_topic_layout = QHBoxLayout()
        output_topic_layout.addWidget(QLabel("Output Topic:"))
        self.main_window.output_broker_output_topic_input = QLineEdit()
        self.main_window.output_broker_output_topic_input.setText(
            self.main_window.state["mqtt"].get(
                "output_broker_output_topic", "oscilloscope/republish"
            )
        )
        output_topic_layout.addWidget(self.main_window.output_broker_output_topic_input)
        secondary_layout.addLayout(output_topic_layout)

        # Output broker statistics topic
        output_stats_topic_layout = QHBoxLayout()
        output_stats_topic_layout.addWidget(QLabel("Output Stats Topic:"))
        self.main_window.output_broker_stats_topic_input = QLineEdit()
        self.main_window.output_broker_stats_topic_input.setText(
            self.main_window.state["mqtt"].get(
                "output_broker_statistics_topic", "lqt/lqt_ms/statistics"
            )
        )
        output_stats_topic_layout.addWidget(
            self.main_window.output_broker_stats_topic_input
        )
        secondary_layout.addLayout(output_stats_topic_layout)

        # Apply output broker button
        self.main_window.apply_output_mqtt_btn = QPushButton(
            "Apply Output Broker & Connect"
        )
        self.main_window.apply_output_mqtt_btn.clicked.connect(
            self.main_window.apply_output_mqtt_config
        )
        secondary_layout.addWidget(self.main_window.apply_output_mqtt_btn)

        # Disconnect output broker button
        self.main_window.disconnect_output_btn = QPushButton("Disconnect Output Broker")
        self.main_window.disconnect_output_btn.clicked.connect(
            self.main_window.disconnect_output_mqtt
        )
        self.main_window.disconnect_output_btn.setEnabled(False)
        secondary_layout.addWidget(self.main_window.disconnect_output_btn)

        secondary_group.setLayout(secondary_layout)
        layout.addWidget(secondary_group)

        # Update control states based on checkbox
        self.main_window._update_output_broker_controls()

        # Close button
        layout.addSpacing(10)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)


class StatisticsWorker(QThread):
    """Worker thread for calculating statistics to avoid blocking the GUI."""

    # Signal emitted when calculation is complete
    calculation_complete = pyqtSignal(list, int)  # (statistics, num_averaged_waveforms)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveforms_to_process = None
        self.rois = None
        self.scaling_functions = None
        self.statistics_calculator = None
        self._should_stop = False

    def setup_calculation(
        self,
        waveforms: List[WaveformData],
        rois: List[ROIDefinition],
        scaling_functions: Dict[str, Callable],
        statistics_calculator: ROIStatisticsCalculator,
    ):
        """Set up the data for the next calculation."""
        self.waveforms_to_process = waveforms
        self.rois = rois
        self.scaling_functions = scaling_functions
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
            # Average waveforms if more than one
            if len(self.waveforms_to_process) > 1:
                try:
                    averaged_waveform = self._average_waveforms(
                        self.waveforms_to_process
                    )
                except ValueError as e:
                    # Waveform dimensions changed - use only the last one
                    logging.warning(
                        f"Cannot average waveforms with different dimensions: {e}. Using last waveform only."
                    )
                    averaged_waveform = self.waveforms_to_process[-1]
            else:
                averaged_waveform = self.waveforms_to_process[0]

            if self._should_stop:
                return

            # Apply scaling
            scaled_channels = {}
            for ch_name, data in averaged_waveform.channels.items():
                if self.scaling_functions and ch_name in self.scaling_functions:
                    scaled_channels[ch_name] = self.scaling_functions[ch_name](data)
                else:
                    scaled_channels[ch_name] = data

            scaled_waveform = WaveformData.from_preprocessed(
                source=averaged_waveform,
                channels=scaled_channels,
            )

            if self._should_stop:
                return

            # Calculate statistics
            if not self.statistics_calculator:
                return

            statistics = self.statistics_calculator.calculate_all_statistics(
                scaled_waveform, self.rois
            )

            num_averaged = len(self.waveforms_to_process)

            # Emit result
            self.calculation_complete.emit(statistics, num_averaged)

        except Exception as e:
            logging.error(f"Error in statistics calculation thread: {e}", exc_info=True)

    def _average_waveforms(self, waveforms: List[WaveformData]) -> WaveformData:
        """Average multiple waveforms together.

        Args:
            waveforms: List of WaveformData objects to average

        Returns:
            Averaged WaveformData
        """
        if not waveforms:
            raise ValueError("Cannot average empty waveform list")

        if len(waveforms) == 1:
            return waveforms[0]

        # Use first waveform as template
        first = waveforms[0]

        # Validate that all waveforms have compatible dimensions
        for wf in waveforms[1:]:
            if self._should_stop:
                raise ValueError("Calculation stopped")

            # Check if channel sets match
            if set(wf.channels.keys()) != set(first.channels.keys()):
                raise ValueError("Cannot average waveforms with different channel sets")

            # Check if array shapes match for each channel
            for ch_name in first.channels.keys():
                if wf.channels[ch_name].shape != first.channels[ch_name].shape:
                    raise ValueError(
                        f"Cannot average waveforms: channel '{ch_name}' has "
                        f"incompatible shapes ({first.channels[ch_name].shape} vs {wf.channels[ch_name].shape})"
                    )

        # Average channels
        averaged_channels = {}
        for ch_name in first.channels.keys():
            if self._should_stop:
                raise ValueError("Calculation stopped")

            # Collect all channel data
            channel_data = []
            for wf in waveforms:
                if ch_name in wf.channels:
                    channel_data.append(wf.channels[ch_name])

            if channel_data:
                # Average across all waveforms
                averaged_channels[ch_name] = np.mean(channel_data, axis=0)

        # Use latest timestamp
        latest_timestamp = max(wf.timestamp for wf in waveforms)

        return WaveformData.from_preprocessed(
            source=first,
            channels=averaged_channels,
            timestamp=latest_timestamp,
        )


class ExportWorker(QThread):
    """Worker thread for exporting waveform data to avoid blocking the GUI."""

    # Signal emitted when export is complete (success, message)
    export_complete = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_path = None
        self.waveform = None
        self.export_type = "csv"  # or "payload"
        self._should_stop = False

    def setup_export(
        self, file_path: str, waveform: WaveformData, export_type: str = "csv"
    ):
        """Set up the export operation."""
        self.file_path = file_path
        self.waveform = waveform
        self.export_type = export_type
        self._should_stop = False

    def stop(self):
        """Signal the worker to stop processing."""
        self._should_stop = True

    def run(self):
        """Execute the export in the background."""
        if not self.waveform or not self.file_path:
            self.export_complete.emit(False, "No data or path specified")
            return

        try:
            if self.export_type == "csv":
                success = self._export_csv()
                if success:
                    self.export_complete.emit(
                        True, f"Waveform exported to {self.file_path}"
                    )
                else:
                    self.export_complete.emit(
                        False, "Export failed. Check logs for details."
                    )
            else:
                self.export_complete.emit(
                    False, f"Unknown export type: {self.export_type}"
                )
        except Exception as e:
            logging.error(f"Error in export thread: {e}", exc_info=True)
            self.export_complete.emit(False, f"Export error: {str(e)}")

    def _export_csv(self) -> bool:
        """Export waveform to CSV (adapted from exporter.py)."""
        import csv

        if not self.file_path or not self.waveform:
            return False

        try:
            with open(self.file_path, "w", newline="") as csvfile:
                # Get sorted channel names for consistent ordering
                channel_names = sorted(self.waveform.channels.keys())

                # Create header
                header = ["m/z"] + channel_names

                writer = csv.writer(csvfile)
                writer.writerow(header)

                # Write data rows in batches to allow cancellation
                batch_size = 10000
                mz = self.waveform.mz
                n_samples = len(mz)

                for batch_start in range(0, n_samples, batch_size):
                    if self._should_stop:
                        return False

                    batch_end = min(batch_start + batch_size, n_samples)
                    for i in range(batch_start, batch_end):
                        row = [mz[i]]
                        for ch_name in channel_names:
                            row.append(self.waveform.channels[ch_name][i])
                        writer.writerow(row)

            logging.info(f"Successfully exported waveform to {self.file_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to export waveform to {self.file_path}: {e}")
            return False


class PayloadWorker(QThread):
    """Worker thread for building MQTT payloads to avoid blocking the GUI."""

    # Signal emitted when payload is ready (payload_bytes)
    payload_ready = pyqtSignal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform = None
        self._should_stop = False

    def setup_build(self, waveform: WaveformData):
        """Set up the payload build operation."""
        self.waveform = waveform
        self._should_stop = False

    def stop(self):
        """Signal the worker to stop processing."""
        self._should_stop = True

    def run(self):
        """Execute the payload building in the background."""
        if not self.waveform:
            return

        try:
            # Import here to avoid circular imports
            from .exporter import build_payload

            if self._should_stop:
                return

            payload = build_payload(self.waveform)

            if self._should_stop:
                return

            self.payload_ready.emit(payload)

        except Exception as e:
            logging.error(f"Error in payload building thread: {e}", exc_info=True)


class PlotDataWorker(QThread):
    """Worker thread for preparing plot data (scaling) to avoid blocking the GUI."""

    # Signal emitted when preparation is complete
    # Emits: (x_data, scaled_channels_dict, timestamp)
    data_ready = pyqtSignal(object, dict, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform = None
        self.scaling_functions = None
        self.x_axis_mode = "mz"
        self._should_stop = False

    def setup_preparation(
        self,
        waveform: WaveformData,
        scaling_functions: Dict[str, Callable],
        x_axis_mode: str = "mz",
    ):
        """Set up the data for the next preparation."""
        self.waveform = waveform
        self.scaling_functions = scaling_functions
        self.x_axis_mode = x_axis_mode
        self._should_stop = False

    def stop(self):
        """Signal the worker to stop processing."""
        self._should_stop = True

    def run(self):
        """Execute the plot data preparation in the background."""
        if not self.waveform:
            return

        try:
            scaled_channels = {}

            # Apply scaling to each channel
            for ch_name, data in self.waveform.channels.items():
                if self._should_stop:
                    return

                if self.scaling_functions and ch_name in self.scaling_functions:
                    scaled_channels[ch_name] = self.scaling_functions[ch_name](data)
                else:
                    scaled_channels[ch_name] = data.copy()

            if self._should_stop:
                return

            # Get x-axis data based on mode
            if self.x_axis_mode == "frequency":
                x_data = self.waveform.f_AC_kHz
            elif self.x_axis_mode == "rf_amplitude":
                x_data = self.waveform.rf_amp_V0p
            else:  # default to mz
                x_data = self.waveform.mz

            # Emit result
            self.data_ready.emit(x_data, scaled_channels, self.waveform.timestamp)

        except Exception as e:
            logging.error(f"Error in plot data preparation thread: {e}", exc_info=True)


class MainWindow(QMainWindow):
    """
    Main application window with waveform plotting and controls.
    """

    def __init__(self, mqtt_client: MqttClient, state_path: str):
        """
        Initialize the main window.

        Args:
            mqtt_client: MqttClient instance
            state_path: Path to the state file
        """
        super().__init__()

        self.mqtt_client = mqtt_client
        self.mqtt_output_client: Optional[MqttClient] = (
            None  # Secondary broker for publishing
        )
        self.state_path = state_path
        self.current_waveform: Optional[WaveformData] = None
        self.original_waveform: Optional[WaveformData] = (
            None  # Store original unmodified waveform
        )
        self.plot_curves: Dict[str, pg.PlotDataItem] = {}
        self.regions: List[pg.LinearRegionItem] = []
        # Store m/z bounds for each region (key: region object, value: (start_mz, end_mz))
        self.region_mz_bounds: Dict[pg.LinearRegionItem, tuple] = {}
        # Track if trapping parameters were received
        self.trapping_params_received = False

        # Channel scaling parameters: {channel_name: {"gain": float, "offset": float}}
        self.channel_scaling: Dict[str, Dict[str, float]] = {}
        # UI controls for channel scaling: {channel_name: {"gain": QDoubleSpinBox, "offset": QDoubleSpinBox}}
        self.scaling_controls: Dict[str, Dict[str, QDoubleSpinBox]] = {}

        # Load state first (needed for statistics calculator)
        self.state = state_manager.load_state(state_path)

        # Excitation step selection for m/z calculation
        self.selected_excitation_step = self.state.get("preferences", {}).get(
            "selected_excitation_step", 0
        )
        self.excitation_step_spinbox: Optional[QSpinBox] = None

        # Excitation step selection for m/z calculation
        self.selected_excitation_step = self.state["preferences"].get(
            "selected_excitation_step", 0
        )
        self.excitation_step_spinbox: Optional[QSpinBox] = None
        # List of active step indices for the spinbox
        self._active_step_indices: List[int] = []

        # Statistics tracking
        stats_config = self.state.get("statistics", {})
        self.statistics_calculator = ROIStatisticsCalculator(
            plugin_dir=stats_config.get("plugin_directory"),
            enabled_plugins=stats_config.get("enabled_plugins", []),
            expression_stats=stats_config.get("expression_statistics", []),
            enabled_builtin_stats=stats_config.get("builtin_statistics"),
        )
        self.current_statistics: List[ROIStatistics] = []
        # Store historical statistics for plotting: {metric_name: {channel_roi_combo: [(timestamp, value)]}}
        self.statistics_history: Dict[str, Dict[str, List]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Performance optimization: throttle statistics updates
        self._waveform_counter = 0
        self._last_statistics_update_time = datetime.now()

        # Accumulate waveforms for averaging during throttle period
        self._accumulated_waveforms: List[WaveformData] = []
        self._num_averaged_waveforms = 0  # Track how many waveforms were averaged

        # Statistics calculation worker thread
        self._statistics_worker = StatisticsWorker(self)
        self._statistics_worker.calculation_complete.connect(
            self._on_statistics_calculated
        )
        self._statistics_calculation_pending = False

        # Plot data preparation worker thread
        self._plot_data_worker = PlotDataWorker(self)
        self._plot_data_worker.data_ready.connect(self._on_plot_data_ready)
        self._plot_data_pending = False
        self._pending_plot_waveform = None

        # Export worker thread (for CSV exports with large data)
        self._export_worker = ExportWorker(self)
        self._export_worker.export_complete.connect(self._on_export_complete)

        # Payload building worker thread (for republishing large waveforms)
        self._payload_worker = PayloadWorker(self)
        self._payload_worker.payload_ready.connect(self._on_payload_ready)
        self._pending_republish = False

        # Load configuration from state
        self.max_history_points = self.state["preferences"].get(
            "max_history_points", 100
        )
        # Statistics update throttling (for high-rate data transfers)
        self._statistics_update_interval = self.state["preferences"].get(
            "statistics_update_interval", 10
        )  # Update every N waveforms
        self._statistics_update_min_interval = (
            self.state["preferences"].get("statistics_update_min_interval_ms", 100)
            / 1000.0
        )  # Convert ms to seconds
        self.channel_colors = self.state["preferences"].get(
            "channel_colors",
            [
                [255, 0, 0],  # Red
                [0, 255, 0],  # Green
                [0, 0, 255],  # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
                [255, 128, 0],  # Orange
                [128, 0, 255],  # Purple
            ],
        )
        self.statistics_colors = self.state["preferences"].get(
            "statistics_colors",
            [
                [255, 0, 0],  # Red
                [0, 255, 0],  # Green
                [0, 0, 255],  # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
                [255, 128, 0],  # Orange
                [128, 0, 255],  # Purple
                [255, 128, 128],  # Light Red
                [128, 255, 128],  # Light Green
                [128, 128, 255],  # Light Blue
                [192, 192, 0],  # Olive
            ],
        )

        # Set up UI
        self._setup_ui()

        # Apply MQTT config from state
        self._apply_mqtt_config_from_state()

        # Set initial selected excitation step in MQTT client
        self.mqtt_client.set_selected_excitation_step(self.selected_excitation_step)

        # Connect MQTT signals
        self.mqtt_client.waveform_received.connect(self.on_waveform_received)
        self.mqtt_client.trapping_parameters_received.connect(
            self.on_trapping_parameters_received
        )
        self.mqtt_client.connection_state_changed.connect(
            self.on_connection_state_changed
        )
        self.mqtt_client.error_occurred.connect(self.on_error)

        # Initialize secondary MQTT client if configured
        if self.state["mqtt"].get("output_broker_enabled", False):
            self._init_output_mqtt_client()

        # Restore window geometry if available
        if self.state["gui"].get("window_geometry"):
            geom = self.state["gui"]["window_geometry"]
            self.setGeometry(geom["x"], geom["y"], geom["width"], geom["height"])

        # Restore regions
        self._restore_regions()

        # Restore splitter position
        if self.state["gui"].get("splitter_sizes"):
            self.plots_splitter.setSizes(self.state["gui"]["splitter_sizes"])

    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("MS LQIT")
        self.setMinimumSize(1200, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Top: Splitter with waveform plot and statistics plot side by side
        self._setup_plots()
        main_layout.addWidget(self.plots_splitter, stretch=10)

        # Bottom: All controls in one compact row
        controls_layout = QHBoxLayout()

        # MQTT config button (opens dialog) and X-axis selection
        mqtt_button_widget = QWidget()
        mqtt_button_layout = QVBoxLayout(mqtt_button_widget)
        mqtt_config_btn = QPushButton("MQTT Configuration...")
        mqtt_config_btn.clicked.connect(self._open_mqtt_config_dialog)
        mqtt_button_layout.addWidget(mqtt_config_btn)

        # X-axis selection radio buttons
        self._setup_xaxis_controls()
        mqtt_button_layout.addWidget(self.xaxis_group)

        mqtt_button_layout.addStretch()
        controls_layout.addWidget(mqtt_button_widget, stretch=1)

        # Channel scaling controls
        self._setup_scaling_controls()
        controls_layout.addWidget(self.scaling_group, stretch=2)

        # ROI table
        self._setup_region_table()
        controls_layout.addWidget(self.region_group, stretch=2)

        # Statistics table
        self._setup_statistics_table()
        controls_layout.addWidget(self.statistics_group, stretch=3)

        # Action buttons
        self._setup_action_buttons()
        controls_layout.addWidget(self.action_group, stretch=1)

        main_layout.addLayout(controls_layout, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Not connected")

    def _setup_plots(self):
        """Set up side-by-side plots with movable divider."""
        self.plots_splitter = QSplitter(QtCore.Orientation.Horizontal)

        # Left: Waveform plot
        self._setup_plot()
        self.plots_splitter.addWidget(self.plot_widget)

        # Right: Statistics plot
        self._setup_statistics_plot()
        self.plots_splitter.addWidget(self.stats_plot_widget)

        # Set initial sizes (60% waveform, 40% statistics)
        self.plots_splitter.setSizes([600, 400])
        self.plots_splitter.setStretchFactor(0, 3)
        self.plots_splitter.setStretchFactor(1, 2)

    def _setup_plot(self):
        """Set up the PyQtGraph plot widget."""
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("left", "Amplitude", units="mV")

        # Set x-axis label based on saved state
        x_axis_mode = self.state.get("gui", {}).get("x_axis_mode", "mz")
        if x_axis_mode == "frequency":
            self.plot_widget.setLabel("bottom", "f_AC", units="kHz")
        elif x_axis_mode == "rf_amplitude":
            self.plot_widget.setLabel("bottom", "RF Amplitude", units="V0p")
        else:
            self.plot_widget.setLabel("bottom", "m/z", units="Da")

        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self._update_plot_title()

    def _setup_statistics_plot(self):
        """Set up the statistics plot with PyQtGraph."""
        self.stats_plot_widget = QWidget()
        layout = QVBoxLayout(self.stats_plot_widget)

        # Control panel for selecting metrics
        control_panel = QGroupBox("Select Metrics to Display")
        control_layout = QVBoxLayout()

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.stats_checkbox_layout = QVBoxLayout(scroll_content)

        # Checkboxes for different statistics
        self.stats_checkboxes = {}
        self._refresh_statistics_checkboxes()

        self.stats_checkbox_layout.addStretch()
        scroll.setWidget(scroll_content)
        control_layout.addWidget(scroll)

        # Clear history button
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_statistics_history)
        control_layout.addWidget(clear_btn)

        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(250)

        # PyQtGraph plot widget
        self.stats_plot = pg.PlotWidget()
        self.stats_plot.setBackground("w")
        self.stats_plot.setLabel("left", "Value")
        self.stats_plot.setLabel("bottom", "Time")
        self.stats_plot.showGrid(x=True, y=True, alpha=0.3)
        self.stats_plot.addLegend()
        self.stats_plot.setTitle("Statistics Over Time")

        # Store plot curves for statistics
        self.stats_plot_curves = {}

        # Layout
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(control_panel)
        plot_layout.addWidget(self.stats_plot, stretch=1)
        layout.addLayout(plot_layout)

    def _setup_statistics_table(self):
        """Set up the statistics display table."""
        self.statistics_group = QGroupBox("ROI Statistics")
        layout = QVBoxLayout()

        # Table
        self.statistics_table = QTableWidget()
        self.statistics_table.setColumnCount(13)
        self.statistics_table.setHorizontalHeaderLabels(
            [
                "ROI",
                "Channel",
                "Mean",
                "Median",
                "Std Dev",
                "Min",
                "Max",
                "Peak-Peak",
                "Area",
                "Points",
                "m/z Span",
                "Start m/z",
                "End m/z",
            ]
        )
        header = self.statistics_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.statistics_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.statistics_table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout.addWidget(self.statistics_table)
        self.statistics_group.setLayout(layout)

    def _setup_region_table(self):
        """Set up the region management UI."""
        self.region_group = QGroupBox("Regions of Interest")
        layout = QVBoxLayout()

        # Table
        self.region_table = QTableWidget()
        self.region_table.setColumnCount(3)
        self.region_table.setHorizontalHeaderLabels(["Name", "Start m/z", "End m/z"])
        header = self.region_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.Stretch)
        self.region_table.setSelectionBehavior(QTableWidget.SelectRows)

        # Enable editing of table cells
        self.region_table.setEditTriggers(
            QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed
        )

        # Connect cell change signal to update regions
        self.region_table.cellChanged.connect(self._on_region_table_cell_changed)

        layout.addWidget(self.region_table)

        # Buttons
        btn_layout = QHBoxLayout()

        self.add_region_btn = QPushButton("Add Region")
        self.add_region_btn.clicked.connect(self.add_region)
        btn_layout.addWidget(self.add_region_btn)

        self.remove_region_btn = QPushButton("Remove Selected")
        self.remove_region_btn.clicked.connect(self.remove_selected_region)
        btn_layout.addWidget(self.remove_region_btn)

        layout.addLayout(btn_layout)
        self.region_group.setLayout(layout)

    def _open_mqtt_config_dialog(self):
        """Open the MQTT configuration dialog."""
        dialog = MqttConfigDialog(self)
        dialog.exec_()

    def _setup_xaxis_controls(self):
        """Set up x-axis selection controls."""
        self.xaxis_group = QGroupBox("X-Axis")
        layout = QVBoxLayout()

        # Create radio buttons
        self.xaxis_mz_radio = QRadioButton("m/z (Da)")
        self.xaxis_freq_radio = QRadioButton("f_AC (kHz)")
        self.xaxis_rf_radio = QRadioButton("RF Amp (V0p)")

        # Create button group
        self.xaxis_button_group = QButtonGroup()
        self.xaxis_button_group.addButton(self.xaxis_mz_radio, 0)
        self.xaxis_button_group.addButton(self.xaxis_freq_radio, 1)
        self.xaxis_button_group.addButton(self.xaxis_rf_radio, 2)

        # Set default from state
        x_axis_mode = self.state.get("gui", {}).get("x_axis_mode", "mz")
        if x_axis_mode == "frequency":
            self.xaxis_freq_radio.setChecked(True)
        elif x_axis_mode == "rf_amplitude":
            self.xaxis_rf_radio.setChecked(True)
        else:
            self.xaxis_mz_radio.setChecked(True)

        # Connect signal
        self.xaxis_button_group.buttonClicked.connect(self._on_xaxis_changed)

        # Add to layout
        layout.addWidget(self.xaxis_mz_radio)
        layout.addWidget(self.xaxis_freq_radio)
        layout.addWidget(self.xaxis_rf_radio)

        # Add excitation step selection
        excitation_layout = QHBoxLayout()
        excitation_layout.addWidget(QLabel("Excitation Step:"))
        self.excitation_step_spinbox = QSpinBox()
        self.excitation_step_spinbox.setMinimum(1)
        self.excitation_step_spinbox.setMaximum(
            1
        )  # Will be updated when trapping params are received
        self.excitation_step_spinbox.setValue(
            self.selected_excitation_step + 1
        )  # Display as 1-based
        self.excitation_step_spinbox.setToolTip(
            "Select which excitation step to use for m/z calculation"
        )
        self.excitation_step_spinbox.valueChanged.connect(
            self._on_excitation_step_changed
        )
        excitation_layout.addWidget(self.excitation_step_spinbox)
        layout.addLayout(excitation_layout)

        layout.addStretch()

        self.xaxis_group.setLayout(layout)

    def _on_xaxis_changed(self, button):
        """Handle x-axis mode change."""
        if button == self.xaxis_mz_radio:
            x_axis_mode = "mz"
            self.plot_widget.setLabel("bottom", "m/z", units="Da")
        elif button == self.xaxis_freq_radio:
            x_axis_mode = "frequency"
            self.plot_widget.setLabel("bottom", "f_AC", units="kHz")
        else:  # rf_amplitude
            x_axis_mode = "rf_amplitude"
            self.plot_widget.setLabel("bottom", "RF Amplitude", units="V0p")

        # Save to state
        if "gui" not in self.state:
            self.state["gui"] = {}
        self.state["gui"]["x_axis_mode"] = x_axis_mode
        state_manager.save_state(self.state_path, self.state)

        # Update region positions to match new x-axis
        if self.current_waveform:
            self._update_region_positions()

        # Update plot if we have data
        if self.current_waveform:
            self._update_plot(self.current_waveform)

    def _on_excitation_step_changed(self, value: int):
        """Handle excitation step selection change."""
        # Convert from 1-based display to 0-based internal index
        internal_value = value - 1

        # Validate that the selected step is active
        if (
            self._active_step_indices
            and internal_value not in self._active_step_indices
        ):
            logger.warning(f"Step {value} is not active, ignoring change")
            # Reset to current valid step
            if self.excitation_step_spinbox is not None:
                self.excitation_step_spinbox.blockSignals(True)
                self.excitation_step_spinbox.setValue(
                    self.selected_excitation_step + 1
                )  # Display as 1-based
                self.excitation_step_spinbox.blockSignals(False)
            return

        self.selected_excitation_step = internal_value

        # Save to state
        if "preferences" not in self.state:
            self.state["preferences"] = {}
        self.state["preferences"]["selected_excitation_step"] = internal_value
        state_manager.save_state(self.state_path, self.state)

        # Update MQTT clients to use new excitation step for future waveforms
        self.mqtt_client.set_selected_excitation_step(internal_value)
        if self.mqtt_output_client:
            self.mqtt_output_client.set_selected_excitation_step(internal_value)

        # Update plot title to show new step info (without recalculating waveform)
        self._update_plot_title()

    def _setup_action_buttons(self):
        """Set up action buttons panel."""
        self.action_group = QGroupBox("Actions")
        layout = QVBoxLayout()

        self.export_csv_btn = QPushButton("Export to CSV...")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setEnabled(False)
        layout.addWidget(self.export_csv_btn)

        self.republish_btn = QPushButton("Republish Waveform")
        self.republish_btn.clicked.connect(self.republish_waveform)
        self.republish_btn.setEnabled(False)
        layout.addWidget(self.republish_btn)

        self.reload_plugins_btn = QPushButton("Reload Plugins")
        self.reload_plugins_btn.clicked.connect(self._reload_plugins)
        self.reload_plugins_btn.setToolTip(
            "Reload custom statistic plugins from plugin directory"
        )
        layout.addWidget(self.reload_plugins_btn)

        layout.addStretch()
        self.action_group.setLayout(layout)

    def _setup_scaling_controls(self):
        """Set up channel scaling controls panel."""
        self.scaling_group = QGroupBox("Channel Scaling")
        layout = QVBoxLayout()

        # Label
        info_label = QLabel("y := Gain × (y - Offset)")
        info_label.setStyleSheet("font-style: italic;")
        layout.addWidget(info_label)

        # Scroll area for channel controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scaling_layout = QVBoxLayout(scroll_content)

        # Initially empty - will be populated when waveform is received
        self.scaling_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self.scaling_group.setLayout(layout)

    def _update_scaling_controls(self):
        """Update scaling controls based on current channels."""
        if self.current_waveform is None:
            return

        channel_names = sorted(self.current_waveform.channels.keys())

        # Remove controls for channels that no longer exist
        for ch_name in list(self.scaling_controls.keys()):
            if ch_name not in channel_names:
                # Remove from layout
                gain_widget = self.scaling_controls[ch_name]["gain"]
                offset_widget = self.scaling_controls[ch_name]["offset"]
                # Find and remove the parent widget containing this channel's controls
                for i in reversed(range(self.scaling_layout.count())):
                    item = self.scaling_layout.itemAt(i)
                    if item:
                        widget = item.widget()
                        if (
                            widget
                            and hasattr(widget, "objectName")
                            and widget.objectName() == f"scaling_{ch_name}"
                        ):
                            self.scaling_layout.removeWidget(widget)
                            widget.deleteLater()
                            break
                del self.scaling_controls[ch_name]

        # Add or update controls for each channel
        for i, ch_name in enumerate(channel_names):
            # Initialize scaling parameters if not present
            if ch_name not in self.channel_scaling:
                self.channel_scaling[ch_name] = {"gain": 1.0, "offset": 0.0}

            # Load from config if available
            if ch_name in self.state.get("channel_scaling", {}):
                config = self.state["channel_scaling"][ch_name]
                self.channel_scaling[ch_name]["gain"] = config.get("gain", 1.0)
                self.channel_scaling[ch_name]["offset"] = config.get("offset", 0.0)

            if ch_name not in self.scaling_controls:
                # Create new controls for this channel
                channel_widget = QWidget()
                channel_widget.setObjectName(f"scaling_{ch_name}")
                channel_layout = QVBoxLayout(channel_widget)
                channel_layout.setContentsMargins(2, 2, 2, 2)

                # Channel label
                color = tuple(self.channel_colors[i % len(self.channel_colors)])
                label = QLabel(f"<b>{ch_name}</b>")
                label.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]});")
                channel_layout.addWidget(label)

                # Gain control
                gain_layout = QHBoxLayout()
                gain_layout.addWidget(QLabel("Gain:"))
                gain_spin = QDoubleSpinBox()
                gain_spin.setRange(-1000.0, 1000.0)
                gain_spin.setDecimals(4)
                gain_spin.setSingleStep(0.1)
                gain_spin.setValue(self.channel_scaling[ch_name]["gain"])
                gain_spin.valueChanged.connect(
                    lambda value, ch=ch_name: self._on_scaling_changed(
                        ch, "gain", value
                    )
                )
                gain_layout.addWidget(gain_spin)
                channel_layout.addLayout(gain_layout)

                # Offset control
                offset_layout = QHBoxLayout()
                offset_layout.addWidget(QLabel("Offset:"))
                offset_spin = QDoubleSpinBox()
                offset_spin.setRange(-1000.0, 1000.0)
                offset_spin.setDecimals(4)
                offset_spin.setSingleStep(0.01)
                offset_spin.setValue(self.channel_scaling[ch_name]["offset"])
                offset_spin.valueChanged.connect(
                    lambda value, ch=ch_name: self._on_scaling_changed(
                        ch, "offset", value
                    )
                )
                offset_layout.addWidget(offset_spin)
                channel_layout.addLayout(offset_layout)

                # Store controls
                self.scaling_controls[ch_name] = {
                    "gain": gain_spin,
                    "offset": offset_spin,
                }

                # Add to layout (before the stretch)
                self.scaling_layout.insertWidget(
                    self.scaling_layout.count() - 1, channel_widget
                )
            else:
                # Update existing controls
                self.scaling_controls[ch_name]["gain"].setValue(
                    self.channel_scaling[ch_name]["gain"]
                )
                self.scaling_controls[ch_name]["offset"].setValue(
                    self.channel_scaling[ch_name]["offset"]
                )

    def _on_scaling_changed(self, channel: str, param: str, value: float):
        """Handle changes to scaling parameters."""
        if channel not in self.channel_scaling:
            self.channel_scaling[channel] = {"gain": 1.0, "offset": 0.0}

        self.channel_scaling[channel][param] = value

        # Replot with new scaling
        if self.current_waveform is not None:
            self._update_plot(self.current_waveform)
            self._calculate_and_display_statistics()

    def _apply_channel_scaling(self, channel_name: str, data: np.ndarray) -> np.ndarray:
        """
        Apply scaling to channel data.

        Args:
            channel_name: Name of the channel
            data: Original data array

        Returns:
            Scaled data array: y := Gain * (y - Offset)
        """
        if channel_name not in self.channel_scaling:
            return data

        gain = self.channel_scaling[channel_name]["gain"]
        offset = self.channel_scaling[channel_name]["offset"]

        return gain * (data - offset)

    def _get_excitation_step_info(self) -> str:
        """Get formatted excitation step information for plot title."""
        # Try to get trapping parameters from waveform or MQTT client
        if self.current_waveform is not None:
            trapping_params: TrapControls = self.current_waveform.trapping_parameters
        elif self.mqtt_client.trapping_parameters is not None:
            trapping_params: TrapControls = self.mqtt_client.trapping_parameters
        else:
            return ""

        # Check if there are any active steps
        active_steps = [
            i
            for i, step in enumerate(trapping_params.excitation_steps)
            if step.step_active
        ]

        if not active_steps:
            return "⚠ No active excitation steps"

        step_index = self.selected_excitation_step
        excitation_steps = trapping_params.excitation_steps

        if step_index < 0 or step_index >= len(excitation_steps):
            return ""

        step = excitation_steps[step_index]

        # Check if current step is active
        if not step.step_active:
            return f"⚠ Step {step_index + 1} is inactive"

        # Build step info string with step number (display as 1-based)
        display_index = step_index + 1
        if step.step_name:
            step_label = f"Step {display_index} ({step.step_name})"
        else:
            step_label = f"Step {display_index}"

        mz_range = trapping_params.calculate_step_mz_range(step_index)

        if mz_range is not None:
            if mz_range[0] == mz_range[1]:
                return f"{step_label}: m/z {mz_range[0]:.1f} Da"
            else:
                return f"{step_label}: m/z {mz_range[0]:.1f} → {mz_range[1]:.1f} Da"

        return ""

    def _update_plot_title(self, timestamp_str: Optional[str] = None):
        """Update the plot title with excitation step info and optional timestamp.

        Args:
            timestamp_str: Optional timestamp string to display. If None, shows "No data received yet".
        """
        step_info = self._get_excitation_step_info()

        if timestamp_str:
            if step_info:
                title = f"{step_info} | {timestamp_str}"
            else:
                title = timestamp_str
        else:
            if step_info:
                title = f"{step_info} | No data received yet"
            else:
                title = "No data received yet"

        self.plot_widget.setTitle(title)

    

    def _apply_mqtt_config_from_state(self):
        """Apply MQTT configuration from loaded state."""
        mqtt_config = self.state["mqtt"]
        self.mqtt_client.set_config(
            broker_host=mqtt_config["broker_host"],
            broker_port=mqtt_config["broker_port"],
            input_topic=mqtt_config["input_topic"],
            trapping_parameters_topic=mqtt_config["trapping_parameters_topic"],
            collector_parameters_topic=mqtt_config.get(
                "collector_parameters_topic", "lqt/collector/parameters"
            ),
            output_topic=mqtt_config["output_topic"],
            client_id=mqtt_config.get("client_id", "waveform_viewer"),
            keepalive=mqtt_config.get("keepalive", 60),
        )

    def _mz_to_xaxis(self, start_mz: float, end_mz: float) -> tuple:
        """Convert m/z bounds to current x-axis units."""
        if not self.current_waveform:
            # No waveform data, return m/z as-is
            return start_mz, end_mz

        x_axis_mode = self.state.get("gui", {}).get("x_axis_mode", "mz")

        if x_axis_mode == "mz":
            return start_mz, end_mz

        # Use interpolation to convert m/z to x-axis values
        # This handles inverse relationships correctly (e.g., f_AC decreases as m/z increases)
        mz = self.current_waveform.mz

        if x_axis_mode == "frequency":
            x_data = self.current_waveform.f_AC_kHz
        else:  # rf_amplitude
            x_data = self.current_waveform.rf_amp_V0p

        # Interpolate to find x-axis values corresponding to m/z bounds
        start_x = float(np.interp(start_mz, mz, x_data))
        end_x = float(np.interp(end_mz, mz, x_data))

        return start_x, end_x

    def _xaxis_to_mz(self, start_x: float, end_x: float) -> tuple:
        """Convert current x-axis bounds to m/z values."""
        if not self.current_waveform:
            # No waveform data, assume it's already m/z
            return start_x, end_x

        x_axis_mode = self.state.get("gui", {}).get("x_axis_mode", "mz")

        if x_axis_mode == "mz":
            return start_x, end_x

        # Get appropriate x-axis array
        if x_axis_mode == "frequency":
            x_data = self.current_waveform.f_AC_kHz
        else:  # rf_amplitude
            x_data = self.current_waveform.rf_amp_V0p

        mz = self.current_waveform.mz

        # Check if x_data is monotonically increasing or decreasing
        # For interp to work, x-values must be monotonically increasing
        if len(x_data) > 1 and x_data[0] > x_data[-1]:
            # x_data is decreasing (e.g., f_AC decreases as m/z increases)
            # Reverse both arrays for interpolation
            x_data_sorted = x_data[::-1]
            mz_sorted = mz[::-1]
            start_mz = float(np.interp(start_x, x_data_sorted, mz_sorted))
            end_mz = float(np.interp(end_x, x_data_sorted, mz_sorted))
        else:
            # x_data is increasing (normal case)
            start_mz = float(np.interp(start_x, x_data, mz))
            end_mz = float(np.interp(end_x, x_data, mz))

        # Ensure m/z values are in correct order (start < end)
        if start_mz > end_mz:
            start_mz, end_mz = end_mz, start_mz

        return start_mz, end_mz

    def _update_region_positions(self):
        """Update all region positions when x-axis mode changes."""
        for region in self.regions:
            if region in self.region_mz_bounds:
                start_mz, end_mz = self.region_mz_bounds[region]
                start_x, end_x = self._mz_to_xaxis(start_mz, end_mz)
                # Block signals to prevent triggering _update_region_table during programmatic update
                region.blockSignals(True)
                region.setRegion([start_x, end_x])
                region.blockSignals(False)

    def _restore_regions(self):
        """Restore regions from saved state."""
        regions_data = self.state["gui"].get("regions", [])
        for region_data in regions_data:
            self._add_region_visual(
                region_data["start"],
                region_data["end"],
                region_data.get("name", "Region"),
            )

    def _add_region_visual(self, start_mz: float, end_mz: float, name: str = "Region"):
        """
        Add a visual region to the plot.

        Args:
            start_mz: Start m/z
            end_mz: End m/z
            name: Region name
        """
        # Store m/z bounds
        # Convert m/z to current x-axis units for display
        start_x, end_x = self._mz_to_xaxis(start_mz, end_mz)

        region = pg.LinearRegionItem(values=[start_x, end_x], brush=(100, 100, 255, 50))
        self.plot_widget.addItem(region)
        self.regions.append(region)
        self.region_mz_bounds[region] = (start_mz, end_mz)

        # Add to table (block signals to avoid triggering cellChanged)
        row = self.region_table.rowCount()
        self.region_table.blockSignals(True)
        self.region_table.insertRow(row)
        self.region_table.setItem(row, 0, QTableWidgetItem(name))
        self.region_table.setItem(row, 1, QTableWidgetItem(f"{start_mz:.6f}"))
        self.region_table.setItem(row, 2, QTableWidgetItem(f"{end_mz:.6f}"))
        self.region_table.blockSignals(False)

        # Connect region change to table update
        region.sigRegionChanged.connect(lambda: self._update_region_table(region, row))

    def _update_region_table(self, region: pg.LinearRegionItem, row: int):
        """Update table when region is moved."""
        region_values = region.getRegion()
        start_x = float(region_values[0])  # type: ignore[arg-type]
        end_x = float(region_values[1])  # type: ignore[arg-type]

        # Convert x-axis position back to m/z
        start_mz, end_mz = self._xaxis_to_mz(start_x, end_x)

        # Update stored m/z bounds
        self.region_mz_bounds[region] = (start_mz, end_mz)

        # Block signals to avoid triggering cellChanged
        self.region_table.blockSignals(True)
        self.region_table.setItem(row, 1, QTableWidgetItem(f"{start_mz:.6f}"))
        self.region_table.setItem(row, 2, QTableWidgetItem(f"{end_mz:.6f}"))
        self.region_table.blockSignals(False)

        # Recalculate statistics when region is moved (throttled)
        self._calculate_and_display_statistics_throttled()

    def _on_region_table_cell_changed(self, row: int, column: int):
        """
        Handle changes to table cells - update region position when user edits values.

        Args:
            row: Row that was changed
            column: Column that was changed (0=name, 1=start, 2=end)
        """
        if row >= len(self.regions):
            return

        region = self.regions[row]

        # Only update if start or end m/z was changed
        if column == 1 or column == 2:
            try:
                # Get current values from table
                start_item = self.region_table.item(row, 1)
                end_item = self.region_table.item(row, 2)

                if start_item is None or end_item is None:
                    return

                # Parse values
                start = float(start_item.text())
                end = float(end_item.text())

                # Validate
                if start >= end:
                    raise ValueError("Start m/z must be less than end m/z")

                # Store m/z bounds
                self.region_mz_bounds[region] = (start, end)

                # Convert m/z to current x-axis units for visual update
                start_x, end_x = self._mz_to_xaxis(start, end)

                # Update the region (will trigger _update_region_table, but signals are blocked)
                region.setRegion([start_x, end_x])

            except ValueError as e:
                # Restore previous values if parsing fails
                current_start, current_end = region.getRegion()
                self.region_table.blockSignals(True)
                self.region_table.setItem(
                    row, 1, QTableWidgetItem(f"{current_start:.6f}")
                )
                self.region_table.setItem(
                    row, 2, QTableWidgetItem(f"{current_end:.6f}")
                )
                self.region_table.blockSignals(False)

                QMessageBox.warning(
                    self,
                    "Invalid Value",
                    f"Invalid region m/z: {e}",
                )

    # Slots for user actions

    def apply_mqtt_config(self):
        """Apply MQTT configuration and connect."""
        broker_host = self.broker_host_input.text().strip()
        broker_port = self.broker_port_input.value()
        input_topic = self.input_topic_input.text().strip()
        trapping_parameters_topic = self.trapping_parameters_topic_input.text().strip()
        collector_parameters_topic = (
            self.collector_parameters_topic_input.text().strip()
        )
        output_topic = self.output_topic_input.text().strip()
        statistics_topic = self.stats_topic_input.text().strip()

        if (
            not broker_host
            or not input_topic
            or not trapping_parameters_topic
            or not collector_parameters_topic
            or not output_topic
            or not statistics_topic
        ):
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Please fill in all MQTT configuration fields.",
            )
            return

        # Update state
        self.state["mqtt"]["broker_host"] = broker_host
        self.state["mqtt"]["broker_port"] = broker_port
        self.state["mqtt"]["input_topic"] = input_topic
        self.state["mqtt"]["trapping_parameters_topic"] = trapping_parameters_topic
        self.state["mqtt"]["collector_parameters_topic"] = collector_parameters_topic
        self.state["mqtt"]["output_topic"] = output_topic
        self.state["mqtt"]["statistics_topic"] = statistics_topic

        # Apply to MQTT client
        self.mqtt_client.set_config(
            broker_host,
            broker_port,
            input_topic,
            trapping_parameters_topic,
            collector_parameters_topic,
            output_topic,
        )

        # Connect
        self.mqtt_client.connect()

        self.apply_mqtt_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)

    def disconnect_mqtt(self):
        """Disconnect from MQTT broker."""
        self.mqtt_client.disconnect_from_broker()
        self.apply_mqtt_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

    def _on_output_broker_enabled_changed(self, state):
        """Handle output broker enable checkbox state change."""
        self.state["mqtt"]["output_broker_enabled"] = bool(state)
        self._update_output_broker_controls()

    def _update_output_broker_controls(self):
        """Enable/disable output broker controls based on checkbox state."""
        enabled = self.output_broker_enabled.isChecked()
        self.output_broker_host_input.setEnabled(enabled)
        self.output_broker_port_input.setEnabled(enabled)
        self.output_broker_output_topic_input.setEnabled(enabled)
        self.output_broker_stats_topic_input.setEnabled(enabled)
        self.apply_output_mqtt_btn.setEnabled(enabled)

        # Disconnect button only enabled if connected
        if self.mqtt_output_client and self.mqtt_output_client.is_connected():
            self.disconnect_output_btn.setEnabled(enabled)
        else:
            self.disconnect_output_btn.setEnabled(False)

    def _init_output_mqtt_client(self):
        """Initialize the secondary MQTT client for output/statistics publishing."""
        if self.mqtt_output_client is not None:
            logger.warning("Output MQTT client already exists")
            return

        self.mqtt_output_client = MqttClient(self)
        self.mqtt_output_client.connection_state_changed.connect(
            self.on_output_connection_state_changed
        )
        self.mqtt_output_client.error_occurred.connect(self.on_output_error)
        logger.info("Secondary MQTT client initialized")

    def apply_output_mqtt_config(self):
        """Apply output MQTT configuration and connect."""
        if not self.output_broker_enabled.isChecked():
            QMessageBox.warning(
                self,
                "Not Enabled",
                "Please enable the secondary broker first.",
            )
            return

        output_broker_host = self.output_broker_host_input.text().strip()
        output_broker_port = self.output_broker_port_input.value()
        output_topic = self.output_broker_output_topic_input.text().strip()
        stats_topic = self.output_broker_stats_topic_input.text().strip()

        if not output_broker_host or not output_topic or not stats_topic:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Please fill in all output broker configuration fields.",
            )
            return

        # Update state
        self.state["mqtt"]["output_broker_host"] = output_broker_host
        self.state["mqtt"]["output_broker_port"] = output_broker_port
        self.state["mqtt"]["output_broker_output_topic"] = output_topic
        self.state["mqtt"]["output_broker_statistics_topic"] = stats_topic

        # Create client if needed
        if self.mqtt_output_client is None:
            self._init_output_mqtt_client()

        assert self.mqtt_output_client is not None  # Type hint for checker

        # Configure and connect
        self.mqtt_output_client.set_config(
            broker_host=output_broker_host,
            broker_port=output_broker_port,
            input_topic="",  # Not used for output-only client
            trapping_parameters_topic="",  # Not used for output-only client
            collector_parameters_topic="",  # Not used for output-only client
            output_topic=output_topic,
            client_id="waveform_viewer_output",
            keepalive=60,
        )

        self.mqtt_output_client.connect()

        self.apply_output_mqtt_btn.setEnabled(False)
        self.disconnect_output_btn.setEnabled(True)

    def disconnect_output_mqtt(self):
        """Disconnect from output MQTT broker."""
        if self.mqtt_output_client:
            self.mqtt_output_client.disconnect_from_broker()
        self.apply_output_mqtt_btn.setEnabled(True)
        self.disconnect_output_btn.setEnabled(False)

    def add_region(self):
        """Add a new region of interest."""
        if self.current_waveform is None:
            QMessageBox.information(
                self, "No Data", "Please wait for waveform data before adding regions."
            )
            return

        # Add region in the middle of the m/z range
        mz = self.current_waveform.mz
        mid = (mz[0] + mz[-1]) / 2
        width = (mz[-1] - mz[0]) / 10

        self._add_region_visual(
            mid - width / 2, mid + width / 2, f"Region {len(self.regions) + 1}"
        )

        # Recalculate statistics with new region
        self._calculate_and_display_statistics()

    def remove_selected_region(self):
        """Remove the selected region."""
        selection_model = self.region_table.selectionModel()
        if selection_model is None:
            return
        selected_rows = selection_model.selectedRows()
        if not selected_rows:
            return

        # Remove in reverse order to avoid index issues
        for row_index in sorted([r.row() for r in selected_rows], reverse=True):
            if row_index < len(self.regions):
                # Remove from plot
                region = self.regions[row_index]
                self.plot_widget.removeItem(region)

                # Remove from m/z bounds dictionary
                if region in self.region_mz_bounds:
                    del self.region_mz_bounds[region]

                # Remove from list and table
                del self.regions[row_index]
                self.region_table.removeRow(row_index)

        # Recalculate statistics after removing region(s)
        self._calculate_and_display_statistics()

    def export_csv(self):
        """Export current waveform to CSV."""
        if self.current_waveform is None:
            QMessageBox.warning(self, "No Data", "No waveform data to export.")
            return

        # Get save path
        last_dir = self.state["preferences"].get(
            "last_export_dir", str(state_manager.Path.home())
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Waveform to CSV", last_dir, "CSV Files (*.csv)"
        )

        if file_path:
            # Check if we should use threaded export for large waveforms
            max_samples = max(
                len(data) for data in self.current_waveform.channels.values()
            )
            use_threaded = max_samples > 100000  # Use threading for >100k points

            if use_threaded:
                # Use worker thread for large exports
                if not self._export_worker.isRunning():
                    self.status_bar.showMessage(
                        "Exporting large waveform in background...", 0
                    )
                    self._export_worker.setup_export(
                        file_path, self.current_waveform, "csv"
                    )
                    self._export_worker.start()
                    # Store path for later
                    self._pending_export_path = file_path
                else:
                    QMessageBox.warning(
                        self,
                        "Export in Progress",
                        "Another export is already in progress. Please wait.",
                    )
            else:
                # Small waveform - export immediately
                if export_waveform_to_csv(file_path, self.current_waveform):
                    QMessageBox.information(
                        self, "Export Successful", f"Waveform exported to {file_path}"
                    )

                    # Update last directory
                    self.state["preferences"]["last_export_dir"] = str(
                        state_manager.Path(file_path).parent
                    )
                else:
                    QMessageBox.critical(
                        self,
                        "Export Failed",
                        "Failed to export waveform. Check logs for details.",
                    )

    def republish_waveform(self):
        """Republish current waveform to MQTT."""
        if self.current_waveform is None:
            QMessageBox.warning(self, "No Data", "No waveform data to republish.")
            return

        # Determine which client to use
        use_output_broker = (
            self.state["mqtt"].get("output_broker_enabled", False)
            and self.mqtt_output_client is not None
            and self.mqtt_output_client.is_connected()
        )

        client_to_use = (
            self.mqtt_output_client if use_output_broker else self.mqtt_client
        )
        topic_to_use = (
            self.state["mqtt"].get(
                "output_broker_output_topic", "oscilloscope/republish"
            )
            if use_output_broker
            else self.mqtt_client.output_topic
        )

        if client_to_use is None or not client_to_use.is_connected():
            broker_name = "output broker" if use_output_broker else "MQTT broker"
            QMessageBox.warning(
                self, "Not Connected", f"Not connected to {broker_name}."
            )
            return

        # Check if we should use threaded payload building for large waveforms
        max_samples = max(len(data) for data in self.current_waveform.channels.values())
        use_threaded = max_samples > 100000  # Use threading for >100k points

        if use_threaded:
            # Use worker thread for large payloads
            if not self._payload_worker.isRunning():
                self.status_bar.showMessage("Building payload in background...", 0)
                self._payload_worker.setup_build(self.current_waveform)
                self._payload_worker.start()
                self._pending_republish = True
                self._republish_client = client_to_use
                self._republish_topic = topic_to_use
            else:
                QMessageBox.warning(
                    self,
                    "Build in Progress",
                    "Payload building is already in progress. Please wait.",
                )
        else:
            # Small waveform - build and publish immediately
            payload = build_payload(self.current_waveform)
            if client_to_use is not None:  # Type guard
                client_to_use.publish(topic_to_use, payload)
                broker_name = "output broker" if use_output_broker else "primary broker"
                self.status_bar.showMessage(
                    f"Waveform republished to {broker_name}", 3000
                )

    def _calculate_and_display_statistics(self):
        """Calculate statistics for all ROIs and update displays (in background thread)."""
        if self.current_waveform is None or len(self.regions) == 0:
            self.current_statistics = []
            self._update_statistics_table()
            self._accumulated_waveforms.clear()
            return

        # Skip if a calculation is already in progress
        if self._statistics_worker.isRunning():
            self._statistics_calculation_pending = True
            return

        # Build ROI definitions from current regions using stored m/z bounds
        rois = []
        for i, region in enumerate(self.regions):
            # Use stored m/z bounds instead of visual position
            if region in self.region_mz_bounds:
                start, end = self.region_mz_bounds[region]
            else:
                # Fallback to visual position (convert to m/z if needed)
                region_values = region.getRegion()
                start_x = float(region_values[0])  # type: ignore[arg-type]
                end_x = float(region_values[1])  # type: ignore[arg-type]
                start, end = self._xaxis_to_mz(start_x, end_x)

            # Get name from table
            name = f"Region {i+1}"
            if i < self.region_table.rowCount():
                item = self.region_table.item(i, 0)
                if item is not None:
                    name = item.text()

            rois.append(ROIDefinition(name=name, start_mz=start, end_mz=end))

        # Build scaling functions dictionary with captured scaling parameters (thread-safe)
        scaling_functions = {}
        for ch_name in self.current_waveform.channels.keys():
            if ch_name in self.channel_scaling:
                # Capture scaling parameters in the lambda closure
                gain = self.channel_scaling[ch_name]["gain"]
                offset = self.channel_scaling[ch_name]["offset"]
                scaling_functions[ch_name] = lambda data, g=gain, o=offset: g * (
                    data - o
                )
            else:
                # No scaling - return data as-is
                scaling_functions[ch_name] = lambda data: data

        # Copy accumulated waveforms for thread safety
        waveforms_copy = self._accumulated_waveforms.copy()

        # Clear accumulated waveforms now (don't wait for thread)
        self._accumulated_waveforms.clear()

        # Start calculation in background thread
        self._statistics_worker.setup_calculation(
            waveforms_copy, rois, scaling_functions, self.statistics_calculator
        )
        self._statistics_worker.start()

    def _on_statistics_calculated(
        self, statistics: List[ROIStatistics], num_averaged: int
    ):
        """Handle completion of statistics calculation from worker thread."""
        # Update results
        self.current_statistics = statistics
        self._num_averaged_waveforms = num_averaged

        # Update displays
        self._update_statistics_table()
        self._update_statistics_history()
        self._update_statistics_plot()

        # Publish statistics to MQTT
        self._publish_statistics()

        # If another calculation is pending, trigger it
        if self._statistics_calculation_pending:
            self._statistics_calculation_pending = False
            if len(self.regions) > 0 and self.current_waveform is not None:
                # Re-accumulate current waveform and trigger calculation
                self._accumulated_waveforms.append(self.current_waveform)
                self._calculate_and_display_statistics()

    def _on_plot_data_ready(
        self,
        mz: np.ndarray,
        scaled_channels: Dict[str, np.ndarray],
        timestamp: datetime,
    ):
        """Handle completion of plot data preparation from worker thread."""
        # Update plot with prepared data
        self._update_plot_with_scaled_data(mz, scaled_channels, timestamp)

        # If another preparation is pending, trigger it
        if self._plot_data_pending and self._pending_plot_waveform is not None:
            self._plot_data_pending = False
            waveform = self._pending_plot_waveform
            self._pending_plot_waveform = None

            # Build scaling functions
            scaling_functions = {}
            for ch_name in waveform.channels.keys():
                if ch_name in self.channel_scaling:
                    gain = self.channel_scaling[ch_name]["gain"]
                    offset = self.channel_scaling[ch_name]["offset"]
                    scaling_functions[ch_name] = lambda data, g=gain, o=offset: g * (
                        data - o
                    )
                else:
                    scaling_functions[ch_name] = lambda data: data.copy()

            x_axis_mode = self.state.get("gui", {}).get("x_axis_mode", "mz")
            self._plot_data_worker.setup_preparation(
                waveform, scaling_functions, x_axis_mode
            )
            self._plot_data_worker.start()

    def _on_export_complete(self, success: bool, message: str):
        """Handle completion of export from worker thread."""
        self.status_bar.clearMessage()

        if success:
            QMessageBox.information(self, "Export Successful", message)
            # Update last directory
            if hasattr(self, "_pending_export_path"):
                self.state["preferences"]["last_export_dir"] = str(
                    state_manager.Path(self._pending_export_path).parent
                )
                delattr(self, "_pending_export_path")
        else:
            QMessageBox.critical(self, "Export Failed", message)

    def _on_payload_ready(self, payload: bytes):
        """Handle completion of payload building from worker thread."""
        if self._pending_republish:
            self._pending_republish = False
            # Use stored client and topic from republish_waveform
            client = getattr(self, "_republish_client", self.mqtt_client)
            topic = getattr(self, "_republish_topic", self.mqtt_client.output_topic)

            # Publish the payload
            client.publish(topic, payload)

            # Determine which broker was used for status message
            use_output = client == self.mqtt_output_client
            broker_name = "output broker" if use_output else "primary broker"
            self.status_bar.showMessage(f"Waveform republished to {broker_name}", 3000)

    def _calculate_and_display_statistics_throttled(self):
        """Throttled version of statistics calculation for interactive updates."""
        now = datetime.now()
        time_since_last = (now - self._last_statistics_update_time).total_seconds()

        # Only update if enough time has passed
        if time_since_last >= self._statistics_update_min_interval:
            self._calculate_and_display_statistics()
            self._last_statistics_update_time = now

    def _update_statistics_table(self):
        """Update the statistics table with current statistics."""
        # Get all custom metric names across all statistics
        custom_metric_names = set()
        for stats in self.current_statistics:
            custom_metric_names.update(stats.custom_metrics.keys())
        custom_metric_names = sorted(custom_metric_names)

        # Update table columns if custom metrics changed
        base_columns = [
            "ROI",
            "Channel",
            "Mean",
            "Median",
            "Std Dev",
            "Min",
            "Max",
            "Peak-Peak",
            "Area",
            "Points",
            "m/z Span",
            "Start m/z",
            "End m/z",
        ]
        all_columns = base_columns + custom_metric_names

        if self.statistics_table.columnCount() != len(all_columns):
            self.statistics_table.setColumnCount(len(all_columns))
            self.statistics_table.setHorizontalHeaderLabels(all_columns)
            header = self.statistics_table.horizontalHeader()
            if header is not None:
                header.setSectionResizeMode(QHeaderView.ResizeToContents)

        # Set row count to match statistics
        required_rows = len(self.current_statistics)
        self.statistics_table.setRowCount(required_rows)

        for row_idx, stats in enumerate(self.current_statistics):
            col = 0
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(stats.roi_name)
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(stats.channel_name)
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.mean:.6f}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.median:.6f}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.std_dev:.6f}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.min_value:.6f}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.max_value:.6f}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.peak_to_peak:.6f}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.area_under_curve:.6e}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(str(stats.n_points))
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.mz_span:.6f}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.start_mz:.6f}")
            )
            col += 1
            self.statistics_table.setItem(
                row_idx, col, QTableWidgetItem(f"{stats.end_mz:.6f}")
            )
            col += 1

            # Add custom metrics
            for metric_name in custom_metric_names:
                value = stats.custom_metrics.get(metric_name, float("nan"))
                if np.isfinite(value):
                    self.statistics_table.setItem(
                        row_idx, col, QTableWidgetItem(f"{value:.6e}")
                    )
                else:
                    self.statistics_table.setItem(row_idx, col, QTableWidgetItem("N/A"))
                col += 1

    def _update_statistics_history(self):
        """Add current statistics to history for plotting over time."""
        if not self.current_statistics:
            return

        timestamp = (
            self.current_waveform.timestamp if self.current_waveform else datetime.now()
        )

        for stats in self.current_statistics:
            # Create a unique key for this channel/ROI combination
            key = f"{stats.channel_name}:{stats.roi_name}"

            # Mapping from metric names (as returned by get_all_metric_names) to attribute names
            metric_to_attr = {
                "mean": "mean",
                "median": "median",
                "std_dev": "std_dev",
                "min": "min_value",
                "max": "max_value",
                "peak_to_peak": "peak_to_peak",
                "area": "area_under_curve",
                "n_points": "n_points",
                "mz_span": "mz_span",
                "start_mz": "start_mz",
                "end_mz": "end_mz",
            }

            # Add data points for each built-in statistic using metric names as keys
            for metric_name, attr_name in metric_to_attr.items():
                value = getattr(stats, attr_name)
                self.statistics_history[metric_name][key].append((timestamp, value))

                # Limit history size
                if (
                    len(self.statistics_history[metric_name][key])
                    > self.max_history_points
                ):
                    self.statistics_history[metric_name][key].pop(0)

            # Add data points for custom metrics
            for metric_name, value in stats.custom_metrics.items():
                self.statistics_history[metric_name][key].append((timestamp, value))

                # Limit history size
                if (
                    len(self.statistics_history[metric_name][key])
                    > self.max_history_points
                ):
                    self.statistics_history[metric_name][key].pop(0)

    def _update_statistics_plot(self):
        """Update the statistics plot based on selected metrics."""
        # Determine which curves should be visible
        desired_curves = set()
        for stat_name, checkbox in self.stats_checkboxes.items():
            if checkbox.isChecked() and stat_name in self.statistics_history:
                for key in self.statistics_history[stat_name].keys():
                    curve_key = f"{stat_name}:{key}"
                    desired_curves.add(curve_key)

        # Remove curves that should no longer be visible
        for curve_key in list(self.stats_plot_curves.keys()):
            if curve_key not in desired_curves:
                self.stats_plot.removeItem(self.stats_plot_curves[curve_key])
                del self.stats_plot_curves[curve_key]

        # Update or create curves
        color_idx = 0
        has_data = False
        for stat_name, checkbox in self.stats_checkboxes.items():
            if checkbox.isChecked() and stat_name in self.statistics_history:
                # Plot each channel/ROI combination
                for key, data_points in self.statistics_history[stat_name].items():
                    if data_points:
                        has_data = True
                        # Convert timestamps to seconds since first timestamp for plotting
                        first_time = data_points[0][0]
                        times = [
                            (dp[0] - first_time).total_seconds() for dp in data_points
                        ]
                        values = [dp[1] for dp in data_points]

                        curve_key = f"{stat_name}:{key}"

                        # Update existing curve or create new one
                        if curve_key in self.stats_plot_curves:
                            # Update existing curve data (fast)
                            curve = self.stats_plot_curves[curve_key]
                            curve.setData(times, values)
                        else:
                            # Create new curve
                            label = f"{key} - {stat_name.replace('_', ' ').title()}"
                            color = tuple(
                                self.statistics_colors[
                                    color_idx % len(self.statistics_colors)
                                ]
                            )
                            pen = pg.mkPen(color=color, width=2)

                            curve = self.stats_plot.plot(
                                times,
                                values,
                                pen=pen,
                                symbol="o",
                                symbolSize=6,
                                symbolBrush=color,
                                name=label,
                            )
                            self.stats_plot_curves[curve_key] = curve

                        color_idx += 1

        if not has_data:
            self.stats_plot.setTitle("Statistics Over Time (No data yet)")
        else:
            # Show number of averaged waveforms in title
            if self._num_averaged_waveforms > 1:
                self.stats_plot.setTitle(
                    f"Statistics Over Time (averaged from {self._num_averaged_waveforms} waveforms)"
                )
            else:
                self.stats_plot.setTitle("Statistics Over Time")

    def _refresh_statistics_checkboxes(self):
        """Refresh the statistics metric checkboxes with built-in and custom metrics."""
        # Get all available metrics from calculator
        all_metrics = self.statistics_calculator.get_all_metric_names()

        # Remove checkboxes for metrics that no longer exist
        for metric_name in list(self.stats_checkboxes.keys()):
            if metric_name not in all_metrics:
                checkbox = self.stats_checkboxes.pop(metric_name)
                self.stats_checkbox_layout.removeWidget(checkbox)
                checkbox.deleteLater()

        # Add checkboxes for new metrics
        # Get saved selected metrics from state
        selected_metrics = self.state["gui"].get("selected_metrics", ["mean"])

        for metric_name in all_metrics:
            if metric_name not in self.stats_checkboxes:
                checkbox = QCheckBox(metric_name.replace("_", " ").title())
                # Restore checkbox state from saved selection
                checkbox.setChecked(metric_name in selected_metrics)
                checkbox.stateChanged.connect(self._update_statistics_plot)
                self.stats_checkboxes[metric_name] = checkbox

                # Insert before the stretch at the end
                count = self.stats_checkbox_layout.count()
                self.stats_checkbox_layout.insertWidget(count - 1, checkbox)

    def _clear_statistics_history(self):
        """Clear all statistics history."""
        self.statistics_history.clear()
        self._update_statistics_plot()

    def _reload_plugins(self):
        """Reload custom statistic plugins."""
        stats_config = self.state.get("statistics", {})
        plugin_dir = stats_config.get("plugin_directory")
        enabled_plugins = stats_config.get("enabled_plugins", [])

        if not plugin_dir:
            QMessageBox.information(
                self,
                "No Plugin Directory",
                "No plugin directory configured. Set 'statistics.plugin_directory' "
                "in your config file.",
            )
            return

        try:
            self.statistics_calculator.reload_plugins(plugin_dir, enabled_plugins)

            # Get all available metrics for display
            all_metrics = self.statistics_calculator.get_all_metric_names()

            QMessageBox.information(
                self,
                "Plugins Reloaded",
                f"Successfully reloaded plugins.\n\n"
                f"Available metrics: {', '.join(all_metrics)}",
            )

            # Refresh checkbox list with new metrics
            self._refresh_statistics_checkboxes()

            # Recalculate statistics with new plugins
            if self.current_waveform:
                self._calculate_and_display_statistics()

            logger.info("Custom statistic plugins reloaded")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Plugin Reload Failed",
                f"Failed to reload plugins: {str(e)}",
            )
            logger.error(f"Failed to reload plugins: {e}", exc_info=True)

    def _update_status_bar_connection(self, message: str):
        """Update status bar with connection info and trapping parameters status."""
        parts = [message]

        if self.trapping_params_received:
            parts.append("Trapping params: ✓")
            # Add active step count
            num_active = len(self._active_step_indices)
            total_steps = (
                len(self._active_step_indices)
                if self.mqtt_client.trapping_parameters
                else 0
            )
            if self.mqtt_client.trapping_parameters:
                total_steps = len(self.mqtt_client.trapping_parameters.excitation_steps)
                parts.append(f"Active steps: {num_active}/{total_steps}")
        else:
            parts.append("Trapping params: ✗")

        status_message = " | ".join(parts)
        self.status_bar.showMessage(status_message)

    def _get_active_step_indices(self) -> List[int]:
        """Get list of indices for active excitation steps."""
        trapping_params = None
        if self.current_waveform is not None:
            trapping_params = self.current_waveform.trapping_parameters
        elif self.mqtt_client.trapping_parameters is not None:
            trapping_params = self.mqtt_client.trapping_parameters

        if trapping_params is None:
            return []

        return [
            i
            for i, step in enumerate(trapping_params.excitation_steps)
            if step.step_active
        ]

    def _find_next_active_step(self, current_index: int) -> Optional[int]:
        """Find the next active step after the given index."""
        if not self._active_step_indices:
            return None

        # Find next active step after current
        for idx in self._active_step_indices:
            if idx > current_index:
                return idx

        # No next step, return None
        return None

    def _find_previous_active_step(self, current_index: int) -> Optional[int]:
        """Find the previous active step before the given index."""
        if not self._active_step_indices:
            return None

        # Find previous active step before current
        for idx in reversed(self._active_step_indices):
            if idx < current_index:
                return idx

        # No previous step, return None
        return None

    def _update_excitation_step_spinbox(self, trapping_params):
        """Update the excitation step spinbox with only active steps."""
        if self.excitation_step_spinbox is None or trapping_params is None:
            return

        # Get active step indices
        self._active_step_indices = [
            i
            for i, step in enumerate(trapping_params.excitation_steps)
            if step.step_active
        ]

        if not self._active_step_indices:
            # No active steps
            self.excitation_step_spinbox.setEnabled(False)
            self.excitation_step_spinbox.setMinimum(1)
            self.excitation_step_spinbox.setMaximum(1)
            self.excitation_step_spinbox.setValue(1)
            self._update_plot_title()
            return

        self.excitation_step_spinbox.setEnabled(True)

        # Check if current step is still active
        if self.selected_excitation_step not in self._active_step_indices:
            # Try to find next active step
            new_step = self._find_next_active_step(self.selected_excitation_step)
            if new_step is None:
                # Try previous active step
                new_step = self._find_previous_active_step(
                    self.selected_excitation_step
                )
            if new_step is None:
                # Fallback to first active step
                new_step = self._active_step_indices[0]

            logger.info(
                f"Current step {self.selected_excitation_step} is inactive, switching to step {new_step}"
            )
            self.selected_excitation_step = new_step
            # Save to state
            self.state["preferences"]["selected_excitation_step"] = new_step
            state_manager.save_state(self.state_path, self.state)

            # Update MQTT clients to use new excitation step for future waveforms
            self.mqtt_client.set_selected_excitation_step(new_step)
            if self.mqtt_output_client:
                self.mqtt_output_client.set_selected_excitation_step(new_step)

        # Update spinbox range to only show active steps (display as 1-based)
        self.excitation_step_spinbox.blockSignals(True)
        self.excitation_step_spinbox.setMinimum(min(self._active_step_indices) + 1)
        self.excitation_step_spinbox.setMaximum(max(self._active_step_indices) + 1)
        self.excitation_step_spinbox.setValue(self.selected_excitation_step + 1)
        self.excitation_step_spinbox.blockSignals(False)

        # Update tooltip with active steps info (display as 1-based)
        active_steps_str = ", ".join(str(i + 1) for i in self._active_step_indices)
        self.excitation_step_spinbox.setToolTip(
            f"Select active excitation step for m/z calculation.\nActive steps: {active_steps_str}"
        )

    def _publish_statistics(self):
        """Publish current statistics to MQTT statistics topic."""
        import json

        if not self.current_statistics:
            return

        # Determine which client to use
        use_output_broker = (
            self.state["mqtt"].get("output_broker_enabled", False)
            and self.mqtt_output_client is not None
            and self.mqtt_output_client.is_connected()
        )

        client_to_use = (
            self.mqtt_output_client if use_output_broker else self.mqtt_client
        )
        statistics_topic = (
            self.state["mqtt"].get(
                "output_broker_statistics_topic", "lqt/lqt_ms/statistics"
            )
            if use_output_broker
            else self.state["mqtt"].get("statistics_topic", "lqt/lqt_ms/statistics")
        )

        # Check if client is connected
        if client_to_use is None or not client_to_use.is_connected():
            return

        # Build payload with all statistics
        payload_data = {
            "timestamp": (
                self.current_waveform.timestamp.isoformat()
                if self.current_waveform
                else datetime.now().isoformat()
            ),
            "num_averaged_waveforms": self._num_averaged_waveforms,
            "statistics": [stats.to_dict() for stats in self.current_statistics],
        }

        payload_bytes = json.dumps(payload_data).encode("utf-8")

        try:
            if client_to_use is not None:  # Type guard
                client_to_use.publish(statistics_topic, payload_bytes)
                broker_name = "output broker" if use_output_broker else "primary broker"
                logger.debug(
                    f"Published statistics to {statistics_topic} on {broker_name}"
                )
        except Exception as e:
            logger.error(f"Failed to publish statistics: {e}")

    # Slots for MQTT signals

    @pyqtSlot(object)
    def on_trapping_parameters_received(self, trapping_params):
        """Handle received trapping parameters from MQTT."""
        if not self.trapping_params_received:
            self.trapping_params_received = True
            logger.info("Trapping parameters received and status updated")

        # Update excitation step spinbox with only active steps
        self._update_excitation_step_spinbox(trapping_params)

        # Update status bar with active step count
        if self.mqtt_client.is_connected():
            self._update_status_bar_connection("Connected")

        # Update plot title with step info
        self._update_plot_title()

    @pyqtSlot(object)
    def on_waveform_received(self, waveform: WaveformData):
        """Handle received waveform data."""
        first_waveform = self.current_waveform is None
        self.current_waveform = waveform
        self.original_waveform = waveform  # Store original unmodified waveform
        self._waveform_counter += 1

        # Update region positions on first waveform if we have restored regions
        if first_waveform and len(self.regions) > 0:
            self._update_region_positions()

        # Update scaling controls for new channels (only if changed)
        if (
            not hasattr(self, "_last_channel_set")
            or set(waveform.channels.keys()) != self._last_channel_set
        ):
            self._update_scaling_controls()
            self._last_channel_set = set(waveform.channels.keys())

        # Update excitation step spinbox with only active steps
        self._update_excitation_step_spinbox(waveform.trapping_parameters)

        # Update plot - use threaded scaling for large waveforms
        max_samples = max(len(data) for data in waveform.channels.values())
        use_threaded = max_samples > 100000  # Use threading for >100k points

        if use_threaded:
            # Prepare plot data in background thread
            if not self._plot_data_worker.isRunning():
                # Build scaling functions dictionary with captured parameters
                scaling_functions = {}
                for ch_name in waveform.channels.keys():
                    if ch_name in self.channel_scaling:
                        gain = self.channel_scaling[ch_name]["gain"]
                        offset = self.channel_scaling[ch_name]["offset"]
                        scaling_functions[ch_name] = (
                            lambda data, g=gain, o=offset: g * (data - o)
                        )
                    else:
                        scaling_functions[ch_name] = lambda data: data.copy()

                x_axis_mode = self.state.get("gui", {}).get("x_axis_mode", "mz")
                self._plot_data_worker.setup_preparation(
                    waveform, scaling_functions, x_axis_mode
                )
                self._plot_data_worker.start()
            else:
                # Worker is busy, mark as pending
                self._plot_data_pending = True
                self._pending_plot_waveform = waveform
        else:
            # Small waveform - process immediately
            self._update_plot(waveform)

        # Accumulate waveforms and calculate statistics (throttled for performance)
        if len(self.regions) > 0:
            # Add current waveform to accumulation buffer
            self._accumulated_waveforms.append(waveform)

            now = datetime.now()
            time_since_last_update = (
                now - self._last_statistics_update_time
            ).total_seconds()

            should_update = (
                self._waveform_counter % self._statistics_update_interval == 0
                or time_since_last_update >= self._statistics_update_min_interval
            )

            if should_update:
                # Calculate statistics from averaged accumulated waveforms
                self._calculate_and_display_statistics()
                self._last_statistics_update_time = now

        # Enable action buttons (only on first waveform)
        if not self.export_csv_btn.isEnabled():
            self.export_csv_btn.setEnabled(True)
            self.republish_btn.setEnabled(True)

        # Update status
        self.status_bar.showMessage(
            f"Received MS with {len(waveform.mz)} samples at {waveform.timestamp.strftime('%H:%M:%S')}",
            5000,
        )

    @pyqtSlot(bool, str)
    def on_connection_state_changed(self, connected: bool, message: str):
        """Handle connection state changes."""
        if connected:
            self._update_status_bar_connection(f"Connected: {message}")
            self.apply_mqtt_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
        else:
            # Reset trapping params flag on disconnect
            self.trapping_params_received = False
            self._update_status_bar_connection(f"Disconnected: {message}")
            self.apply_mqtt_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)

    @pyqtSlot(str)
    def on_error(self, message: str):
        """Handle error messages."""
        logger.error(f"MQTT error: {message}")
        self.status_bar.showMessage(f"Error: {message}", 10000)

    @pyqtSlot(bool, str)
    def on_output_connection_state_changed(self, connected: bool, message: str):
        """Handle output broker connection state changes."""
        if connected:
            self.status_bar.showMessage(f"Output broker connected: {message}", 5000)
            self.apply_output_mqtt_btn.setEnabled(False)
            self.disconnect_output_btn.setEnabled(True)
        else:
            self.status_bar.showMessage(f"Output broker disconnected: {message}", 5000)
            self.apply_output_mqtt_btn.setEnabled(True)
            self.disconnect_output_btn.setEnabled(False)

    @pyqtSlot(str)
    def on_output_error(self, message: str):
        """Handle output broker error messages."""
        logger.error(f"Output MQTT error: {message}")
        self.status_bar.showMessage(f"Output broker error: {message}", 10000)

    def _update_plot(self, waveform: WaveformData):
        """
        Update the plot with new waveform data.

        Args:
            waveform: WaveformData to plot
        """
        # Get x-axis data based on current mode
        x_axis_mode = self.state.get("gui", {}).get("x_axis_mode", "mz")
        if x_axis_mode == "frequency":
            x_data = waveform.f_AC_kHz
        elif x_axis_mode == "rf_amplitude":
            x_data = waveform.rf_amp_V0p
        else:
            x_data = waveform.mz

        # Get sorted channel names for consistent ordering
        channel_names = sorted(waveform.channels.keys())

        # Remove curves that no longer exist
        for ch_name in list(self.plot_curves.keys()):
            if ch_name not in channel_names:
                self.plot_widget.removeItem(self.plot_curves[ch_name])
                del self.plot_curves[ch_name]

        # Update or create curves
        for i, ch_name in enumerate(channel_names):
            color = tuple(self.channel_colors[i % len(self.channel_colors)])

            # Apply scaling: y := Gain * (y - Offset)
            scaled_data = self._apply_channel_scaling(
                ch_name, waveform.channels[ch_name]
            )

            if ch_name in self.plot_curves:
                # Update existing curve
                self.plot_curves[ch_name].setData(x_data, scaled_data)
            else:
                # Create new curve
                pen = pg.mkPen(color=color, width=2)
                curve = self.plot_widget.plot(
                    x_data, scaled_data, pen=pen, name=ch_name
                )
                curve.setDownsampling(auto=True)
                curve.setClipToView(True)
                self.plot_curves[ch_name] = curve

        # Update title with timestamp and excitation step info
        timestamp_str = f"Last timestamp: {waveform.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
        self._update_plot_title(timestamp_str)

    def _update_plot_with_scaled_data(
        self,
        x_data: np.ndarray,
        scaled_channels: Dict[str, np.ndarray],
        timestamp: datetime,
    ):
        """
        Update the plot with pre-scaled channel data (from worker thread).

        Args:
            x_data: X-axis data array (m/z, f_AC, or rf_amp depending on mode)
            scaled_channels: Dictionary of channel_name -> scaled_data
            timestamp: Waveform timestamp
        """
        # Get sorted channel names for consistent ordering
        channel_names = sorted(scaled_channels.keys())

        # Remove curves that no longer exist
        for ch_name in list(self.plot_curves.keys()):
            if ch_name not in channel_names:
                self.plot_widget.removeItem(self.plot_curves[ch_name])
                del self.plot_curves[ch_name]

        # Update or create curves
        for i, ch_name in enumerate(channel_names):
            color = tuple(self.channel_colors[i % len(self.channel_colors)])
            scaled_data = scaled_channels[ch_name]

            if ch_name in self.plot_curves:
                # Update existing curve
                self.plot_curves[ch_name].setData(x_data, scaled_data)
            else:
                # Create new curve
                pen = pg.mkPen(color=color, width=2)
                curve = self.plot_widget.plot(
                    x_data, scaled_data, pen=pen, name=ch_name
                )
                curve.setDownsampling(auto=True)
                curve.setClipToView(True)
                self.plot_curves[ch_name] = curve

        # Update title with timestamp and excitation step info
        timestamp_str = (
            f"Last timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
        )
        self._update_plot_title(timestamp_str)

    def closeEvent(self, event):  # type: ignore[override]
        """Handle window close event - save state."""
        # Save window geometry
        geom = self.geometry()
        self.state["gui"]["window_geometry"] = {
            "x": geom.x(),
            "y": geom.y(),
            "width": geom.width(),
            "height": geom.height(),
        }

        # Save splitter position
        self.state["gui"]["splitter_sizes"] = self.plots_splitter.sizes()

        # Save regions (always save as m/z values, not visual positions)
        regions_data = []
        for i, region in enumerate(self.regions):
            # Use stored m/z bounds instead of visual position
            if region in self.region_mz_bounds:
                start, end = self.region_mz_bounds[region]
            else:
                # Fallback: convert visual position to m/z
                region_values = region.getRegion()
                start_x = float(region_values[0])  # type: ignore[arg-type]
                end_x = float(region_values[1])  # type: ignore[arg-type]
                start, end = self._xaxis_to_mz(start_x, end_x)

            name = f"Region {i+1}"
            if i < self.region_table.rowCount():
                item = self.region_table.item(i, 0)
                if item is not None:
                    name = item.text()
            regions_data.append({"start": start, "end": end, "name": name})
        self.state["gui"]["regions"] = regions_data

        # Save selected metrics from checkboxes
        selected_metrics = [
            metric_name
            for metric_name, checkbox in self.stats_checkboxes.items()
            if checkbox.isChecked()
        ]
        self.state["gui"]["selected_metrics"] = selected_metrics

        # Save channel scaling parameters
        self.state["channel_scaling"] = self.channel_scaling.copy()

        # Save state to file
        state_manager.save_state(self.state_path, self.state)

        # Disconnect MQTT
        self.mqtt_client.disconnect_from_broker()

        event.accept()
