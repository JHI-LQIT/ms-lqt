"""
Unit tests for threaded statistics calculation.
"""

import time
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from waveform_viewer.gui import StatisticsWorker
from waveform_viewer.data_model import WaveformData
from waveform_viewer.roi_statistics import ROIDefinition, ROIStatisticsCalculator


# Create QApplication instance for Qt tests
app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestStatisticsWorker:
    """Tests for the StatisticsWorker thread."""

    def test_worker_basic_calculation(self):
        """Test that the worker can perform basic statistics calculation."""
        # Create test data
        waveforms = [
            WaveformData(
                t0=0.0,
                dt=0.001,
                channels={"ch1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
                timestamp="2025-01-01T00:00:00.000Z",
            )
        ]
        
        rois = [ROIDefinition(name="Test ROI", start_time=0.0, end_time=0.005)]
        scaling_functions = {"ch1": lambda data: data}
        calculator = ROIStatisticsCalculator()
        
        # Create worker
        worker = StatisticsWorker()
        
        # Set up signal capture
        results = []
        worker.calculation_complete.connect(lambda stats, num: results.append((stats, num)))
        
        # Run calculation
        worker.setup_calculation(waveforms, rois, scaling_functions, calculator)
        worker.start()
        
        # Wait for completion and process events
        timeout = 5000  # 5 second timeout
        elapsed = 0
        while not worker.isFinished() and elapsed < timeout:
            app.processEvents()
            worker.wait(10)
            elapsed += 10
        
        # Process remaining events to ensure signal is delivered
        app.processEvents()
        
        # Verify results
        assert len(results) == 1
        stats, num_averaged = results[0]
        assert num_averaged == 1
        assert len(stats) == 1
        assert stats[0].roi_name == "Test ROI"
        assert stats[0].channel_name == "ch1"

    def test_worker_averages_multiple_waveforms(self):
        """Test that the worker correctly averages multiple waveforms."""
        # Create test data - 3 waveforms with different values
        waveforms = [
            WaveformData(
                t0=0.0,
                dt=0.001,
                channels={"ch1": np.array([1.0, 2.0, 3.0])},
                timestamp="2025-01-01T00:00:00.000Z",
            ),
            WaveformData(
                t0=0.0,
                dt=0.001,
                channels={"ch1": np.array([2.0, 4.0, 6.0])},
                timestamp="2025-01-01T00:00:00.001Z",
            ),
            WaveformData(
                t0=0.0,
                dt=0.001,
                channels={"ch1": np.array([3.0, 6.0, 9.0])},
                timestamp="2025-01-01T00:00:00.002Z",
            ),
        ]
        
        rois = [ROIDefinition(name="Test ROI", start_time=0.0, end_time=0.003)]
        scaling_functions = {"ch1": lambda data: data}
        calculator = ROIStatisticsCalculator()
        
        # Create worker
        worker = StatisticsWorker()
        
        # Set up signal capture
        results = []
        worker.calculation_complete.connect(lambda stats, num: results.append((stats, num)))
        
        # Run calculation
        worker.setup_calculation(waveforms, rois, scaling_functions, calculator)
        worker.start()
        
        # Wait for completion
        timeout = 5000
        elapsed = 0
        while not worker.isFinished() and elapsed < timeout:
            app.processEvents()
            worker.wait(10)
            elapsed += 10
        app.processEvents()
        
        # Verify results
        assert len(results) == 1
        stats, num_averaged = results[0]
        assert num_averaged == 3
        assert len(stats) == 1
        
        # Check that averaging worked: mean should be (1+2+3)/3=2, (2+4+6)/3=4, (3+6+9)/3=6
        # So the mean of the averaged waveform should be (2+4+6)/3 = 4
        assert abs(stats[0].mean - 4.0) < 0.01

    def test_worker_handles_dimension_mismatch(self):
        """Test that the worker handles waveforms with different dimensions."""
        # Create test data with mismatched dimensions
        waveforms = [
            WaveformData(
                t0=0.0,
                dt=0.001,
                channels={"ch1": np.array([1.0, 2.0, 3.0])},
                timestamp="2025-01-01T00:00:00.000Z",
            ),
            WaveformData(
                t0=0.0,
                dt=0.001,
                channels={"ch1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])},  # Different size
                timestamp="2025-01-01T00:00:00.001Z",
            ),
        ]
        
        rois = [ROIDefinition(name="Test ROI", start_time=0.0, end_time=0.003)]
        scaling_functions = {"ch1": lambda data: data}
        calculator = ROIStatisticsCalculator()
        
        # Create worker
        worker = StatisticsWorker()
        
        # Set up signal capture
        results = []
        worker.calculation_complete.connect(lambda stats, num: results.append((stats, num)))
        
        # Run calculation
        worker.setup_calculation(waveforms, rois, scaling_functions, calculator)
        worker.start()
        
        # Wait for completion
        timeout = 5000
        elapsed = 0
        while not worker.isFinished() and elapsed < timeout:
            app.processEvents()
            worker.wait(10)
            elapsed += 10
        app.processEvents()
        
        # Verify that it used the last waveform only (fallback behavior)
        assert len(results) == 1
        stats, num_averaged = results[0]
        assert num_averaged == 2  # Still reports all waveforms

    def test_worker_applies_scaling(self):
        """Test that the worker correctly applies scaling functions."""
        # Create test data
        waveforms = [
            WaveformData(
                t0=0.0,
                dt=0.001,
                channels={"ch1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
                timestamp="2025-01-01T00:00:00.000Z",
            )
        ]
        
        rois = [ROIDefinition(name="Test ROI", start_time=0.0, end_time=0.005)]
        
        # Scaling: y = 2 * (y - 1)
        scaling_functions = {"ch1": lambda data: 2.0 * (data - 1.0)}
        calculator = ROIStatisticsCalculator()
        
        # Create worker
        worker = StatisticsWorker()
        
        # Set up signal capture
        results = []
        worker.calculation_complete.connect(lambda stats, num: results.append((stats, num)))
        
        # Run calculation
        worker.setup_calculation(waveforms, rois, scaling_functions, calculator)
        worker.start()
        
        # Wait for completion
        timeout = 5000
        elapsed = 0
        while not worker.isFinished() and elapsed < timeout:
            app.processEvents()
            worker.wait(10)
            elapsed += 10
        app.processEvents()
        
        # Verify results - scaled data should be [0, 2, 4, 6, 8]
        assert len(results) == 1
        stats, num_averaged = results[0]
        
        # Mean of [0, 2, 4, 6, 8] = 4.0
        assert abs(stats[0].mean - 4.0) < 0.01
