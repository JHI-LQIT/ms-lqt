"""
Unit tests for custom statistics plugin system.

Tests plugin loading, expression evaluation, and custom statistics calculation.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from waveform_viewer.data_model import WaveformData
from waveform_viewer.roi_statistics import (
    CustomStatisticPlugin,
    ExpressionStatistic,
    ROIDefinition,
    ROIStatistics,
    ROIStatisticsCalculator,
)


# Test plugin classes
class SimplePlugin(CustomStatisticPlugin):
    """Simple test plugin that calculates RMS."""

    def calculate(self, times, amplitudes, roi_start, roi_end):
        rms = np.sqrt(np.mean(amplitudes**2))
        return {"rms": float(rms)}

    def get_metrics(self):
        return ["rms"]


class MultiMetricPlugin(CustomStatisticPlugin):
    """Plugin that returns multiple metrics."""

    def calculate(self, times, amplitudes, roi_start, roi_end):
        return {
            "metric1": float(np.mean(amplitudes)),
            "metric2": float(np.std(amplitudes)),
            "metric3": float(np.max(amplitudes) - np.min(amplitudes)),
        }

    def get_metrics(self):
        return ["metric1", "metric2", "metric3"]


class FailingPlugin(CustomStatisticPlugin):
    """Plugin that raises an exception."""

    def calculate(self, times, amplitudes, roi_start, roi_end):
        raise ValueError("Intentional test failure")

    def get_metrics(self):
        return ["failing_metric"]


class TestExpressionStatistic:
    """Test expression-based statistics."""

    def test_simple_expression(self):
        """Test simple expression evaluation."""
        expr = ExpressionStatistic("test_mean", "np.mean(data)")
        times = np.array([0.0, 0.1, 0.2])
        amplitudes = np.array([1.0, 2.0, 3.0])

        result = expr.calculate(times, amplitudes)
        assert result == 2.0

    def test_complex_expression(self):
        """Test more complex expression."""
        expr = ExpressionStatistic("rms", "np.sqrt(np.mean(data**2))")
        amplitudes = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.1, 0.2])

        result = expr.calculate(times, amplitudes)
        expected = np.sqrt((1 + 4 + 9) / 3)
        assert abs(result - expected) < 1e-10

    def test_expression_with_times(self):
        """Test expression using both times and amplitudes."""
        expr = ExpressionStatistic("duration", "times[-1] - times[0]")
        times = np.array([0.0, 0.5, 1.0])
        amplitudes = np.array([1.0, 2.0, 3.0])

        result = expr.calculate(times, amplitudes)
        assert result == 1.0

    def test_invalid_expression_syntax(self):
        """Test that invalid syntax raises error."""
        with pytest.raises(SyntaxError):
            ExpressionStatistic("bad", "this is not valid python")

    def test_expression_timeout(self):
        """Test that timeout is enforced (Unix-like systems only)."""
        import signal

        if not hasattr(signal, "SIGALRM"):
            pytest.skip("SIGALRM not available on this platform")

        # Expression that would take a long time - compute intensive operation
        # Using nested loops with numpy operations that should exceed timeout
        expr = ExpressionStatistic(
            "slow", "sum([np.sum(data ** 2) for _ in [0] * 10**7])", timeout=1
        )
        times = np.array([0.0, 0.1, 0.2])
        amplitudes = np.array([1.0, 2.0, 3.0])

        # Should timeout
        with pytest.raises(TimeoutError):
            expr.calculate(times, amplitudes)

    def test_display_name(self):
        """Test display name handling."""
        expr1 = ExpressionStatistic("test", "1.0")
        assert expr1.display_name == "test"

        expr2 = ExpressionStatistic("test", "1.0", display_name="Test Metric")
        assert expr2.display_name == "Test Metric"


class TestCustomStatisticPlugin:
    """Test custom statistic plugin base class."""

    def test_simple_plugin(self):
        """Test simple plugin calculation."""
        plugin = SimplePlugin()

        times = np.array([0.0, 0.01, 0.02])
        amplitudes = np.array([1.0, 2.0, 3.0])

        result = plugin.calculate(times, amplitudes, 0.0, 0.02)

        assert "rms" in result
        expected_rms = np.sqrt((1 + 4 + 9) / 3)
        assert abs(result["rms"] - expected_rms) < 1e-10

    def test_multi_metric_plugin(self):
        """Test plugin that returns multiple metrics."""
        plugin = MultiMetricPlugin()

        times = np.array([0.0, 0.01, 0.02])
        amplitudes = np.array([1.0, 2.0, 3.0])

        result = plugin.calculate(times, amplitudes, 0.0, 0.02)

        assert len(result) == 3
        assert "metric1" in result
        assert "metric2" in result
        assert "metric3" in result

        assert result["metric1"] == 2.0
        assert result["metric3"] == 2.0  # max - min

    def test_get_metrics(self):
        """Test get_metrics method."""
        plugin1 = SimplePlugin()
        assert plugin1.get_metrics() == ["rms"]

        plugin2 = MultiMetricPlugin()
        assert set(plugin2.get_metrics()) == {"metric1", "metric2", "metric3"}


class TestROIStatisticsCalculator:
    """Test ROI statistics calculator with custom plugins."""

    def create_test_waveform(self):
        """Create a test waveform."""
        times = np.linspace(0, 0.1, 1000)
        ch1 = np.sin(2 * np.pi * 10 * times)
        ch2 = np.cos(2 * np.pi * 10 * times)

        return WaveformData(
            t0=0.0,
            dt=0.0001,
            channels={"ch1": ch1, "ch2": ch2},
            timestamp=datetime.now(),
        )

    def test_builtin_statistics_only(self):
        """Test calculator with only built-in statistics."""
        calc = ROIStatisticsCalculator()
        waveform = self.create_test_waveform()
        roi = ROIDefinition("test_roi", 0.01, 0.02)

        stats = calc.calculate_statistics(waveform, roi)

        assert len(stats) == 2  # Two channels
        for stat in stats:
            assert stat.roi_name == "test_roi"
            assert isinstance(stat.mean, float)
            assert isinstance(stat.std_dev, float)
            assert len(stat.custom_metrics) == 0  # No custom metrics

    def test_with_custom_plugin(self):
        """Test calculator with a custom plugin."""
        calc = ROIStatisticsCalculator()
        calc.plugins.append(SimplePlugin())

        waveform = self.create_test_waveform()
        roi = ROIDefinition("test_roi", 0.01, 0.02)

        stats = calc.calculate_statistics(waveform, roi)

        assert len(stats) == 2
        for stat in stats:
            assert "rms" in stat.custom_metrics
            assert isinstance(stat.custom_metrics["rms"], float)

    def test_with_expression_statistic(self):
        """Test calculator with expression-based statistic."""
        calc = ROIStatisticsCalculator(
            expression_stats=[{"name": "test_expr", "expression": "np.mean(data**2)"}]
        )

        waveform = self.create_test_waveform()
        roi = ROIDefinition("test_roi", 0.01, 0.02)

        stats = calc.calculate_statistics(waveform, roi)

        assert len(stats) == 2
        for stat in stats:
            assert "test_expr" in stat.custom_metrics

    def test_plugin_failure_handling(self):
        """Test that failing plugin doesn't break calculation."""
        calc = ROIStatisticsCalculator()
        calc.plugins.append(FailingPlugin())

        waveform = self.create_test_waveform()
        roi = ROIDefinition("test_roi", 0.01, 0.02)

        # Should not raise exception
        stats = calc.calculate_statistics(waveform, roi)

        # Should still get built-in stats
        assert len(stats) == 2
        for stat in stats:
            assert "failing_metric" not in stat.custom_metrics

    def test_configurable_builtin_stats(self):
        """Test disabling built-in statistics."""
        # Only enable mean and std_dev
        calc = ROIStatisticsCalculator(enabled_builtin_stats=["mean", "std_dev"])

        waveform = self.create_test_waveform()
        roi = ROIDefinition("test_roi", 0.01, 0.02)

        stats = calc.calculate_statistics(waveform, roi)

        for stat in stats:
            # These should be calculated
            assert stat.mean != 0.0
            assert stat.std_dev != 0.0

            # These should be zero (disabled)
            assert stat.min_value == 0.0
            assert stat.max_value == 0.0

    def test_get_all_metric_names(self):
        """Test getting all available metric names."""
        calc = ROIStatisticsCalculator(
            enabled_builtin_stats=["mean", "std_dev"],
            expression_stats=[{"name": "expr1", "expression": "1.0"}],
        )
        calc.plugins.append(SimplePlugin())

        metrics = calc.get_all_metric_names()

        assert "mean" in metrics
        assert "std_dev" in metrics
        assert "rms" in metrics
        assert "expr1" in metrics

    def test_multiple_rois(self):
        """Test calculating statistics for multiple ROIs."""
        calc = ROIStatisticsCalculator()
        calc.plugins.append(SimplePlugin())

        waveform = self.create_test_waveform()
        rois = [
            ROIDefinition("roi1", 0.01, 0.02),
            ROIDefinition("roi2", 0.05, 0.06),
        ]

        all_stats = calc.calculate_all_statistics(waveform, rois)

        # 2 ROIs × 2 channels = 4 statistics objects
        assert len(all_stats) == 4

        roi_names = {stat.roi_name for stat in all_stats}
        assert roi_names == {"roi1", "roi2"}

    def test_custom_metrics_in_to_dict(self):
        """Test that custom metrics appear in to_dict output."""
        calc = ROIStatisticsCalculator()
        calc.plugins.append(MultiMetricPlugin())

        waveform = self.create_test_waveform()
        roi = ROIDefinition("test_roi", 0.01, 0.02)

        stats = calc.calculate_statistics(waveform, roi)
        stat_dict = stats[0].to_dict()

        # Built-in stats
        assert "mean" in stat_dict
        assert "std_dev" in stat_dict

        # Custom metrics
        assert "metric1" in stat_dict
        assert "metric2" in stat_dict
        assert "metric3" in stat_dict


class TestPluginLoading:
    """Test plugin loading from files."""

    def test_load_plugin_from_file(self):
        """Test loading plugin from a Python file."""
        # Create temporary plugin file
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = Path(tmpdir) / "test_plugin.py"
            plugin_code = """
from waveform_viewer.roi_statistics import CustomStatisticPlugin
import numpy as np

class TestPlugin(CustomStatisticPlugin):
    def calculate(self, times, amplitudes, roi_start, roi_end):
        return {"test_value": 42.0}
    
    def get_metrics(self):
        return ["test_value"]
"""
            plugin_path.write_text(plugin_code)

            # Load plugin
            calc = ROIStatisticsCalculator(
                plugin_dir=tmpdir, enabled_plugins=["test_plugin"]
            )

            # Check plugin was loaded
            assert len(calc.plugins) == 1
            assert "test_value" in calc.get_all_metric_names()

    def test_load_legacy_plugin_format(self):
        """Test loading legacy-style plugin (function-based)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = Path(tmpdir) / "legacy_plugin.py"
            plugin_code = """
def calculate(times, amplitudes, roi_start, roi_end):
    return {"legacy_metric": 99.0}

PLUGIN_INFO = {
    "name": "Legacy Plugin",
    "metrics": ["legacy_metric"]
}
"""
            plugin_path.write_text(plugin_code)

            calc = ROIStatisticsCalculator(
                plugin_dir=tmpdir, enabled_plugins=["legacy_plugin"]
            )

            assert len(calc.plugins) == 1
            assert "legacy_metric" in calc.get_all_metric_names()

    def test_reload_plugins(self):
        """Test reloading plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = Path(tmpdir) / "reloadable.py"
            plugin_code = """
from waveform_viewer.roi_statistics import CustomStatisticPlugin
import signal

class ReloadablePlugin(CustomStatisticPlugin):
    def calculate(self, times, amplitudes, roi_start, roi_end):
        return {"reload_test": 1.0}
    
    def get_metrics(self):
        return ["reload_test"]
"""
            plugin_path.write_text(plugin_code)

            calc = ROIStatisticsCalculator(
                plugin_dir=tmpdir, enabled_plugins=["reloadable"]
            )
            assert len(calc.plugins) == 1

            # Reload should work
            calc.reload_plugins(tmpdir, ["reloadable"])
            assert len(calc.plugins) == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_roi(self):
        """Test ROI with no data points."""
        calc = ROIStatisticsCalculator()

        times = np.array([0.0, 0.1, 0.2])
        channels = {"ch1": np.array([1.0, 2.0, 3.0])}
        waveform = WaveformData(0.0, 0.1, channels, datetime.now())

        # ROI outside data range
        roi = ROIDefinition("empty", 0.5, 0.6)

        stats = calc.calculate_statistics(waveform, roi)

        # Should still return statistics (with at least one point)
        assert len(stats) == 1

    def test_invalid_roi_times(self):
        """Test ROI with start >= end."""
        calc = ROIStatisticsCalculator()

        times = np.array([0.0, 0.1, 0.2])
        channels = {"ch1": np.array([1.0, 2.0, 3.0])}
        waveform = WaveformData(0.0, 0.1, channels, datetime.now())

        roi = ROIDefinition("invalid", 0.2, 0.1)  # start > end

        with pytest.raises(ValueError):
            calc.calculate_statistics(waveform, roi)

    def test_nan_values(self):
        """Test handling of NaN values in data."""
        calc = ROIStatisticsCalculator()

        times = np.array([0.0, 0.1, 0.2])
        channels = {"ch1": np.array([1.0, np.nan, 3.0])}
        waveform = WaveformData(0.0, 0.1, channels, datetime.now())

        roi = ROIDefinition("test", 0.0, 0.2)

        # Should handle NaN gracefully
        stats = calc.calculate_statistics(waveform, roi)
        assert len(stats) == 1

    def test_expression_with_division_by_zero(self):
        """Test expression that could cause division by zero."""
        calc = ROIStatisticsCalculator(
            expression_stats=[
                {"name": "div_zero", "expression": "np.mean(data) / np.std(data)"}
            ]
        )

        # Constant signal (std = 0)
        times = np.array([0.0, 0.1, 0.2])
        channels = {"ch1": np.array([1.0, 1.0, 1.0])}
        waveform = WaveformData(0.0, 0.1, channels, datetime.now())

        roi = ROIDefinition("test", 0.0, 0.2)

        # Should handle division by zero (result will be inf)
        stats = calc.calculate_statistics(waveform, roi)
        assert len(stats) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
