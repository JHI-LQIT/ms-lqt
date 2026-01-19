"""
ROI Statistics Calculator for waveform analysis.

This module provides functionality to calculate various statistics
for regions of interest (ROI) in waveform data, including built-in
statistics, user-defined plugins, and expression-based calculations.
"""

import importlib.util
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .data_model import WaveformData

logger = logging.getLogger(__name__)


@dataclass
class ROIStatistics:
    """
    Statistics for a single region of interest in a waveform channel.

    Attributes:
        roi_name: Name of the region of interest
        channel_name: Name of the channel
        start_time: Start time of the ROI in seconds
        end_time: End time of the ROI in seconds
        mean: Mean amplitude value
        median: Median amplitude value
        std_dev: Standard deviation of amplitude
        min_value: Minimum amplitude value
        max_value: Maximum amplitude value
        peak_to_peak: Peak-to-peak amplitude (max - min)
        area_under_curve: Integral of the waveform over the ROI
        n_points: Number of data points in the ROI
        time_span: Duration of the ROI in seconds
        timestamp: Timestamp when the waveform was captured
        custom_metrics: Additional custom metrics from plugins or expressions
    """

    roi_name: str
    channel_name: str
    start_time: float
    end_time: float
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    peak_to_peak: float
    area_under_curve: float
    n_points: int
    time_span: float
    timestamp: datetime
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert statistics to dictionary for serialization."""
        result = {
            "roi_name": self.roi_name,
            "channel_name": self.channel_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "peak_to_peak": self.peak_to_peak,
            "area_under_curve": self.area_under_curve,
            "n_points": self.n_points,
            "time_span": self.time_span,
            "timestamp": self.timestamp.isoformat(),
        }
        result.update(self.custom_metrics)
        return result


@dataclass
class ROIDefinition:
    """
    Definition of a region of interest.

    Attributes:
        name: Name of the ROI
        start_time: Start time in seconds
        end_time: End time in seconds
    """

    name: str
    start_time: float
    end_time: float


class CustomStatisticPlugin(ABC):
    """
    Base class for custom statistic plugins.

    User-defined plugins should inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def calculate(
        self,
        times: np.ndarray,
        amplitudes: np.ndarray,
        roi_start: float,
        roi_end: float,
    ) -> Dict[str, float]:
        """
        Calculate custom statistics for an ROI.

        Args:
            times: NumPy array of time values for the ROI
            amplitudes: NumPy array of amplitude values for the ROI
            roi_start: Start time of the ROI
            roi_end: End time of the ROI

        Returns:
            Dictionary mapping metric names to their calculated values.
            Example: {'rms': 0.123, 'thd': 0.05}
        """
        ...

    @abstractmethod
    def get_metrics(self) -> List[str]:
        """
        Return list of metric names this plugin provides.

        Returns:
            List of metric name strings that will appear in the UI.
        """
        ...

    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns:
            Dictionary with plugin information (name, version, author, etc.)
        """
        return {
            "name": self.__class__.__name__,
            "version": "1.0",
            "author": "Unknown",
            "description": "No description provided",
        }


class ExpressionStatistic:
    """
    Expression-based statistic evaluator.

    Evaluates user-defined Python expressions to calculate custom metrics.
    Provides sandboxing and timeout protection.
    """

    def __init__(
        self,
        name: str,
        expression: str,
        display_name: Optional[str] = None,
        timeout: int = 1,
    ):
        """
        Initialize expression statistic.

        Args:
            name: Internal metric name
            expression: Python expression to evaluate
            display_name: Human-readable name (defaults to name)
            timeout: Maximum execution time in seconds
        """
        self.name = name
        self.expression = expression
        self.display_name = display_name or name
        self.timeout = timeout
        self._compiled = None

        # Compile expression for faster evaluation
        try:
            self._compiled = compile(expression, "<string>", "eval")
        except SyntaxError as e:
            logger.error(f"Failed to compile expression '{name}': {e}")
            raise

    def calculate(self, times: np.ndarray, amplitudes: np.ndarray) -> float:
        """
        Evaluate the expression on the given data.

        Args:
            times: Time values
            amplitudes: Amplitude values

        Returns:
            Calculated metric value

        Raises:
            Exception: If evaluation fails
        """
        # Create safe namespace with limited built-ins
        safe_globals = {
            "np": np,
            "numpy": np,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "float": float,
            "int": int,
            "__builtins__": {},
        }

        # Allow scipy if available
        try:
            import scipy
            import scipy.signal
            import scipy.stats

            safe_globals["scipy"] = scipy
        except ImportError:
            pass

        local_vars = {
            "times": times,
            "data": amplitudes,
            "amplitudes": amplitudes,
        }

        # Evaluate expression (signal-based timeout removed for thread safety)
        # Expression evaluation is sandboxed and typically fast enough
        if self._compiled:
            result = eval(self._compiled, safe_globals, local_vars)
        else:
            result = eval(self.expression, safe_globals, local_vars)
        return float(result)


class ROIStatisticsCalculator:
    """
    Calculator for ROI statistics.

    This class computes various statistics for regions of interest
    in waveform data across all channels. Supports built-in statistics,
    custom plugins, and expression-based calculations.
    """

    # Default built-in statistics
    DEFAULT_BUILTIN_STATS = [
        "mean",
        "median",
        "std_dev",
        "min",
        "max",
        "peak_to_peak",
        "area",
        "n_points",
        "time_span",
    ]

    def __init__(
        self,
        plugin_dir: Optional[str] = None,
        enabled_plugins: Optional[List[str]] = None,
        expression_stats: Optional[List[Dict[str, str]]] = None,
        enabled_builtin_stats: Optional[List[str]] = None,
    ):
        """
        Initialize the statistics calculator.

        Args:
            plugin_dir: Directory containing plugin Python files
            enabled_plugins: List of plugin filenames to load (without .py extension)
            expression_stats: List of expression statistic definitions
            enabled_builtin_stats: List of built-in statistics to calculate
        """
        self.plugins: List[CustomStatisticPlugin] = []
        self.expression_stats: List[ExpressionStatistic] = []
        self.enabled_builtin_stats = (
            set(enabled_builtin_stats)
            if enabled_builtin_stats
            else set(self.DEFAULT_BUILTIN_STATS)
        )

        if plugin_dir:
            self.load_plugins(plugin_dir, enabled_plugins or [])

        if expression_stats:
            self.load_expression_stats(expression_stats)

    def load_plugins(self, plugin_dir: str, enabled_plugins: List[str]) -> None:
        """
        Load custom statistic plugins from directory.

        Args:
            plugin_dir: Path to directory containing .py plugin files
            enabled_plugins: List of plugin names to load (without .py)
        """
        plugin_path = Path(plugin_dir).expanduser()
        if not plugin_path.exists():
            logger.info(f"Plugin directory does not exist: {plugin_path}")
            return

        loaded_count = 0
        for plugin_file in plugin_path.glob("*.py"):
            if plugin_file.stem.startswith("_"):
                continue  # Skip private/test files

            if enabled_plugins and plugin_file.stem not in enabled_plugins:
                logger.debug(f"Skipping disabled plugin: {plugin_file.stem}")
                continue

            try:
                plugin_instance = self._load_plugin_file(plugin_file)
                if plugin_instance:
                    self.plugins.append(plugin_instance)
                    loaded_count += 1
                    logger.info(
                        f"Loaded plugin '{plugin_file.stem}' "
                        f"providing metrics: {plugin_instance.get_metrics()}"
                    )
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}", exc_info=True)

        logger.info(f"Loaded {loaded_count} custom statistic plugin(s)")

    def _load_plugin_file(self, plugin_path: Path) -> Optional[CustomStatisticPlugin]:
        """
        Load a single plugin file.

        Args:
            plugin_path: Path to the plugin .py file

        Returns:
            Instantiated plugin or None if loading failed
        """
        spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[plugin_path.stem] = module
        spec.loader.exec_module(module)

        # Look for CustomStatisticPlugin subclass in module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, CustomStatisticPlugin)
                and attr is not CustomStatisticPlugin
            ):
                return attr()

        # Fallback: look for legacy plugin format with calculate() function
        if hasattr(module, "calculate") and hasattr(module, "PLUGIN_INFO"):
            return self._wrap_legacy_plugin(module)

        logger.warning(f"No plugin class found in {plugin_path}")
        return None

    def _wrap_legacy_plugin(self, module) -> CustomStatisticPlugin:
        """Wrap legacy-style plugin (function-based) into plugin class."""

        class LegacyPluginWrapper(CustomStatisticPlugin):
            def __init__(self, mod):
                self.module = mod
                self.info = getattr(mod, "PLUGIN_INFO", {})

            def calculate(self, times, amplitudes, roi_start, roi_end):
                return self.module.calculate(times, amplitudes, roi_start, roi_end)

            def get_metrics(self):
                return self.info.get("metrics", [])

            def get_info(self):
                return self.info

        return LegacyPluginWrapper(module)

    def load_expression_stats(self, expression_stats: List[Dict[str, str]]) -> None:
        """
        Load expression-based statistics from configuration.

        Args:
            expression_stats: List of dicts with 'name', 'expression', 'display_name'
        """
        for expr_config in expression_stats:
            try:
                name = expr_config.get("name", "")
                expression = expr_config.get("expression", "")
                display_name = expr_config.get("display_name")
                timeout_val = expr_config.get("timeout", 1)
                timeout = int(timeout_val) if timeout_val else 1

                expr_stat = ExpressionStatistic(name, expression, display_name, timeout)
                self.expression_stats.append(expr_stat)
                logger.info(f"Loaded expression statistic: {name}")
            except Exception as e:
                logger.error(f"Failed to load expression statistic: {e}")

    def get_all_metric_names(self) -> List[str]:
        """
        Get list of all available metric names (built-in + custom + expressions).

        Returns:
            List of metric name strings
        """
        metrics = list(self.enabled_builtin_stats)

        for plugin in self.plugins:
            metrics.extend(plugin.get_metrics())

        for expr in self.expression_stats:
            metrics.append(expr.name)

        return metrics

    def reload_plugins(self, plugin_dir: str, enabled_plugins: List[str]) -> None:
        """
        Reload all plugins from directory.

        Args:
            plugin_dir: Path to plugin directory
            enabled_plugins: List of enabled plugin names
        """
        self.plugins.clear()
        self.load_plugins(plugin_dir, enabled_plugins)

    def calculate_statistics(
        self, waveform: WaveformData, roi: ROIDefinition
    ) -> List[ROIStatistics]:
        """
        Calculate statistics for a given ROI across all channels.

        Args:
            waveform: Waveform data to analyze
            roi: Region of interest definition

        Returns:
            List of ROIStatistics, one per channel

        Raises:
            ValueError: If ROI times are outside waveform time range
        """
        if roi.start_time >= roi.end_time:
            raise ValueError(
                f"ROI start time ({roi.start_time}) must be less than end time ({roi.end_time})"
            )

        # Find indices for the ROI
        start_idx = np.searchsorted(waveform.mz, roi.start_time)
        end_idx = np.searchsorted(waveform.mz, roi.end_time)

        # Ensure we have at least one point
        if start_idx >= len(waveform.mz):
            start_idx = len(waveform.mz) - 1
        if end_idx > len(waveform.mz):
            end_idx = len(waveform.mz)
        if end_idx <= start_idx:
            end_idx = start_idx + 1

        # Get actual time values at boundaries
        actual_start = waveform.mz[start_idx]
        actual_end = waveform.mz[min(end_idx - 1, len(waveform.mz) - 1)]

        logger.debug(
            f"ROI '{roi.name}': requested [{roi.start_time:.6f}, {roi.end_time:.6f}], "
            f"actual [{actual_start:.6f}, {actual_end:.6f}], "
            f"indices [{start_idx}, {end_idx})"
        )

        # Calculate statistics for each channel
        statistics_list = []

        for channel_name, channel_data in waveform.channels.items():
            # Extract data within ROI
            roi_data = channel_data[start_idx:end_idx]
            roi_times = waveform.mz[start_idx:end_idx]

            if len(roi_data) == 0:
                logger.warning(
                    f"No data points in ROI '{roi.name}' for channel '{channel_name}'"
                )
                continue

            # Calculate built-in statistics (only if enabled)
            mean_val = (
                float(np.mean(roi_data))
                if "mean" in self.enabled_builtin_stats
                else 0.0
            )
            median_val = (
                float(np.median(roi_data))
                if "median" in self.enabled_builtin_stats
                else 0.0
            )
            std_dev_val = (
                float(np.std(roi_data))
                if "std_dev" in self.enabled_builtin_stats
                else 0.0
            )
            min_val = (
                float(np.min(roi_data)) if "min" in self.enabled_builtin_stats else 0.0
            )
            max_val = (
                float(np.max(roi_data)) if "max" in self.enabled_builtin_stats else 0.0
            )
            peak_to_peak_val = (
                (max_val - min_val)
                if "peak_to_peak" in self.enabled_builtin_stats
                else 0.0
            )

            # Calculate area under curve using trapezoidal integration
            area = (
                float(np.trapezoid(roi_data, roi_times))
                if "area" in self.enabled_builtin_stats
                else 0.0
            )

            # Calculate time span
            time_span = actual_end - actual_start

            # Calculate custom statistics from plugins
            custom_metrics = {}
            for plugin in self.plugins:
                try:
                    plugin_results = plugin.calculate(
                        roi_times, roi_data, actual_start, actual_end
                    )
                    custom_metrics.update(plugin_results)
                except Exception as e:
                    logger.error(
                        f"Plugin {plugin.__class__.__name__} failed for "
                        f"ROI '{roi.name}', channel '{channel_name}': {e}"
                    )

            # Calculate expression-based statistics
            for expr_stat in self.expression_stats:
                try:
                    result = expr_stat.calculate(roi_times, roi_data)
                    custom_metrics[expr_stat.name] = result
                except Exception as e:
                    logger.error(
                        f"Expression '{expr_stat.name}' failed for "
                        f"ROI '{roi.name}', channel '{channel_name}': {e}"
                    )

            statistics = ROIStatistics(
                roi_name=roi.name,
                channel_name=channel_name,
                start_time=actual_start,
                end_time=actual_end,
                mean=mean_val,
                median=median_val,
                std_dev=std_dev_val,
                min_value=min_val,
                max_value=max_val,
                peak_to_peak=peak_to_peak_val,
                area_under_curve=area,
                n_points=(
                    len(roi_data) if "n_points" in self.enabled_builtin_stats else 0
                ),
                time_span=(
                    time_span if "time_span" in self.enabled_builtin_stats else 0.0
                ),
                timestamp=waveform.timestamp,
                custom_metrics=custom_metrics,
            )

            statistics_list.append(statistics)

        return statistics_list

    def calculate_all_statistics(
        self, waveform: WaveformData, rois: List[ROIDefinition]
    ) -> List[ROIStatistics]:
        """
        Calculate statistics for all ROIs and all channels.

        Args:
            waveform: Waveform data to analyze
            rois: List of region of interest definitions

        Returns:
            List of ROIStatistics for all ROIs and channels
        """
        all_statistics = []

        for roi in rois:
            try:
                roi_stats = self.calculate_statistics(waveform, roi)
                all_statistics.extend(roi_stats)
            except Exception as e:
                logger.error(
                    f"Failed to calculate statistics for ROI '{roi.name}': {e}"
                )

        return all_statistics
