"""
State manager for persisting application configuration and GUI state.

This module handles loading and saving application state to a YAML file,
including MQTT configuration, GUI state, and user preferences.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_state(path: str) -> Dict[str, Any]:
    """
    Load application state from a YAML file.

    Args:
        path: Path to the state file

    Returns:
        Dictionary containing application state. Returns default state if file
        doesn't exist or is corrupted.
    """
    default_state = {
        "mqtt": {
            "broker_host": "localhost",
            "broker_port": 1883,
            "input_topic": "lqt/picoscope/waveform",
            "output_topic": "lqt/lqt_ms/republish",
            "statistics_topic": "lqt/lqt_ms/statistics",
            "trapping_parameters_topic": "lqt/trapping/parameters",
            "client_id": "ms_viewer",
            "keepalive": 60,
            # Secondary broker for output/statistics publishing
            "output_broker_enabled": False,
            "output_broker_host": "localhost",
            "output_broker_port": 1883,
            "output_broker_output_topic": "lqt/lqt_ms/republish",
            "output_broker_statistics_topic": "lqt/lqt_ms/statistics",
        },
        "gui": {
            "regions": [],  # List of {'start': float, 'end': float, 'name': str}
            "window_geometry": None,  # {'x': int, 'y': int, 'width': int, 'height': int}
            "splitter_sizes": None,  # [int, int] - sizes of waveform and statistics plots
            "selected_metrics": [
                "mean"
            ],  # List of metric names to display in statistics plot
        },
        "preferences": {
            "last_export_dir": str(Path.home()),
            "max_history_points": 100,
            "channel_colors": [
                [255, 0, 0],  # Red
                [0, 255, 0],  # Green
                [0, 0, 255],  # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
                [255, 128, 0],  # Orange
                [128, 0, 255],  # Purple
            ],
            "statistics_colors": [
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
        },
        "statistics": {
            "plugin_directory": str(Path.home() / ".waveform_viewer" / "plugins"),
            "enabled_plugins": [],  # List of plugin names (without .py) to load
            "builtin_statistics": [  # List of built-in statistics to calculate
                "mean",
                "median",
                "std_dev",
                "min",
                "max",
                "peak_to_peak",
                "area",
                "n_points",
                "time_span",
            ],
            "expression_statistics": [],  # List of expression-based statistics
            # Example expression statistic:
            # - name: "rms"
            #   display_name: "RMS Value"
            #   expression: "np.sqrt(np.mean(data**2))"
            #   timeout: 1
        },
        "channel_scaling": {
            # Per-channel scaling parameters
            # Scaling formula: y := Gain * (y - Offset)
            # Example:
            # "ch1": {"gain": 1.0, "offset": 0.0}
            # "ch2": {"gain": 2.0, "offset": 0.5}
        },
    }

    path_obj = Path(path)

    # If file doesn't exist, return default state
    if not path_obj.exists():
        logger.info(f"State file {path} does not exist, using default state")
        return default_state

    try:
        with open(path_obj, "r") as f:
            state = yaml.safe_load(f)

        # Validate that we got a dictionary
        if not isinstance(state, dict):
            logger.warning(
                f"State file {path} contains invalid data (not a dict), using default state"
            )
            return default_state

        # Merge with defaults to ensure all required keys exist
        merged_state = default_state.copy()

        # Merge MQTT config
        if "mqtt" in state and isinstance(state["mqtt"], dict):
            merged_state["mqtt"].update(state["mqtt"])

        # Merge GUI state
        if "gui" in state and isinstance(state["gui"], dict):
            merged_state["gui"].update(state["gui"])

        # Merge preferences
        if "preferences" in state and isinstance(state["preferences"], dict):
            merged_state["preferences"].update(state["preferences"])

        # Merge statistics config
        if "statistics" in state and isinstance(state["statistics"], dict):
            merged_state["statistics"].update(state["statistics"])

        # Merge channel scaling config
        if "channel_scaling" in state and isinstance(state["channel_scaling"], dict):
            merged_state["channel_scaling"].update(state["channel_scaling"])

        logger.info(f"Successfully loaded state from {path}")
        return merged_state

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML from {path}: {e}")
        return default_state
    except Exception as e:
        logger.error(f"Failed to load state from {path}: {e}")
        return default_state


def save_state(path: str, state_dict: Dict[str, Any]) -> bool:
    """
    Save application state to a YAML file.

    Args:
        path: Path to the state file
        state_dict: Dictionary containing application state

    Returns:
        True if successful, False otherwise
    """
    path_obj = Path(path)

    try:
        # Create parent directories if they don't exist
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write state to file
        with open(path_obj, "w") as f:
            yaml.safe_dump(state_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Successfully saved state to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save state to {path}: {e}")
        return False


def get_default_state_path() -> str:
    """
    Get the default path for the state file.

    Returns:
        Path to the state file in the user's home directory
    """
    return str(Path.home() / ".waveform_viewer_config.yaml")
