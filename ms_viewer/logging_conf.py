"""
Logging configuration for the waveform viewer application.

This module sets up logging to both console and file.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file. If None, only console logging is used.
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Failed to set up file logging to {log_file}: {e}")

    # Reduce verbosity of some third-party loggers
    logging.getLogger("paho").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
