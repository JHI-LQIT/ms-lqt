"""
Main entry point for the MQTT Oscilloscope Waveform Viewer application.

This script initializes the application, sets up logging, creates the MQTT client
and GUI, and starts the Qt event loop.
"""

import logging
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from ms_viewer.gui import MainWindow
from ms_viewer.logging_conf import setup_logging
from ms_viewer.mqtt_client import MqttClient
from ms_viewer.state_manager import get_default_state_path

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    # Set up logging
    log_file = Path.home() / ".waveform_viewer.log"
    setup_logging(log_level=logging.INFO, log_file=str(log_file))

    logger.info("Starting MQTT Oscilloscope Waveform Viewer")

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("MQTT Oscilloscope Waveform Viewer")

    # Get state file path
    state_path = get_default_state_path()
    logger.info(f"Using state file: {state_path}")

    # Create MQTT client
    mqtt_client = MqttClient()

    # Create main window
    main_window = MainWindow(mqtt_client, state_path)
    main_window.show()

    logger.info("Application started successfully")

    # Start Qt event loop
    exit_code = app.exec_()

    logger.info(f"Application exiting with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
