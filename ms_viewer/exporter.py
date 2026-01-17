"""
Exporter module for saving waveform data and building MQTT payloads.

This module provides functions for exporting waveforms to CSV format
and building MQTT payloads for republishing.
"""

import csv
import json
import logging
from typing import Dict

from .data_model import WaveformData, encode_payload

logger = logging.getLogger(__name__)


def export_waveform_to_csv(path: str, waveform: WaveformData) -> bool:
    """
    Export waveform data to a CSV file.

    The CSV format is:
    - First row: header with column names (time, ch1, ch2, ...)
    - Subsequent rows: time value followed by amplitude for each channel

    Args:
        path: Path to the output CSV file
        waveform: WaveformData object to export

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(path, "w", newline="") as csvfile:
            # Get sorted channel names for consistent ordering
            channel_names = sorted(waveform.channels.keys())

            # Create header
            header = ["time"] + channel_names

            writer = csv.writer(csvfile)
            writer.writerow(header)

            # Write data rows
            for i, time_val in enumerate(waveform.times):
                row = [time_val]
                for ch_name in channel_names:
                    row.append(waveform.channels[ch_name][i])
                writer.writerow(row)

        logger.info(f"Successfully exported waveform to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export waveform to {path}: {e}")
        return False


def build_payload(waveform: WaveformData) -> bytes:
    """
    Build an MQTT payload from a WaveformData object.

    The payload uses a compact binary format for efficient transmission:
    - Header with metadata (t0, dt, timestamp, channel count, sample count)
    - Channel names
    - Interleaved channel data as float64

    This is significantly more efficient than JSON encoding, reducing
    payload size by ~40-60% for typical waveforms.

    Args:
        waveform: WaveformData object to convert

    Returns:
        Binary encoded payload bytes
    """
    return encode_payload(waveform)
