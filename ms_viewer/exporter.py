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
    - Comment lines (starting with #) containing metadata
    - First data row: header with column names (mz, ch1, ch2, ...)
    - Subsequent rows: mz value followed by amplitude for each channel

    Args:
        path: Path to the output CSV file (.csv extension added automatically if missing)
        waveform: WaveformData object to export

    Returns:
        True if successful, False otherwise
    """
    try:
        # Add .csv extension if not present
        if not path.lower().endswith('.csv'):
            path = path + '.csv'
        
        with open(path, "w", newline="") as csvfile:
            # Write metadata as a single comment line with # as delimiter
            step = waveform.trapping_parameters.excitation_steps[
                waveform.selected_excitation_step
            ]
            basic = waveform.trapping_parameters.basic_controls

            metadata = (
                f"# Timestamp: {waveform.timestamp.isoformat()} "
                f"# Selected Excitation Step: {waveform.selected_excitation_step} "
                f"# Step Name: {step.step_name} "
                f"# Step Active: {step.step_active} "
                f"# Step Delay (ms): {step.step_delay_ms} "
                f"# Step Duration (ms): {step.step_duration_ms} "
                f"# Frequency Sweep Enabled: {step.enable_frequency_sweep} "
                f"# Frequency Begin (kHz): {step.frequency_begin_kHz} "
                f"# Frequency End (kHz): {step.frequency_end_kHz} "
                f"# AC Amplitude Sweep Enabled: {step.enable_ac_amplitude_sweep} "
                f"# AC Amplitude Begin (mV): {step.ac_amplitude_begin_mV} "
                f"# AC Amplitude End (mV): {step.ac_amplitude_end_mV} "
                f"# AC Soft Start Enabled: {step.enable_ac_amplitude_soft_start} "
                f"# AC Soft Start Duration (ms): {step.ac_soft_start_duration_ms} "
                f"# RF Amplitude Sweep Enabled: {step.enable_rf_amplitude_sweep} "
                f"# RF Amplitude Begin (V0p): {step.rf_amplitude_begin_V0p} "
                f"# RF Amplitude End (V0p): {step.rf_amplitude_end_V0p} "
                f"# Trap r0 (cm): {basic.r0_cm} "
                f"# RF Frequency (MHz): {basic.f_RF_MHz} "
                f"# RF Amplitude (V0p): {basic.rf_amplitude_V0p} "
                f"# Injection Duration (ms): {basic.injection_duration_ms} "
                f"# Injection Open Voltage (V): {basic.injection_open_voltage_V} "
                f"# Injection Close Voltage (V): {basic.injection_close_voltage_V} "
                f"# Injection RF Amplitude (V): {basic.injection_rf_amplitude_V} "
                f"# Trapping Duration (ms): {basic.trapping_duration_ms} "
                f"# Trapping RF Amplitude (V0p): {basic.trapping_rf_amplitude_V0p} "
                f"# t0 (s): {waveform.t0} "
                f"# dt (s): {waveform.dt} "
                f"# Number of Samples: {len(waveform.mz)}\n"
            )
            csvfile.write(metadata)

            # Get sorted channel names for consistent ordering
            channel_names = sorted(waveform.channels.keys())

            # Create header
            header = ["mz"] + channel_names

            writer = csv.writer(csvfile)
            writer.writerow(header)

            # Write data rows
            for i, mz_val in enumerate(waveform.mz):
                row = [mz_val]
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
