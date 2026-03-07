"""
Data model for waveform data received via MQTT.

This module handles parsing and validation of MQTT payloads containing
oscilloscope waveform data.

Binary format specification:
- Header (36 bytes):
  * 4 bytes: Magic number (0x57415645 = "WAVE")
  * 8 bytes: t0 (float64)
  * 8 bytes: dt (float64)
  * 8 bytes: timestamp (Unix time in microseconds, int64)
  * 4 bytes: number of channels (uint32)
  * 4 bytes: number of samples per channel (uint32)
- For each channel:
  * 1 byte: length of channel name (uint8)
  * N bytes: channel name (UTF-8 encoded)
- Data section:
  * All channel data interleaved: [ch1[0], ch2[0], ..., ch1[1], ch2[1], ...]
  * Each value is 8 bytes (float64)
"""

import json
import struct
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .trapping_parameters import TrapControls
from .waveform_data import WaveformData


class PayloadParseError(Exception):
    """Exception raised when payload parsing or validation fails."""

    pass


def parse_trapping_parameters(payload_bytes: bytes) -> TrapControls:
    """
    Parse MQTT payload into TrapControls object.

    Args:
        payload_bytes (bytes): JSON-encoded payload

    Returns:
        TrapControls: Parsed trapping parameters

    Raises:
        PayloadParseError: If payload is malformed or validation fails

    Expected JSON format (new structure):
    {
        "basic_controls": {
            "r0_cm": 0.475,
            "f_RF_MHz": 2.112,
            "injection_duration_ms": 5.0,
            ...
        },
        "excitation_steps": [
            {
                "step_name": "Step1",
                "step_active": true,
                "step_duration_ms": 20.0,
                ...
            }
        ]
    }

    Also supports legacy flat format for backward compatibility.
    """

    try:
        # Decode UTF-8 to string
        payload_str = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise PayloadParseError(f"Failed to decode payload as UTF-8: {e}")

    try:
        # Parse JSON
        data = json.loads(payload_str)
    except json.JSONDecodeError as e:
        raise PayloadParseError(f"Failed to parse JSON: {e}")

    # Check if this is the new nested format or legacy flat format
    if "basic_controls" in data or "excitation_steps" in data:
        # New nested format
        try:
            return TrapControls.from_dict(data)
        except (ValueError, TypeError) as e:
            raise PayloadParseError(f"Invalid parameter value in new format: {e}")
    else:
        # Legacy flat format - convert to new structure
        # Validate required fields for legacy format
        required_fields = [
            "enable_trapping",
            "r0_cm",
            "f_RF_MHz",
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise PayloadParseError(f"Missing required fields: {missing_fields}")

        try:
            return TrapControls.from_flat_dict(data)
        except (ValueError, TypeError) as e:
            raise PayloadParseError(f"Invalid parameter value in legacy format: {e}")


def parse_payload(
    payload_bytes: bytes,
    trapping_parameters: TrapControls,
    selected_excitation_step: int = 0,
    collector_metadata: Optional[str] = None,
) -> WaveformData:
    """
    Parse MQTT payload bytes into a WaveformData object.

    Automatically detects binary or JSON format:
    - Binary format starts with magic number 0x57415645 ("WAVE")
    - JSON format starts with '{'

    Binary format is preferred for efficiency. JSON format is supported
    for backward compatibility.

    Args:
        payload_bytes: Binary or UTF-8 encoded JSON payload
        trapping_parameters: TrapControls with basic controls and excitation steps
        selected_excitation_step: Index of the excitation step to use (default: 0)
        collector_metadata: Optional JSON metadata from collector parameters topic

    Returns:
        WaveformData: Parsed waveform data

    Raises:
        PayloadParseError: If payload is malformed or validation fails
    """
    # Check if payload is binary or JSON
    if len(payload_bytes) >= 4 and payload_bytes[:4] == b"WAVE":
        return _parse_binary_payload(
            payload_bytes,
            trapping_parameters,
            selected_excitation_step,
            collector_metadata,
        )
    else:
        return _parse_json_payload(
            payload_bytes,
            trapping_parameters,
            selected_excitation_step,
            collector_metadata,
        )


def _parse_json_payload(
    payload_bytes: bytes,
    trapping_parameters: TrapControls,
    selected_excitation_step: int = 0,
    collector_metadata: Optional[str] = None,
) -> WaveformData:
    """
    Parse JSON-encoded MQTT payload.

    Expected JSON format:
    {
      "t0": 0.0,
      "dt": 1e-6,
      "channels": {
        "ch1": [0.0, 0.1, ...],
        "ch2": [-0.01, 0.02, ...]
      },
      "timestamp": "2025-01-01T12:34:56.789Z"
    }

    Args:
        payload_bytes: UTF-8 encoded JSON payload
        trapping_parameters: TrapControls with basic controls and excitation steps
        selected_excitation_step: Index of the excitation step to use (default: 0)
        collector_metadata: Optional JSON metadata from collector parameters topic

    Returns:
        WaveformData: Parsed waveform data

    Raises:
        PayloadParseError: If payload is malformed or validation fails
    """
    try:
        # Decode UTF-8 to string
        payload_str = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise PayloadParseError(f"Failed to decode payload as UTF-8: {e}")

    try:
        # Parse JSON
        data = json.loads(payload_str)
    except json.JSONDecodeError as e:
        raise PayloadParseError(f"Failed to parse JSON: {e}")

    # Validate required fields
    required_fields = ["t0", "dt", "channels", "timestamp"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise PayloadParseError(f"Missing required fields: {missing_fields}")

    # Extract and validate t0
    try:
        t0 = float(data["t0"])
    except (ValueError, TypeError) as e:
        raise PayloadParseError(f"Invalid t0 value: {e}")

    # Extract and validate dt
    try:
        dt = float(data["dt"])
        if dt <= 0:
            raise PayloadParseError(f"dt must be positive, got {dt}")
    except (ValueError, TypeError) as e:
        raise PayloadParseError(f"Invalid dt value: {e}")

    # Extract and validate channels
    channels_data = data["channels"]
    if not isinstance(channels_data, dict):
        raise PayloadParseError("channels must be a dictionary")

    if not channels_data:
        raise PayloadParseError("channels dictionary is empty")

    # Convert channel data to numpy arrays and validate
    channels = {}
    channel_lengths = {}

    for channel_name, channel_values in channels_data.items():
        if not isinstance(channel_values, list):
            raise PayloadParseError(f"Channel '{channel_name}' data must be a list")

        try:
            channel_array = np.array(channel_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise PayloadParseError(
                f"Failed to convert channel '{channel_name}' to numeric array: {e}"
            )

        if channel_array.size == 0:
            raise PayloadParseError(f"Channel '{channel_name}' is empty")

        channels[channel_name] = channel_array
        channel_lengths[channel_name] = len(channel_array)

    # Verify all channels have the same length
    unique_lengths = set(channel_lengths.values())
    if len(unique_lengths) > 1:
        raise PayloadParseError(
            f"All channels must have equal length. Got: {channel_lengths}"
        )

    # Parse timestamp
    timestamp_str = data["timestamp"]
    try:
        # Parse ISO-8601 format
        # Handle both with and without microseconds
        if "." in timestamp_str:
            # Has microseconds
            if timestamp_str.endswith("Z"):
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        else:
            # No microseconds
            if timestamp_str.endswith("Z"):
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
            else:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as e:
        raise PayloadParseError(f"Failed to parse timestamp '{timestamp_str}': {e}")

    # Create and return WaveformData
    try:
        return WaveformData(
            t0=t0,
            dt=dt,
            channels=channels,
            timestamp=timestamp,
            trapping_parameters=trapping_parameters,
            selected_excitation_step=selected_excitation_step,
            collector_metadata=collector_metadata,
        )
    except ValueError as e:
        raise PayloadParseError(f"Failed to create WaveformData: {e}")


def _parse_binary_payload(
    payload_bytes: bytes,
    trapping_parameters: TrapControls,
    selected_excitation_step: int = 0,
    collector_metadata: Optional[str] = None,
) -> WaveformData:
    """
    Parse binary-encoded MQTT payload.

    Binary format:
    - Header (36 bytes):
      * 4 bytes: Magic number (0x57415645 = "WAVE")
      * 8 bytes: t0 (float64)
      * 8 bytes: dt (float64)
      * 8 bytes: timestamp (Unix time in microseconds, int64)
      * 4 bytes: number of channels (uint32)
      * 4 bytes: number of samples per channel (uint32)
    - Channel names:
      * For each channel:
        - 1 byte: length of channel name (uint8)
        - N bytes: channel name (UTF-8)
    - Data section:
      * All channel data interleaved as float64

    Args:
        payload_bytes: Binary encoded payload
        trapping_parameters: TrapControls with basic controls and excitation steps
        selected_excitation_step: Index of the excitation step to use (default: 0)
        collector_metadata: Optional JSON metadata from collector parameters topic

    Returns:
        WaveformData: Parsed waveform data

    Raises:
        PayloadParseError: If payload is malformed or validation fails
    """
    try:
        # Parse header (36 bytes)
        if len(payload_bytes) < 36:
            raise PayloadParseError(f"Payload too short: {len(payload_bytes)} bytes")

        # Unpack header
        magic, t0, dt, timestamp_us, n_channels, n_samples = struct.unpack(
            "<4sddQII", payload_bytes[:36]
        )

        # Verify magic number
        if magic != b"WAVE":
            raise PayloadParseError(f"Invalid magic number: {magic}")

        # Validate parameters
        if dt <= 0:
            raise PayloadParseError(f"dt must be positive, got {dt}")
        if n_channels == 0:
            raise PayloadParseError("Number of channels must be > 0")
        if n_samples == 0:
            raise PayloadParseError("Number of samples must be > 0")

        # Convert timestamp from microseconds to datetime
        timestamp = datetime.fromtimestamp(timestamp_us / 1e6, tz=timezone.utc)

        # Parse channel names
        offset = 36
        channel_names = []
        for _ in range(n_channels):
            if offset >= len(payload_bytes):
                raise PayloadParseError(
                    "Unexpected end of payload while reading channel names"
                )

            name_len = struct.unpack("<B", payload_bytes[offset : offset + 1])[0]
            offset += 1

            if offset + name_len > len(payload_bytes):
                raise PayloadParseError(
                    "Unexpected end of payload while reading channel name"
                )

            channel_name = payload_bytes[offset : offset + name_len].decode("utf-8")
            channel_names.append(channel_name)
            offset += name_len

        # Parse data section
        expected_data_size = n_channels * n_samples * 8  # 8 bytes per float64
        if len(payload_bytes) - offset != expected_data_size:
            raise PayloadParseError(
                f"Data size mismatch: expected {expected_data_size} bytes, "
                f"got {len(payload_bytes) - offset} bytes"
            )

        # Unpack all data at once
        data_format = f"<{n_channels * n_samples}d"
        all_data = struct.unpack(data_format, payload_bytes[offset:])

        # Deinterleave data into channels
        channels = {}
        for i, channel_name in enumerate(channel_names):
            # Extract every n_channels-th value starting at index i
            channel_data = np.array(all_data[i::n_channels], dtype=np.float64)
            channels[channel_name] = channel_data

        # Create and return WaveformData
        return WaveformData(
            t0=t0,
            dt=dt,
            channels=channels,
            timestamp=timestamp,
            trapping_parameters=trapping_parameters,
            selected_excitation_step=selected_excitation_step,
            collector_metadata=collector_metadata,
        )

    except struct.error as e:
        raise PayloadParseError(f"Binary unpacking error: {e}")
    except UnicodeDecodeError as e:
        raise PayloadParseError(f"Failed to decode channel name: {e}")
    except Exception as e:
        raise PayloadParseError(f"Failed to parse binary payload: {e}")


def encode_payload(waveform: WaveformData) -> bytes:
    """
    Encode WaveformData into binary format for MQTT transmission.

    Binary format:
    - Header (36 bytes):
      * 4 bytes: Magic number (0x57415645 = "WAVE")
      * 8 bytes: t0 (float64)
      * 8 bytes: dt (float64)
      * 8 bytes: timestamp (Unix time in microseconds, int64)
      * 4 bytes: number of channels (uint32)
      * 4 bytes: number of samples per channel (uint32)
    - Channel names:
      * For each channel:
        - 1 byte: length of channel name (uint8)
        - N bytes: channel name (UTF-8)
    - Data section:
      * All channel data interleaved as float64

    Args:
        waveform: WaveformData object to encode

    Returns:
        Binary encoded payload bytes
    """
    # Get sorted channel names for consistent ordering
    channel_names = sorted(waveform.channels.keys())
    n_channels = len(channel_names)
    n_samples = len(waveform.mz)

    # Convert timestamp to Unix time in microseconds
    timestamp_us = int(waveform.timestamp.timestamp() * 1e6)

    # Build header
    header = struct.pack(
        "<4sddQII",
        b"WAVE",  # Magic number
        waveform.t0,  # t0 (float64)
        waveform.dt,  # dt (float64)
        timestamp_us,  # timestamp (int64, microseconds)
        n_channels,  # number of channels (uint32)
        n_samples,  # number of samples (uint32)
    )

    # Build channel names section
    channel_names_bytes = b""
    for channel_name in channel_names:
        name_bytes = channel_name.encode("utf-8")
        if len(name_bytes) > 255:
            raise ValueError(f"Channel name too long: {channel_name}")
        channel_names_bytes += struct.pack("<B", len(name_bytes)) + name_bytes

    # Interleave channel data
    interleaved_data = []
    for i in range(n_samples):
        for channel_name in channel_names:
            interleaved_data.append(float(waveform.channels[channel_name][i]))

    # Pack data section
    data_format = f"<{len(interleaved_data)}d"
    data_bytes = struct.pack(data_format, *interleaved_data)

    # Combine all sections
    return header + channel_names_bytes + data_bytes
