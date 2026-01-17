"""
Unit tests for data_model.py
"""

import json
from datetime import datetime

import numpy as np
import pytest

from waveform_viewer.data_model import PayloadParseError, WaveformData, parse_payload


class TestWaveformData:
    """Tests for WaveformData class."""

    def test_valid_creation(self):
        """Test creating a valid WaveformData object."""
        channels = {"ch1": np.array([1.0, 2.0, 3.0]), "ch2": np.array([4.0, 5.0, 6.0])}
        timestamp = datetime.now()

        wf = WaveformData(t0=0.0, dt=1e-3, channels=channels, timestamp=timestamp)

        assert wf.t0 == 0.0
        assert wf.dt == 1e-3
        assert len(wf.channels) == 2
        assert len(wf.times) == 3
        assert np.allclose(wf.times, [0.0, 1e-3, 2e-3])

    def test_invalid_dt(self):
        """Test that negative or zero dt raises ValueError."""
        channels = {"ch1": np.array([1.0, 2.0])}
        timestamp = datetime.now()

        with pytest.raises(ValueError, match="dt must be positive"):
            WaveformData(t0=0.0, dt=0.0, channels=channels, timestamp=timestamp)

        with pytest.raises(ValueError, match="dt must be positive"):
            WaveformData(t0=0.0, dt=-1e-3, channels=channels, timestamp=timestamp)

    def test_empty_channels(self):
        """Test that empty channels dict raises ValueError."""
        timestamp = datetime.now()

        with pytest.raises(ValueError, match="At least one channel"):
            WaveformData(t0=0.0, dt=1e-3, channels={}, timestamp=timestamp)

    def test_mismatched_channel_lengths(self):
        """Test that mismatched channel lengths raise ValueError."""
        channels = {
            "ch1": np.array([1.0, 2.0, 3.0]),
            "ch2": np.array([4.0, 5.0]),  # Different length
        }
        timestamp = datetime.now()

        with pytest.raises(ValueError, match="equal length"):
            WaveformData(t0=0.0, dt=1e-3, channels=channels, timestamp=timestamp)


class TestParsePayload:
    """Tests for parse_payload function."""

    def test_valid_payload(self):
        """Test parsing a valid payload."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": [0.0, 0.1, 0.2], "ch2": [-0.1, 0.0, 0.1]},
            "timestamp": "2025-01-01T12:34:56.789Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        wf = parse_payload(payload_bytes)

        assert wf.t0 == 0.0
        assert wf.dt == 1e-6
        assert len(wf.channels) == 2
        assert "ch1" in wf.channels
        assert "ch2" in wf.channels
        assert len(wf.times) == 3
        assert wf.timestamp.year == 2025

    def test_invalid_utf8(self):
        """Test that invalid UTF-8 raises PayloadParseError."""
        invalid_bytes = b"\x80\x81\x82"

        with pytest.raises(PayloadParseError, match="UTF-8"):
            parse_payload(invalid_bytes)

    def test_invalid_json(self):
        """Test that invalid JSON raises PayloadParseError."""
        invalid_json = b'{"invalid": json}'

        with pytest.raises(PayloadParseError, match="JSON"):
            parse_payload(invalid_json)

    def test_missing_fields(self):
        """Test that missing required fields raise PayloadParseError."""
        # Missing dt
        payload = {
            "t0": 0.0,
            "channels": {"ch1": [1.0, 2.0]},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="Missing required fields"):
            parse_payload(payload_bytes)

        # Missing channels
        payload = {"t0": 0.0, "dt": 1e-6, "timestamp": "2025-01-01T12:34:56Z"}
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="Missing required fields"):
            parse_payload(payload_bytes)

    def test_invalid_t0(self):
        """Test that invalid t0 raises PayloadParseError."""
        payload = {
            "t0": "not a number",
            "dt": 1e-6,
            "channels": {"ch1": [1.0, 2.0]},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="Invalid t0"):
            parse_payload(payload_bytes)

    def test_invalid_dt(self):
        """Test that invalid dt raises PayloadParseError."""
        # Non-numeric dt
        payload = {
            "t0": 0.0,
            "dt": "not a number",
            "channels": {"ch1": [1.0, 2.0]},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="Invalid dt"):
            parse_payload(payload_bytes)

        # Negative dt
        payload = {
            "t0": 0.0,
            "dt": -1e-6,
            "channels": {"ch1": [1.0, 2.0]},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="dt must be positive"):
            parse_payload(payload_bytes)

    def test_empty_channels(self):
        """Test that empty channels dict raises PayloadParseError."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="empty"):
            parse_payload(payload_bytes)

    def test_channels_not_dict(self):
        """Test that channels must be a dictionary."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": [1.0, 2.0, 3.0],
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="must be a dictionary"):
            parse_payload(payload_bytes)

    def test_channel_not_list(self):
        """Test that channel data must be a list."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": "not a list"},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="must be a list"):
            parse_payload(payload_bytes)

    def test_mismatched_channel_lengths(self):
        """Test that channels with different lengths raise PayloadParseError."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": [1.0, 2.0, 3.0], "ch2": [4.0, 5.0]},  # Different length
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="equal length"):
            parse_payload(payload_bytes)

    def test_invalid_timestamp(self):
        """Test that invalid timestamp raises PayloadParseError."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": [1.0, 2.0]},
            "timestamp": "not a valid timestamp",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(PayloadParseError, match="timestamp"):
            parse_payload(payload_bytes)

    def test_timestamp_formats(self):
        """Test various valid timestamp formats."""
        base_payload = {"t0": 0.0, "dt": 1e-6, "channels": {"ch1": [1.0, 2.0]}}

        # With microseconds and Z
        payload = {**base_payload, "timestamp": "2025-01-01T12:34:56.789Z"}
        wf = parse_payload(json.dumps(payload).encode("utf-8"))
        assert wf.timestamp.microsecond > 0

        # Without microseconds
        payload = {**base_payload, "timestamp": "2025-01-01T12:34:56Z"}
        wf = parse_payload(json.dumps(payload).encode("utf-8"))
        assert wf.timestamp.second == 56
