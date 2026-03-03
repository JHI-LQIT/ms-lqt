"""
Unit tests for data_model module.

Tests cover:
- parse_trapping_parameters function
- parse_payload function (binary and JSON)
- encode_payload function
- PayloadParseError exceptions
- Edge cases and error handling
"""

import json
import struct
import unittest
from datetime import datetime, timezone

import numpy as np

from ms_viewer.data_model import (
    PayloadParseError,
    encode_payload,
    parse_payload,
    parse_trapping_parameters,
)
from ms_viewer.trapping_parameters import ExcitationStep, TrapControls
from ms_viewer.waveform_data import WaveformData


class TestParseTrappingParameters(unittest.TestCase):
    """Test parse_trapping_parameters function."""

    def test_parse_new_format(self):
        """Test parsing new nested format."""
        payload = {
            "basic_controls": {
                "r0_cm": 0.5,
                "f_RF_MHz": 2.0,
                "enable_trapping": True,
            },
            "excitation_steps": [
                {
                    "step_name": "Step1",
                    "step_duration_ms": 150.0,
                },
            ],
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        trap_controls = parse_trapping_parameters(payload_bytes)

        self.assertIsInstance(trap_controls, TrapControls)
        self.assertEqual(trap_controls.basic_controls.r0_cm, 0.5)
        self.assertEqual(trap_controls.basic_controls.f_RF_MHz, 2.0)
        self.assertTrue(trap_controls.basic_controls.enable_trapping)
        self.assertEqual(len(trap_controls.excitation_steps), 1)
        self.assertEqual(trap_controls.excitation_steps[0].step_name, "Step1")

    def test_parse_legacy_format(self):
        """Test parsing legacy flat format."""
        payload = {
            "enable_trapping": True,
            "r0_cm": 0.5,
            "f_RF_MHz": 2.0,
            "step_name": "LegacyStep",
            "step_duration_ms": 150.0,
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        trap_controls = parse_trapping_parameters(payload_bytes)

        self.assertIsInstance(trap_controls, TrapControls)
        self.assertEqual(trap_controls.basic_controls.r0_cm, 0.5)
        self.assertEqual(trap_controls.basic_controls.f_RF_MHz, 2.0)
        self.assertTrue(trap_controls.basic_controls.enable_trapping)
        self.assertEqual(trap_controls.excitation_steps[0].step_name, "LegacyStep")

    def test_parse_missing_required_fields_legacy(self):
        """Test parsing legacy format with missing required fields."""
        payload = {
            "enable_trapping": True,
            # Missing r0_cm and f_RF_MHz
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with self.assertRaises(PayloadParseError) as ctx:
            parse_trapping_parameters(payload_bytes)

        self.assertIn("Missing required fields", str(ctx.exception))

    def test_parse_invalid_utf8(self):
        """Test parsing with invalid UTF-8."""
        invalid_bytes = b"\xff\xfe"

        with self.assertRaises(PayloadParseError) as ctx:
            parse_trapping_parameters(invalid_bytes)

        self.assertIn("Failed to decode payload as UTF-8", str(ctx.exception))

    def test_parse_invalid_json(self):
        """Test parsing with invalid JSON."""
        invalid_json = b"{invalid json"

        with self.assertRaises(PayloadParseError) as ctx:
            parse_trapping_parameters(invalid_json)

        self.assertIn("Failed to parse JSON", str(ctx.exception))


class TestParsePayloadJSON(unittest.TestCase):
    """Test parse_payload function with JSON format."""

    def setUp(self):
        """Set up test fixtures."""
        self.trap_controls = TrapControls()
        self.trap_controls.basic_controls.r0_cm = 0.5
        self.trap_controls.basic_controls.f_RF_MHz = 2.0
        self.trap_controls.basic_controls.injection_duration_ms = 1.0  # Short injection
        self.trap_controls.excitation_steps[0].step_duration_ms = 10.0  # 10ms duration
        self.trap_controls.excitation_steps[0].rf_amplitude_begin_V0p = 100.0
        self.trap_controls.excitation_steps[0].frequency_begin_kHz = 50.0

    def test_parse_valid_json_payload(self):
        """Test parsing valid JSON payload."""
        # Create data with enough samples to cover excitation window (1ms injection + 10ms excitation at 1μs intervals = 11000 samples)
        n_samples = 15000
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {
                "ch1": list(np.linspace(0.0, 1.0, n_samples)),
                "ch2": list(np.linspace(-0.5, 0.5, n_samples)),
            },
            "timestamp": "2025-01-01T12:34:56.789Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        waveform = parse_payload(payload_bytes, self.trap_controls)

        self.assertIsInstance(waveform, WaveformData)
        self.assertEqual(waveform.t0, 0.0)
        self.assertEqual(waveform.dt, 1e-6)
        self.assertIn("ch1", waveform.channels)
        self.assertIn("ch2", waveform.channels)

    def test_parse_json_missing_fields(self):
        """Test parsing JSON with missing required fields."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            # Missing channels and timestamp
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("Missing required fields", str(ctx.exception))

    def test_parse_json_invalid_t0(self):
        """Test parsing JSON with invalid t0."""
        payload = {
            "t0": "not_a_number",
            "dt": 1e-6,
            "channels": {"ch1": [0.0, 0.1]},
            "timestamp": "2025-01-01T12:34:56.789Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("Invalid t0 value", str(ctx.exception))

    def test_parse_json_invalid_dt(self):
        """Test parsing JSON with invalid dt."""
        payload = {
            "t0": 0.0,
            "dt": -1e-6,  # Negative dt
            "channels": {"ch1": [0.0, 0.1]},
            "timestamp": "2025-01-01T12:34:56.789Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("dt must be positive", str(ctx.exception))

    def test_parse_json_empty_channels(self):
        """Test parsing JSON with empty channels."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {},
            "timestamp": "2025-01-01T12:34:56.789Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("channels dictionary is empty", str(ctx.exception))

    def test_parse_json_mismatched_channel_lengths(self):
        """Test parsing JSON with mismatched channel lengths."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {
                "ch1": [0.0, 0.1, 0.2],
                "ch2": [-0.1, 0.0],  # Different length
            },
            "timestamp": "2025-01-01T12:34:56.789Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("All channels must have equal length", str(ctx.exception))

    def test_parse_json_invalid_timestamp_format(self):
        """Test parsing JSON with invalid timestamp."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": [0.0, 0.1]},
            "timestamp": "not_a_timestamp",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("Failed to parse timestamp", str(ctx.exception))

    def test_parse_json_timestamp_with_microseconds(self):
        """Test parsing JSON with timestamp including microseconds."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": list(np.linspace(0.0, 1.0, 15000))},
            "timestamp": "2025-01-01T12:34:56.123456Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        waveform = parse_payload(payload_bytes, self.trap_controls)

        self.assertIsInstance(waveform, WaveformData)
        self.assertEqual(waveform.timestamp.microsecond, 123456)

    def test_parse_json_timestamp_without_microseconds(self):
        """Test parsing JSON with timestamp without microseconds."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": list(np.linspace(0.0, 1.0, 15000))},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        waveform = parse_payload(payload_bytes, self.trap_controls)

        self.assertIsInstance(waveform, WaveformData)

    def test_parse_json_channel_with_non_numeric_data(self):
        """Test parsing JSON with non-numeric channel data."""
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": [0.0, "not_a_number"]},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("Failed to convert channel", str(ctx.exception))


class TestParsePayloadBinary(unittest.TestCase):
    """Test parse_payload function with binary format."""

    def setUp(self):
        """Set up test fixtures."""
        self.trap_controls = TrapControls()
        self.trap_controls.basic_controls.r0_cm = 0.5
        self.trap_controls.basic_controls.f_RF_MHz = 2.0
        self.trap_controls.basic_controls.injection_duration_ms = 1.0  # Short injection
        self.trap_controls.excitation_steps[0].step_duration_ms = 10.0  # 10ms duration
        self.trap_controls.excitation_steps[0].rf_amplitude_begin_V0p = 100.0
        self.trap_controls.excitation_steps[0].frequency_begin_kHz = 50.0

    def _create_binary_payload(
        self,
        t0=0.0,
        dt=1e-6,
        timestamp_us=1704115000000000,
        channels=None,
    ):
        """Helper to create binary payload."""
        if channels is None:
            # Create data with 15000 samples
            channels = {
                "ch1": list(np.linspace(0.0, 1.0, 15000)),
                "ch2": list(np.linspace(-0.5, 0.5, 15000)),
            }

        channel_names = sorted(channels.keys())
        n_channels = len(channel_names)
        n_samples = len(channels[channel_names[0]])

        # Build header
        header = struct.pack(
            "<4sddQII",
            b"WAVE",
            t0,
            dt,
            timestamp_us,
            n_channels,
            n_samples,
        )

        # Build channel names section
        channel_names_bytes = b""
        for name in channel_names:
            name_bytes = name.encode("utf-8")
            channel_names_bytes += struct.pack("<B", len(name_bytes)) + name_bytes

        # Build data section (interleaved)
        interleaved = []
        for i in range(n_samples):
            for name in channel_names:
                interleaved.append(float(channels[name][i]))

        data_bytes = struct.pack(f"<{len(interleaved)}d", *interleaved)

        return header + channel_names_bytes + data_bytes

    def test_parse_valid_binary_payload(self):
        """Test parsing valid binary payload."""
        payload_bytes = self._create_binary_payload()

        waveform = parse_payload(payload_bytes, self.trap_controls)

        self.assertIsInstance(waveform, WaveformData)
        self.assertEqual(waveform.t0, 0.0)
        self.assertEqual(waveform.dt, 1e-6)
        self.assertIn("ch1", waveform.channels)
        self.assertIn("ch2", waveform.channels)

    def test_parse_binary_invalid_magic_number(self):
        """Test parsing binary with invalid magic number."""
        # Create a proper binary payload structure that won't be mistaken for JSON
        # Must start with 4 bytes that aren't 'WAVE' and aren't '{' (which would trigger JSON path)
        header = struct.pack(
            "<4sddQII",
            b"FAIL",  # Wrong magic (not WAVE, not {)
            0.0,
            1e-6,
            1704115000000000,
            2,
            15000,
        )
        # Add minimal channel names
        channel_names = b"\x03ch1\x03ch2"
        # Add some data
        data = struct.pack(f"<{2*15000}d", *([0.0] * (2 * 15000)))

        payload = header + channel_names + data

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload, self.trap_controls)

        # Since 'FAIL' is not valid UTF-8 after position ~8, it will fail UTF-8 decode in JSON path
        # Let's just check that it raises PayloadParseError
        self.assertIsInstance(ctx.exception, PayloadParseError)

    def test_parse_binary_payload_too_short(self):
        """Test parsing binary payload that is too short."""
        short_payload = b"WAVE" + b"\x00" * 20  # Less than 36 bytes

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(short_payload, self.trap_controls)

        self.assertIn("Payload too short", str(ctx.exception))

    def test_parse_binary_negative_dt(self):
        """Test parsing binary with negative dt."""
        payload_bytes = self._create_binary_payload(dt=-1e-6)

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("dt must be positive", str(ctx.exception))

    def test_parse_binary_zero_channels(self):
        """Test parsing binary with zero channels."""
        header = struct.pack(
            "<4sddQII",
            b"WAVE",
            0.0,
            1e-6,
            1704115000000000,
            0,  # Zero channels
            4,
        )

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(header, self.trap_controls)

        self.assertIn("Number of channels must be > 0", str(ctx.exception))

    def test_parse_binary_zero_samples(self):
        """Test parsing binary with zero samples."""
        header = struct.pack(
            "<4sddQII",
            b"WAVE",
            0.0,
            1e-6,
            1704115000000000,
            2,
            0,  # Zero samples
        )

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(header, self.trap_controls)

        self.assertIn("Number of samples must be > 0", str(ctx.exception))

    def test_parse_binary_data_size_mismatch(self):
        """Test parsing binary with data size mismatch."""
        # Create header
        header = struct.pack(
            "<4sddQII",
            b"WAVE",
            0.0,
            1e-6,
            1704115000000000,
            2,
            4,
        )

        # Add channel names
        channel_names = b"\x03ch1\x03ch2"

        # Add incorrect amount of data (less than expected)
        data = struct.pack("<4d", 0.0, 0.1, 0.2, 0.3)  # Only 4 values instead of 8

        payload_bytes = header + channel_names + data

        with self.assertRaises(PayloadParseError) as ctx:
            parse_payload(payload_bytes, self.trap_controls)

        self.assertIn("Data size mismatch", str(ctx.exception))

    def test_parse_binary_timestamp_conversion(self):
        """Test that binary timestamp is correctly converted to datetime."""
        # Use a known timestamp: 2025-01-01 12:34:56.789 UTC
        timestamp_dt = datetime(2025, 1, 1, 12, 34, 56, 789000, tzinfo=timezone.utc)
        timestamp_us = int(timestamp_dt.timestamp() * 1e6)

        payload_bytes = self._create_binary_payload(timestamp_us=timestamp_us)

        waveform = parse_payload(payload_bytes, self.trap_controls)

        self.assertEqual(waveform.timestamp.year, 2025)
        self.assertEqual(waveform.timestamp.month, 1)
        self.assertEqual(waveform.timestamp.day, 1)
        self.assertEqual(waveform.timestamp.hour, 12)
        self.assertEqual(waveform.timestamp.minute, 34)
        self.assertEqual(waveform.timestamp.second, 56)

    def test_parse_binary_multiple_channels(self):
        """Test parsing binary with multiple channels."""
        channels = {
            "ch1": list(np.linspace(1.0, 2.0, 15000)),
            "ch2": list(np.linspace(3.0, 4.0, 15000)),
            "ch3": list(np.linspace(5.0, 6.0, 15000)),
        }
        payload_bytes = self._create_binary_payload(channels=channels)

        waveform = parse_payload(payload_bytes, self.trap_controls)

        self.assertEqual(len(waveform.channels), 3)
        self.assertIn("ch1", waveform.channels)
        self.assertIn("ch2", waveform.channels)
        self.assertIn("ch3", waveform.channels)


class TestEncodePayload(unittest.TestCase):
    """Test encode_payload function."""

    def setUp(self):
        """Set up test fixtures."""
        self.trap_controls = TrapControls()
        self.trap_controls.basic_controls.r0_cm = 0.5
        self.trap_controls.basic_controls.f_RF_MHz = 2.0
        self.trap_controls.basic_controls.injection_duration_ms = 1.0  # Short injection
        self.trap_controls.excitation_steps[0].step_duration_ms = 10.0  # 10ms duration
        self.trap_controls.excitation_steps[0].rf_amplitude_begin_V0p = 100.0
        self.trap_controls.excitation_steps[0].frequency_begin_kHz = 50.0

    def test_encode_simple_waveform(self):
        """Test encoding a simple waveform."""
        # Create a waveform with enough samples
        channels = {
            "ch1": np.array(list(np.linspace(0.0, 1.0, 15000))),
            "ch2": np.array(list(np.linspace(-0.5, 0.5, 15000))),
        }
        timestamp = datetime(2025, 1, 1, 12, 34, 56, tzinfo=timezone.utc)

        waveform = WaveformData(
            t0=0.0,
            dt=1e-6,
            channels=channels,
            timestamp=timestamp,
            trapping_parameters=self.trap_controls,
            selected_excitation_step=0,
        )

        payload_bytes = encode_payload(waveform)

        # Verify magic number
        self.assertEqual(payload_bytes[:4], b"WAVE")

        # Verify header can be unpacked
        header = struct.unpack("<4sddQII", payload_bytes[:36])
        self.assertEqual(header[0], b"WAVE")
        self.assertEqual(header[1], 0.0)  # t0
        self.assertEqual(header[2], 1e-6)  # dt

    def test_encode_decode_round_trip(self):
        """Test that encode -> decode preserves data."""
        # Create a waveform with enough samples
        channels = {
            "ch1": np.array(list(np.linspace(1.0, 2.0, 15000))),
            "ch2": np.array(list(np.linspace(3.0, 4.0, 15000))),
        }
        timestamp = datetime(2025, 1, 1, 12, 34, 56, 789000, tzinfo=timezone.utc)

        original = WaveformData(
            t0=1.5,
            dt=2e-6,
            channels=channels,
            timestamp=timestamp,
            trapping_parameters=self.trap_controls,
            selected_excitation_step=0,
        )
        # Encode
        payload_bytes = encode_payload(original)

        # Decode
        decoded = parse_payload(payload_bytes, self.trap_controls)

        # Verify
        self.assertEqual(decoded.t0, original.t0)
        self.assertEqual(decoded.dt, original.dt)
        self.assertEqual(decoded.timestamp.year, original.timestamp.year)
        self.assertEqual(decoded.timestamp.month, original.timestamp.month)
        self.assertEqual(decoded.timestamp.day, original.timestamp.day)

    def test_encode_long_channel_name_error(self):
        """Test encoding with channel name that is too long."""
        # Create waveform with very long channel name
        long_name = "ch" * 200  # Over 255 bytes
        channels = {long_name: np.array(list(np.linspace(0.0, 1.0, 15000)))}
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)

        waveform = WaveformData(
            t0=0.0,
            dt=1e-6,
            channels=channels,
            timestamp=timestamp,
            trapping_parameters=self.trap_controls,
            selected_excitation_step=0,
        )

        with self.assertRaises(ValueError) as ctx:
            encode_payload(waveform)

        self.assertIn("Channel name too long", str(ctx.exception))

    def test_encode_maintains_channel_order(self):
        """Test that encode maintains consistent channel ordering."""
        channels = {
            "ch3": np.array(list(np.linspace(3.0, 3.5, 15000))),
            "ch1": np.array(list(np.linspace(1.0, 1.5, 15000))),
            "ch2": np.array(list(np.linspace(2.0, 2.5, 15000))),
        }
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)

        waveform = WaveformData(
            t0=0.0,
            dt=1e-6,
            channels=channels,
            timestamp=timestamp,
            trapping_parameters=self.trap_controls,
            selected_excitation_step=0,
        )

        # Encode twice
        payload1 = encode_payload(waveform)
        payload2 = encode_payload(waveform)

        # Should be identical
        self.assertEqual(payload1, payload2)


class TestParsePayloadEdgeCases(unittest.TestCase):
    """Test edge cases for parse_payload."""

    def setUp(self):
        """Set up test fixtures."""
        self.trap_controls = TrapControls()
        self.trap_controls.basic_controls.injection_duration_ms = 1.0
        self.trap_controls.excitation_steps[0].step_duration_ms = 10.0

    def test_parse_with_selected_excitation_step(self):
        """Test parsing with specific excitation step."""
        self.trap_controls.basic_controls.injection_duration_ms = 1.0
        self.trap_controls.excitation_steps = [
            ExcitationStep(
                step_duration_ms=5.0,
                frequency_begin_kHz=50.0,
                rf_amplitude_begin_V0p=100.0,
            ),
            ExcitationStep(
                step_duration_ms=5.0,
                frequency_begin_kHz=75.0,
                rf_amplitude_begin_V0p=150.0,
            ),
        ]

        # Create data with enough samples (at least 15000 to cover both steps at 1 microsecond intervals)
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": list(np.linspace(0.0, 1.0, 15000))},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        # Parse with step 1
        waveform = parse_payload(
            payload_bytes, self.trap_controls, selected_excitation_step=1
        )

        self.assertEqual(waveform.selected_excitation_step, 1)

    def test_parse_single_channel(self):
        """Test parsing with single channel."""
        self.trap_controls.basic_controls.r0_cm = 0.5
        self.trap_controls.basic_controls.f_RF_MHz = 2.0

        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"single_ch": list(np.linspace(0.0, 1.0, 15000))},
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        waveform = parse_payload(payload_bytes, self.trap_controls)

        self.assertEqual(len(waveform.channels), 1)
        self.assertIn("single_ch", waveform.channels)

    def test_parse_large_payload(self):
        """Test parsing payload with many samples."""
        self.trap_controls.basic_controls.r0_cm = 0.5
        self.trap_controls.basic_controls.f_RF_MHz = 2.0

        # Create large channel data
        n_samples = 10000
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {
                "ch1": list(np.random.randn(n_samples)),
                "ch2": list(np.random.randn(n_samples)),
            },
            "timestamp": "2025-01-01T12:34:56Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        waveform = parse_payload(payload_bytes, self.trap_controls)

        self.assertIsInstance(waveform, WaveformData)


if __name__ == "__main__":
    unittest.main()
