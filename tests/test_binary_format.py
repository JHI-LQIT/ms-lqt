"""
Unit tests for binary payload encoding and decoding.

Tests the binary format implementation in data_model module,
including encoding, decoding, format detection, and edge cases.
"""

import unittest
from datetime import datetime, timezone

import numpy as np

from waveform_viewer.data_model import (
    PayloadParseError,
    WaveformData,
    encode_payload,
    parse_payload,
)


class TestBinaryEncoding(unittest.TestCase):
    """Tests for binary payload encoding."""

    def test_basic_encoding(self):
        """Test basic binary encoding of a simple waveform."""
        channels = {
            "ch1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "ch2": np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
        }
        timestamp = datetime(2025, 11, 29, 12, 0, 0, tzinfo=timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)

        # Check payload starts with magic number
        self.assertEqual(payload[:4], b"WAVE")

        # Check payload is bytes
        self.assertIsInstance(payload, bytes)

        # Check payload has reasonable size
        # Header (36) + channel names (2 + 3 + 3) + data (2 * 5 * 8)
        expected_min_size = 36 + 8 + 80
        self.assertGreaterEqual(len(payload), expected_min_size)

    def test_encoding_preserves_values(self):
        """Test that encoding preserves all waveform values."""
        channels = {
            "ch1": np.array([1.234, -5.678, 0.0, 9.999]),
            "ch2": np.array([-0.001, 3.14159, 2.71828, 0.5772]),
        }
        timestamp = datetime.now(timezone.utc)
        waveform = WaveformData(
            t0=1.5, dt=0.0002, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)
        decoded = parse_payload(payload)

        # Check metadata
        self.assertEqual(decoded.t0, waveform.t0)
        self.assertEqual(decoded.dt, waveform.dt)

        # Check channel data
        for ch_name in channels.keys():
            np.testing.assert_array_almost_equal(
                decoded.channels[ch_name], channels[ch_name], decimal=10
            )

    def test_encoding_multiple_channels(self):
        """Test encoding with multiple channels."""
        n_samples = 100
        channels = {
            "ch1": np.sin(2 * np.pi * 50 * np.linspace(0, 0.1, n_samples)),
            "ch2": np.cos(2 * np.pi * 75 * np.linspace(0, 0.1, n_samples)),
            "ch3": np.random.randn(n_samples),
            "ch4": np.ones(n_samples) * 2.5,
        }
        timestamp = datetime.now(timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)
        decoded = parse_payload(payload)

        # Check all channels preserved
        self.assertEqual(set(decoded.channels.keys()), set(channels.keys()))

        for ch_name in channels.keys():
            np.testing.assert_allclose(
                decoded.channels[ch_name], channels[ch_name], rtol=1e-12
            )

    def test_encoding_long_channel_names(self):
        """Test encoding with long channel names."""
        channels = {
            "very_long_channel_name_123456789": np.array([1.0, 2.0]),
            "another_really_long_channel_name": np.array([3.0, 4.0]),
        }
        timestamp = datetime.now(timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)
        decoded = parse_payload(payload)

        self.assertEqual(set(decoded.channels.keys()), set(channels.keys()))


class TestBinaryDecoding(unittest.TestCase):
    """Tests for binary payload decoding."""

    def test_decode_basic_payload(self):
        """Test decoding a basic binary payload."""
        channels = {"ch1": np.array([1.0, 2.0, 3.0]), "ch2": np.array([4.0, 5.0, 6.0])}
        timestamp = datetime.now(timezone.utc)
        original = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(original)
        decoded = parse_payload(payload)

        self.assertIsInstance(decoded, WaveformData)
        self.assertEqual(decoded.t0, original.t0)
        self.assertEqual(decoded.dt, original.dt)

    def test_decode_empty_payload_fails(self):
        """Test that empty payload raises error."""
        with self.assertRaises(PayloadParseError):
            parse_payload(b"")

    def test_decode_short_payload_fails(self):
        """Test that payload shorter than header raises error."""
        with self.assertRaises(PayloadParseError):
            parse_payload(b"WAVE" + b"\x00" * 20)

    def test_decode_invalid_magic_fails(self):
        """Test that invalid magic number raises error."""
        with self.assertRaises(PayloadParseError):
            parse_payload(b"FAKE" + b"\x00" * 32)

    def test_decode_single_sample(self):
        """Test decoding waveform with single sample."""
        channels = {"ch1": np.array([1.234])}
        timestamp = datetime.now(timezone.utc)
        original = WaveformData(t0=0.0, dt=1.0, channels=channels, timestamp=timestamp)

        payload = encode_payload(original)
        decoded = parse_payload(payload)

        self.assertEqual(len(decoded.channels["ch1"]), 1)
        self.assertAlmostEqual(decoded.channels["ch1"][0], 1.234)

    def test_decode_large_waveform(self):
        """Test decoding large realistic waveform."""
        n_samples = 10000
        channels = {
            "ch1": np.random.randn(n_samples),
            "ch2": np.random.randn(n_samples),
        }
        timestamp = datetime.now(timezone.utc)
        original = WaveformData(
            t0=0.0, dt=0.0001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(original)
        decoded = parse_payload(payload)

        for ch_name in channels.keys():
            np.testing.assert_allclose(
                decoded.channels[ch_name], channels[ch_name], rtol=1e-12
            )


class TestFormatDetection(unittest.TestCase):
    """Tests for automatic format detection (binary vs JSON)."""

    def test_detect_binary_format(self):
        """Test that binary format is correctly detected."""
        channels = {"ch1": np.array([1.0, 2.0, 3.0])}
        timestamp = datetime.now(timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)
        decoded = parse_payload(payload)

        self.assertIsInstance(decoded, WaveformData)

    def test_detect_json_format(self):
        """Test that JSON format is correctly detected."""
        import json

        json_payload = json.dumps(
            {
                "t0": 0.0,
                "dt": 0.001,
                "channels": {"ch1": [1.0, 2.0, 3.0], "ch2": [4.0, 5.0, 6.0]},
                "timestamp": "2025-11-29T12:34:56.789Z",
            }
        ).encode("utf-8")

        decoded = parse_payload(json_payload)

        self.assertIsInstance(decoded, WaveformData)
        self.assertEqual(decoded.t0, 0.0)
        self.assertEqual(decoded.dt, 0.001)
        self.assertEqual(len(decoded.channels), 2)

    def test_json_backward_compatibility(self):
        """Test that legacy JSON payloads still work."""
        import json

        json_payload = json.dumps(
            {
                "t0": 1.5,
                "dt": 0.0002,
                "channels": {
                    "ch1": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "ch2": [0.5, 1.5, 2.5, 3.5, 4.5],
                },
                "timestamp": "2025-11-29T12:00:00.000Z",
            }
        ).encode("utf-8")

        decoded = parse_payload(json_payload)

        self.assertEqual(decoded.t0, 1.5)
        self.assertEqual(decoded.dt, 0.0002)
        np.testing.assert_array_equal(
            decoded.channels["ch1"], np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )


class TestBinaryPayloadSize(unittest.TestCase):
    """Tests for binary payload size efficiency."""

    def test_binary_smaller_than_json(self):
        """Test that binary format is smaller than JSON."""
        import json

        n_samples = 1000
        channels = {
            "ch1": np.random.randn(n_samples),
            "ch2": np.random.randn(n_samples),
        }
        timestamp = datetime.now(timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.0001, channels=channels, timestamp=timestamp
        )

        # Binary encoding
        binary_payload = encode_payload(waveform)
        binary_size = len(binary_payload)

        # JSON encoding
        channels_dict = {ch: data.tolist() for ch, data in channels.items()}
        json_payload = json.dumps(
            {
                "t0": 0.0,
                "dt": 0.0001,
                "channels": channels_dict,
                "timestamp": timestamp.isoformat(),
            }
        ).encode("utf-8")
        json_size = len(json_payload)

        # Binary should be significantly smaller
        self.assertLess(binary_size, json_size)
        compression_ratio = json_size / binary_size
        self.assertGreater(compression_ratio, 2.0)  # At least 2x compression

    def test_size_scaling(self):
        """Test that binary size scales linearly with samples."""
        base_samples = 100
        channels = {"ch1": np.ones(base_samples), "ch2": np.ones(base_samples)}
        timestamp = datetime.now(timezone.utc)
        waveform1 = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )
        size1 = len(encode_payload(waveform1))

        # Double the samples
        channels2 = {"ch1": np.ones(base_samples * 2), "ch2": np.ones(base_samples * 2)}
        waveform2 = WaveformData(
            t0=0.0, dt=0.001, channels=channels2, timestamp=timestamp
        )
        size2 = len(encode_payload(waveform2))

        # Size should roughly double (allowing for fixed header overhead)
        header_size = 36 + 8  # Header + channel names
        data_size1 = size1 - header_size
        data_size2 = size2 - header_size

        self.assertAlmostEqual(data_size2 / data_size1, 2.0, places=1)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_channel_name_with_special_chars(self):
        """Test encoding channel names with special characters."""
        channels = {
            "ch-1_test": np.array([1.0, 2.0]),
            "CH2.signal": np.array([3.0, 4.0]),
            "Канал3": np.array([5.0, 6.0]),  # Unicode
        }
        timestamp = datetime.now(timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)
        decoded = parse_payload(payload)

        self.assertEqual(set(decoded.channels.keys()), set(channels.keys()))

    def test_negative_values(self):
        """Test encoding negative values."""
        channels = {
            "ch1": np.array([-10.0, -5.0, -1.0, -0.001]),
        }
        timestamp = datetime.now(timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)
        decoded = parse_payload(payload)

        np.testing.assert_array_almost_equal(decoded.channels["ch1"], channels["ch1"])

    def test_very_small_values(self):
        """Test encoding very small values (near machine precision)."""
        channels = {
            "ch1": np.array([1e-15, 1e-14, 1e-13, 1e-12]),
        }
        timestamp = datetime.now(timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)
        decoded = parse_payload(payload)

        np.testing.assert_allclose(decoded.channels["ch1"], channels["ch1"], rtol=1e-12)

    def test_timestamp_preservation(self):
        """Test that timestamps are preserved through encoding/decoding."""
        channels = {"ch1": np.array([1.0, 2.0, 3.0])}
        timestamp = datetime(2025, 11, 29, 15, 30, 45, 123456, tzinfo=timezone.utc)
        waveform = WaveformData(
            t0=0.0, dt=0.001, channels=channels, timestamp=timestamp
        )

        payload = encode_payload(waveform)
        decoded = parse_payload(payload)

        # Allow for microsecond precision loss
        time_diff = abs((decoded.timestamp - timestamp).total_seconds())
        self.assertLess(time_diff, 1e-6)


if __name__ == "__main__":
    unittest.main()
