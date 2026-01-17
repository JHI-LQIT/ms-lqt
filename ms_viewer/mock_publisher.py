#!/usr/bin/env python3
"""
Mock oscilloscope MQTT publisher.

This script simulates an oscilloscope that publishes waveform data via MQTT.
It generates realistic waveforms with sinusoidal components, negative peaks,
and random noise.
"""

import argparse
import json
import logging
import struct
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockOscilloscope:
    """
    Mock oscilloscope that generates and publishes waveform data.
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        topic: str,
        interval: float = 0.5,
        num_channels: int = 2,
    ):
        """
        Initialize the mock oscilloscope.

        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            topic: Topic to publish to
            interval: Publish interval in seconds
            num_channels: Number of channels to simulate
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.interval = interval
        self.num_channels = num_channels

        # Waveform parameters
        self.t0 = 0.0
        self.dt = 1e-4
        self.duration = 0.1
        self.N = int(self.duration / self.dt)

        # MQTT client
        self.client = mqtt.Client(client_id="mock_oscilloscope")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self.connected = False

    def _on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection."""
        if rc == 0:
            self.connected = True
            logger.info(
                f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}"
            )
        else:
            logger.error(f"Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection."""
        self.connected = False
        if rc == 0:
            logger.info("Disconnected from broker")
        else:
            logger.warning(f"Unexpected disconnection (code {rc})")

    def connect(self):
        """Connect to the MQTT broker."""
        logger.info(f"Connecting to {self.broker_host}:{self.broker_port}...")
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()

            # Wait for connection
            timeout = 5.0
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not self.connected:
                raise TimeoutError("Connection timeout")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    def disconnect(self):
        """Disconnect from the MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()

    def generate_waveform(self, channel_num: int) -> np.ndarray:
        """
        Generate a realistic waveform for a channel.

        Args:
            channel_num: Channel number (affects frequency and amplitude)

        Returns:
            Numpy array of waveform samples
        """
        # Time array
        t = np.arange(self.N) * self.dt

        # Sinusoidal base signal (different frequency per channel)
        base_freq = 50.0 + channel_num * 25.0  # 50, 75, 100, 125 Hz...
        amplitude = 0.1 + channel_num * 0.05
        signal = amplitude * np.sin(2 * np.pi * base_freq * t)

        # Add negative peaks at fixed indices
        peak_indices = [200, 500, 800]
        for idx in peak_indices:
            if idx < self.N:
                # Gaussian pulse
                pulse_width = 5
                pulse_start = max(0, idx - pulse_width)
                pulse_end = min(self.N, idx + pulse_width + 1)
                pulse_range = np.arange(pulse_start, pulse_end) - idx
                pulse = -amplitude * 2 * np.exp(-(pulse_range**2) / 4.0)
                signal[pulse_start:pulse_end] += pulse

        # Add Gaussian noise
        noise_level = 0.01 * (1 + channel_num * 0.2)
        noise = noise_level * np.random.randn(self.N)
        signal += noise

        return signal

    def publish_waveform(self):
        """Generate and publish a waveform."""
        if not self.connected:
            logger.warning("Not connected, skipping publish")
            return

        # Generate channels
        channels = {}
        for i in range(self.num_channels):
            channel_name = f"ch{i+1}"
            channels[channel_name] = self.generate_waveform(i)

        # Build binary payload
        payload_bytes = self._encode_binary_payload(channels)

        # Publish
        result = self.client.publish(self.topic, payload_bytes, qos=1)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info(
                f"Published waveform to '{self.topic}' ({len(payload_bytes)} bytes)"
            )
        else:
            logger.error(f"Failed to publish: {mqtt.error_string(result.rc)}")

    def _encode_binary_payload(self, channels: dict) -> bytes:
        """
        Encode waveform data into binary format.

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
            channels: Dictionary mapping channel names to numpy arrays

        Returns:
            Binary encoded payload bytes
        """
        # Get sorted channel names for consistent ordering
        channel_names = sorted(channels.keys())
        n_channels = len(channel_names)
        n_samples = self.N

        # Get current timestamp in microseconds
        timestamp_us = int(datetime.now(timezone.utc).timestamp() * 1e6)

        # Build header
        header = struct.pack(
            "<4sddQII",
            b"WAVE",  # Magic number
            self.t0,  # t0 (float64)
            self.dt,  # dt (float64)
            timestamp_us,  # timestamp (int64, microseconds)
            n_channels,  # number of channels (uint32)
            n_samples,  # number of samples (uint32)
        )

        # Build channel names section
        channel_names_bytes = b""
        for channel_name in channel_names:
            name_bytes = channel_name.encode("utf-8")
            channel_names_bytes += struct.pack("<B", len(name_bytes)) + name_bytes

        # Interleave channel data
        interleaved_data = []
        for i in range(n_samples):
            for channel_name in channel_names:
                interleaved_data.append(float(channels[channel_name][i]))

        # Pack data section
        data_format = f"<{len(interleaved_data)}d"
        data_bytes = struct.pack(data_format, *interleaved_data)

        # Combine all sections
        return header + channel_names_bytes + data_bytes

    def run(self, duration: Optional[float] = None):
        """
        Run the mock oscilloscope, publishing waveforms at regular intervals.

        Args:
            duration: Total run duration in seconds. If None, runs indefinitely.
        """
        logger.info(
            f"Starting mock oscilloscope (interval={self.interval}s, channels={self.num_channels})"
        )
        logger.info(f"Publishing to topic: {self.topic}")
        logger.info("Press Ctrl+C to stop")

        start_time = time.time()
        count = 0

        try:
            while True:
                # Publish waveform
                self.publish_waveform()
                count += 1

                # Check duration
                if duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        logger.info(f"Duration limit reached ({duration}s)")
                        break

                # Wait for next interval
                time.sleep(self.interval)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            logger.info(f"Published {count} waveforms")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mock oscilloscope MQTT publisher")
    parser.add_argument(
        "--broker",
        "-b",
        default="localhost",
        help="MQTT broker hostname (default: localhost)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=1883, help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--topic",
        "-t",
        default="oscilloscope/waveform",
        help="MQTT topic to publish to (default: oscilloscope/waveform)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=0.5,
        help="Publish interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--channels",
        "-c",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Number of channels to simulate (default: 2)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Total run duration in seconds (default: infinite)",
    )

    args = parser.parse_args()

    # Create and run mock oscilloscope
    oscilloscope = MockOscilloscope(
        broker_host=args.broker,
        broker_port=args.port,
        topic=args.topic,
        interval=args.interval,
        num_channels=args.channels,
    )

    try:
        oscilloscope.connect()
        oscilloscope.run(duration=args.duration)
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        oscilloscope.disconnect()

    return 0


if __name__ == "__main__":
    exit(main())
