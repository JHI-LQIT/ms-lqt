"""
Unit tests for mqtt_client.py
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import paho.mqtt.client as mqtt
import pytest
from PyQt5.QtCore import QObject

from waveform_viewer.data_model import WaveformData
from waveform_viewer.mqtt_client import MqttClient


class TestMqttClient:
    """Tests for MqttClient class."""

    def test_initialization(self):
        """Test MqttClient initialization."""
        client = MqttClient()

        assert client.broker_host == "localhost"
        assert client.broker_port == 1883
        assert client.input_topic == "oscilloscope/waveform"
        assert client.output_topic == "oscilloscope/republish"
        assert not client.is_connected()

    def test_set_config(self):
        """Test setting MQTT configuration."""
        client = MqttClient()

        client.set_config(
            broker_host="test.example.com",
            broker_port=8883,
            input_topic="test/input",
            output_topic="test/output",
            client_id="test_client",
            keepalive=120,
        )

        assert client.broker_host == "test.example.com"
        assert client.broker_port == 8883
        assert client.input_topic == "test/input"
        assert client.output_topic == "test/output"
        assert client.client_id == "test_client"
        assert client.keepalive == 120

    @patch("paho.mqtt.client.Client")
    def test_connect(self, mock_mqtt_client_class):
        """Test connecting to MQTT broker."""
        mock_client_instance = MagicMock()
        mock_mqtt_client_class.return_value = mock_client_instance

        client = MqttClient()
        client.connect()

        # Verify MQTT client was created
        mock_mqtt_client_class.assert_called_once_with(client_id="waveform_viewer")

        # Verify connect_async was called
        mock_client_instance.connect_async.assert_called_once_with(
            "localhost", 1883, 60
        )

        # Verify loop_start was called
        mock_client_instance.loop_start.assert_called_once()

    @patch("paho.mqtt.client.Client")
    def test_disconnect(self, mock_mqtt_client_class):
        """Test disconnecting from MQTT broker."""
        mock_client_instance = MagicMock()
        mock_mqtt_client_class.return_value = mock_client_instance

        client = MqttClient()
        client.connect()
        client.disconnect_from_broker()

        # Verify disconnect was called
        mock_client_instance.loop_stop.assert_called_once()
        mock_client_instance.disconnect.assert_called_once()
        assert not client.is_connected()

    def test_on_connect_success(self):
        """Test _on_connect callback with successful connection."""
        client = MqttClient()

        # Mock signal emission
        connection_changed_signal = Mock()
        client.connection_state_changed.connect(connection_changed_signal)

        # Call on_connect with success code
        client._on_connect(None, None, None, 0)

        # Verify connection state
        assert client.is_connected()

        # Verify signal was emitted
        connection_changed_signal.assert_called_once()
        args = connection_changed_signal.call_args[0]
        assert args[0] is True  # connected

    def test_on_connect_failure(self):
        """Test _on_connect callback with connection failure."""
        client = MqttClient()

        # Mock signal emission
        connection_changed_signal = Mock()
        error_signal = Mock()
        client.connection_state_changed.connect(connection_changed_signal)
        client.error_occurred.connect(error_signal)

        # Call on_connect with error code
        client._on_connect(None, None, None, 5)  # 5 = connection refused

        # Verify connection state
        assert not client.is_connected()

        # Verify signals were emitted
        connection_changed_signal.assert_called_once()
        error_signal.assert_called_once()

    def test_on_disconnect(self):
        """Test _on_disconnect callback."""
        client = MqttClient()
        client._connected = True

        # Mock signal emission
        connection_changed_signal = Mock()
        client.connection_state_changed.connect(connection_changed_signal)

        # Call on_disconnect
        client._on_disconnect(None, None, 0)

        # Verify connection state
        assert not client.is_connected()

        # Verify signal was emitted
        connection_changed_signal.assert_called_once()

    def test_on_message_valid_payload(self):
        """Test _on_message with valid waveform payload."""
        client = MqttClient()

        # Mock signal emission
        waveform_signal = Mock()
        client.waveform_received.connect(waveform_signal)

        # Create valid payload
        payload = {
            "t0": 0.0,
            "dt": 1e-6,
            "channels": {"ch1": [0.0, 0.1, 0.2], "ch2": [-0.1, 0.0, 0.1]},
            "timestamp": "2025-01-01T12:34:56.789Z",
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        # Create mock message
        mock_message = Mock()
        mock_message.payload = payload_bytes

        # Call on_message
        client._on_message(None, None, mock_message)

        # Verify waveform_received signal was emitted
        waveform_signal.assert_called_once()

        # Verify the waveform data
        waveform = waveform_signal.call_args[0][0]
        assert isinstance(waveform, WaveformData)
        assert waveform.t0 == 0.0
        assert waveform.dt == 1e-6
        assert len(waveform.channels) == 2

    def test_on_message_invalid_payload(self):
        """Test _on_message with invalid payload."""
        client = MqttClient()

        # Mock signal emission
        error_signal = Mock()
        client.error_occurred.connect(error_signal)

        # Create invalid payload
        invalid_payload = b"not valid json"

        # Create mock message
        mock_message = Mock()
        mock_message.payload = invalid_payload

        # Call on_message
        client._on_message(None, None, mock_message)

        # Verify error_occurred signal was emitted
        error_signal.assert_called_once()

    @patch("paho.mqtt.client.Client")
    def test_subscribe(self, mock_mqtt_client_class):
        """Test subscribing to input topic."""
        mock_client_instance = MagicMock()
        mock_client_instance.subscribe.return_value = (mqtt.MQTT_ERR_SUCCESS, 1)
        mock_mqtt_client_class.return_value = mock_client_instance

        client = MqttClient()
        client._client = mock_client_instance
        client._connected = True

        client.subscribe()

        # Verify subscribe was called
        mock_client_instance.subscribe.assert_called_once_with(
            "oscilloscope/waveform", qos=1
        )

    @patch("paho.mqtt.client.Client")
    def test_publish(self, mock_mqtt_client_class):
        """Test publishing a message."""
        mock_client_instance = MagicMock()
        mock_result = Mock()
        mock_result.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client_instance.publish.return_value = mock_result
        mock_mqtt_client_class.return_value = mock_client_instance

        client = MqttClient()
        client._client = mock_client_instance
        client._connected = True

        test_payload = b"test payload"
        client.publish("test/topic", test_payload)

        # Verify publish was called
        mock_client_instance.publish.assert_called_once_with(
            "test/topic", test_payload, qos=1
        )

    def test_publish_not_connected(self):
        """Test publishing when not connected."""
        client = MqttClient()

        # Mock signal emission
        error_signal = Mock()
        client.error_occurred.connect(error_signal)

        client.publish("test/topic", b"test")

        # Verify error was emitted
        error_signal.assert_called_once()
