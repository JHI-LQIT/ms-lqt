"""
MQTT client for receiving and publishing waveform data.

This module provides a Qt-compatible MQTT client that connects to an MQTT broker,
subscribes to waveform topics, and emits signals when data is received.
"""

import logging
from typing import Optional

import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, pyqtSignal

from .data_model import PayloadParseError, parse_payload, parse_trapping_parameters
from .trapping_parameters import TrapControls

logger = logging.getLogger(__name__)


def create_unique_id() -> str:
    """
    Create a unique client ID using the current timestamp.

    Returns:
        A unique string to use as MQTT client ID.
    """
    import time

    return f"{int(time.time() * 1000)}"


class MqttClient(QObject):
    """
    Qt-enabled MQTT client for waveform data.

    This client connects to an MQTT broker, subscribes to a topic for incoming
    waveform data, and can publish waveforms to an output topic.

    Signals:
        waveform_received: Emitted when a valid waveform is received (WaveformData)
        connection_state_changed: Emitted when connection state changes (bool, str)
        error_occurred: Emitted when an error occurs (str)
    """

    # Qt signals
    waveform_received = pyqtSignal(object)  # WaveformData
    trapping_parameters_received = pyqtSignal(object)  # TrapControls
    connection_state_changed = pyqtSignal(bool, str)  # connected, message
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, parent=None):
        """
        Initialize the MQTT client.

        Args:
            parent: Optional parent QObject
        """
        super().__init__(parent)

        # Configuration
        self.broker_host = "localhost"
        self.broker_port = 1883
        self.input_topic = "lqt/picoscope/waveform"
        self.trapping_parameters_topic = "lqt/trapping/parameters"
        self.output_topic = "lqt/lqt_ms/republish"
        self.client_id = f"ms_viewer_{create_unique_id()}"
        self.keepalive = 60

        # MQTT client
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._subscribed = False

        self.trapping_parameters: Optional[TrapControls] = None
        self.selected_excitation_step = 0  # Default to step 0

    def set_selected_excitation_step(self, step: int):
        """
        Set the selected excitation step for m/z calculation.

        Args:
            step: Index of the excitation step to use
        """
        self.selected_excitation_step = step
        logger.debug(f"Selected excitation step set to {step}")

    def set_config(
        self,
        broker_host: str,
        broker_port: int,
        input_topic: str,
        trapping_parameters_topic: str,
        output_topic: str,
        client_id: str = "ms_viewer",
        keepalive: int = 60,
    ):
        """
        Update MQTT configuration.

        Args:
            broker_host: MQTT broker hostname or IP
            broker_port: MQTT broker port
            input_topic: Topic to subscribe to for incoming waveforms
            trapping_parameters_topic: Topic to subscribe to for trapping parameters
            output_topic: Topic to publish outgoing waveforms
            client_id: MQTT client ID
            keepalive: Keep-alive interval in seconds
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.input_topic = input_topic
        self.trapping_parameters_topic = trapping_parameters_topic
        self.output_topic = output_topic
        self.client_id = client_id
        self.keepalive = keepalive

        logger.info(
            f"MQTT config updated: {broker_host}:{broker_port}, "
            f"input={input_topic}, trapping_parameters={trapping_parameters_topic}, output={output_topic}"
        )

    def connect(self):
        """
        Connect to the MQTT broker.

        This method is non-blocking and will return immediately.
        Connection status will be reported via the connection_state_changed signal.
        """
        if self._client is not None:
            logger.warning("MQTT client already exists, disconnecting first")
            self.disconnect_from_broker()

        try:
            # Create new client
            self._client = mqtt.Client(client_id=self.client_id)

            # Set callbacks
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
            self._client.on_subscribe = self._on_subscribe

            # Connect to broker
            logger.info(
                f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}"
            )
            self._client.connect_async(
                self.broker_host, self.broker_port, self.keepalive
            )

            # Start network loop in background thread
            self._client.loop_start()

        except Exception as e:
            error_msg = f"Failed to connect to MQTT broker: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self._client = None

    def disconnect_from_broker(self):
        """
        Disconnect from the MQTT broker.
        """
        if self._client is not None:
            logger.info("Disconnecting from MQTT broker")
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            self._connected = False
            self._subscribed = False
            self.connection_state_changed.emit(False, "Disconnected")

    def subscribe(self):
        """
        Subscribe to the input topic.

        Should be called after connection is established.
        """
        if self._client is None or not self._connected:
            logger.warning("Cannot subscribe: not connected to broker")
            return

        try:
            logger.info(f"Subscribing to topic: {self.input_topic}")
            result, mid = self._client.subscribe(self.input_topic, qos=1)

            if result != mqtt.MQTT_ERR_SUCCESS:
                error_msg = f"Failed to subscribe to {self.input_topic}: {mqtt.error_string(result)}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)

        except Exception as e:
            error_msg = f"Exception during subscribe: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

        try:
            logger.info(
                f"Subscribing to trapping parameters topic: {self.trapping_parameters_topic}"
            )
            result, mid = self._client.subscribe(self.trapping_parameters_topic, qos=1)

            if result != mqtt.MQTT_ERR_SUCCESS:
                error_msg = f"Failed to subscribe to {self.trapping_parameters_topic}: {mqtt.error_string(result)}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
        except Exception as e:
            error_msg = f"Exception during subscribe to trapping parameters topic: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def publish(self, topic: str, payload_bytes: bytes):
        """
        Publish a message to the specified topic.

        Args:
            topic: MQTT topic to publish to
            payload_bytes: Message payload as bytes
        """
        if self._client is None or not self._connected:
            error_msg = "Cannot publish: not connected to broker"
            logger.warning(error_msg)
            self.error_occurred.emit(error_msg)
            return

        try:
            result = self._client.publish(topic, payload_bytes, qos=1)

            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                error_msg = (
                    f"Failed to publish to {topic}: {mqtt.error_string(result.rc)}"
                )
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
            else:
                logger.debug(f"Published message to {topic}")

        except Exception as e:
            error_msg = f"Exception during publish: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def is_connected(self) -> bool:
        """
        Check if connected to the broker.

        Returns:
            True if connected, False otherwise
        """
        return self._connected

    # MQTT callbacks

    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback when connection to broker is established.
        """
        if rc == 0:
            self._connected = True
            msg = f"Connected to {self.broker_host}:{self.broker_port}"
            logger.info(msg)
            self.connection_state_changed.emit(True, msg)

            # Automatically subscribe to input topic
            self.subscribe()
        else:
            self._connected = False
            error_msg = f"Connection failed with code {rc}: {mqtt.connack_string(rc)}"
            logger.error(error_msg)
            self.connection_state_changed.emit(False, error_msg)
            self.error_occurred.emit(error_msg)

    def _on_disconnect(self, client, userdata, rc):
        """
        Callback when disconnected from broker.
        """
        self._connected = False
        self._subscribed = False

        if rc == 0:
            msg = "Disconnected from broker"
            logger.info(msg)
            self.connection_state_changed.emit(False, msg)
        else:
            error_msg = f"Unexpected disconnection (code {rc})"
            logger.warning(error_msg)
            self.connection_state_changed.emit(False, error_msg)

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """
        Callback when subscription is confirmed.
        """
        self._subscribed = True
        logger.info(f"Subscribed to {self.input_topic} with QoS {granted_qos}")

    def _on_message(self, client, userdata, message):
        """
        Callback when a message is received.

        Parses the payload and emits waveform_received signal on success,
        or error_occurred signal on failure.
        """
        if message.topic == self.input_topic:
            try:
                # Parse the payload
                if not self.trapping_parameters:
                    return
                # Check if any excitation step is active
                has_active_step = any(
                    step.step_active
                    for step in self.trapping_parameters.excitation_steps
                )
                if not has_active_step:
                    return
                waveform = parse_payload(
                    message.payload, 
                    self.trapping_parameters,
                    selected_excitation_step=self.selected_excitation_step
                )

                # Emit signal with parsed waveform
                logger.debug(f"Received waveform: {waveform}")
                self.waveform_received.emit(waveform)

            except PayloadParseError as e:
                error_msg = f"Failed to parse waveform payload: {e}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)

            except Exception as e:
                error_msg = f"Unexpected error processing message: {e}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
        elif message.topic == self.trapping_parameters_topic:
            # Handle trapping parameters messages if needed
            logger.debug("Received message on trapping parameters topic")
            try:
                self.trapping_parameters = parse_trapping_parameters(message.payload)
                logger.info(
                    f"Received trapping parameters: {self.trapping_parameters.to_dict()}"
                )
                # Emit signal to notify GUI
                self.trapping_parameters_received.emit(self.trapping_parameters)
            except Exception as e:
                error_msg = f"Failed to parse trapping parameters payload: {e}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
