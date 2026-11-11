import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ms_viewer.mqtt_client import MqttClient
from ms_viewer.trapping_parameters import TrapControls


@pytest.mark.parametrize("automatic", [True, False])
def test_reconnect_waits_for_trap_settings_without_blocking_raw_forwarding(automatic):
    client = MqttClient()
    transport = Mock()
    transport.subscribe.return_value = (0, 1)
    client._client = transport

    forwarded, calibrated, errors = [], [], []
    client.raw_waveform_received.connect(forwarded.append)
    client.waveform_received.connect(calibrated.append)
    client.error_occurred.connect(errors.append)

    waveform = json.dumps(
        {
            "t0": 0,
            "dt": 0.001,
            "timestamp": "2026-01-01T00:00:00Z",
            "channels": {"A": [1, 2, 3, 4]},
        }
    ).encode()

    def receive(topic, payload):
        client._on_message(transport, None, SimpleNamespace(topic=topic, payload=payload))

    def trap_payload(rf_amplitude):
        trap = TrapControls()
        trap.basic_controls.injection_duration_ms = 0
        trap.excitation_steps[0].rf_amplitude_begin_V0p = rf_amplitude
        return json.dumps(trap.to_dict()).encode()

    client._on_connect(transport, None, {}, 0)
    receive(client.trapping_parameters_topic, trap_payload(100))
    receive(client.collector_parameters_topic, b'{"run":1}')
    receive(client.input_topic, waveform)
    assert len(calibrated) == 1
    assert calibrated[0].collector_metadata == '{"run":1}'

    if automatic:
        client._on_disconnect(transport, None, 1)
    else:
        client.disconnect_from_broker()
        transport = Mock()
        transport.subscribe.return_value = (0, 1)
        client._client = transport

    assert client.trapping_parameters is None
    assert client.collector_metadata is None

    client._on_connect(transport, None, {}, 0)
    receive(client.input_topic, waveform)
    receive(client.trapping_parameters_topic, b"{")
    receive(client.input_topic, waveform)
    assert forwarded == [waveform] * 3
    assert len(calibrated) == 1
    assert len(errors) == 1

    receive(client.trapping_parameters_topic, trap_payload(200))
    receive(client.input_topic, waveform)
    assert forwarded == [waveform] * 4
    assert len(calibrated) == 2
    np.testing.assert_allclose(calibrated[1].mz, 2 * calibrated[0].mz)
    assert calibrated[1].collector_metadata is None
    assert calibrated[0].collector_metadata == '{"run":1}'

    receive(client.collector_parameters_topic, b'{"run":2}')
    receive(client.input_topic, waveform)
    assert calibrated[2].collector_metadata == '{"run":2}'
