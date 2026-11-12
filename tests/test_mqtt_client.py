import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ms_viewer.data_model import encode_payload
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


def test_waveform_units_from_mqtt5_user_properties():
    client = MqttClient()
    trap = TrapControls()
    trap.basic_controls.injection_duration_ms = 0
    client.trapping_parameters = trap
    received, forwarded = [], []
    client.waveform_received.connect(received.append)
    client.waveform_for_forwarding.connect(
        lambda payload, units: forwarded.append((payload, units))
    )

    def receive(payload, units):
        message = SimpleNamespace(
            topic=client.input_topic,
            payload=payload,
            properties=SimpleNamespace(UserProperty=list(units.items())),
        )
        client._on_message(None, None, message)

    json_payload = json.dumps({
        "t0": 2,
        "dt": 1,
        "timestamp": "2026-01-01T00:00:00Z",
        "channels": {"A": [1, 2, 3, 4]},
    }).encode()
    receive(json_payload, {"sample_unit": "V", "time_unit": "ms"})
    assert received[0].sample_unit == "V"
    assert received[0].t0 == pytest.approx(0.002)
    assert received[0].dt == pytest.approx(0.001)

    binary_payload = encode_payload(received[0])
    units = {"sample_unit": "V", "time_unit": "s"}
    receive(binary_payload, units)
    assert received[1].sample_unit == "V"
    assert received[1].dt == pytest.approx(0.001)
    assert forwarded[-1] == (binary_payload, units)

    receive(binary_payload, {})
    assert received[2].sample_unit == "mV"
