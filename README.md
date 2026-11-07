# MS LQIT

MS LQIT is a desktop viewer for linear quadrupole ion trap mass spectra. It receives oscilloscope waveforms and trap settings over MQTT, calculates the m/z axis for the selected excitation step, and displays spectra and region-of-interest statistics. It can also export a spectrum to CSV or republish waveforms over MQTT.

## Install and run

Use Python 3.10 or newer. From the repository directory on Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ms_viewer.main
```

On Windows PowerShell, create the environment with `py -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`. Then run the same `python` commands. The desktop viewer needs a graphical session for normal use.

Open **MQTT Configuration**, set the broker host and port, then click **Apply & Connect**. Match the input topics to those used by the acquisition software.

| MQTT input | Default topic | Payload |
| --- | --- | --- |
| Waveforms | `lqt/picoscope/waveform` | PicoScope binary or legacy JSON waveform |
| Trapping parameters | `lqt/trapping/parameters` | JSON trap settings and excitation steps. Required to calculate the spectrum. |
| Collector parameters | `lqt/collector/parameters` | Optional JSON metadata included in CSV exports. |

After a reconnect, the viewer waits for trap settings from the new MQTT connection before converting incoming waveforms to m/z. Broker-retained settings can supply these automatically. Collector metadata stays optional, and raw waveform forwarding can resume immediately.

The viewer saves broker settings, ROI bounds, and display preferences in `~/.waveform_viewer_config.yaml`. Close the viewer before editing this file by hand. See [example_config.yaml](example_config.yaml) for additional settings.

## Try it without hardware

Install Mosquitto and its client tools. On Ubuntu, the packages are `mosquitto` and `mosquitto-clients`. Start a dedicated loopback broker in one terminal so the demo is isolated from instrument traffic:

```sh
mosquitto -p 18884
```

In the viewer, set the broker to `127.0.0.1` on port `18884`, use the default input and trapping parameters topics from the table, and click **Apply & Connect**. Select excitation step **1**.

In another terminal, activate the virtual environment and run the following from the repository directory. The retained trap settings define one frequency sweep for the simulated waveforms.

```sh
mosquitto_pub -h 127.0.0.1 -p 18884 -t lqt/trapping/parameters -r -m '{"basic_controls":{"r0_cm":0.475,"f_RF_MHz":2.112,"injection_duration_ms":0},"excitation_steps":[{"step_name":"Demo sweep","step_active":true,"step_duration_ms":100,"enable_frequency_sweep":true,"frequency_begin_kHz":50,"frequency_end_kHz":100,"rf_amplitude_begin_V0p":100}]}'
python -m ms_viewer.mock_publisher --broker 127.0.0.1 --port 18884 --topic lqt/picoscope/waveform
```

The viewer should show two traces on the m/z axis. Use **Add Region** to see ROI statistics or **Export to CSV...** to save the spectrum. Stop the publisher and the demo broker with Ctrl+C in their respective terminals.

## Live acquisition

While a large plot frame is being prepared, the viewer keeps only the latest pending frame. ROI statistics can average multiple scans. The viewer queues up to 64 scans for the next ROI calculation by default and drops the oldest pending scan when the queue fills. The status bar shows the number of ROI scans dropped since startup. Set `preferences.statistics_max_pending_scans` in the viewer configuration to change the limit.

## Run tests

```sh
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

The suite includes a headless viewer startup and shutdown check. It runs without an MQTT broker or instrument. GitHub Actions runs it on Ubuntu with Python 3.10 and 3.12.
