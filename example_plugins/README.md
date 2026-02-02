# Example Custom Statistics Plugins

This directory contains example custom statistic plugins for the MQTT Oscilloscope Waveform Viewer.

## Installation

Copy the plugin files you want to use to your plugin directory:

```bash
mkdir -p ~/.ms_viewer/plugins
cp example_plugins/*.py ~/.ms_viewer/plugins/
```

Then enable them in your configuration file (`~/.ms_viewer_config.yaml`):

```yaml
statistics:
  enabled_plugins:
    - rms_calculator
    - frequency_analysis
    - pulse_analysis
```

## Available Plugins

### rms_calculator.py

Calculates RMS (Root Mean Square) value and RMS in decibels.

**Metrics:**

- `rms`: RMS value
- `rms_db`: RMS in decibels (20*log10(RMS))

### frequency_analysis.py

Performs FFT-based frequency domain analysis.

**Metrics:**

- `dominant_frequency`: Frequency with highest magnitude (Hz)
- `thd_percent`: Total Harmonic Distortion (%)
- `spectral_centroid`: Weighted mean frequency (Hz)

**Requirements:** scipy

### pulse_analysis.py

Analyzes pulse characteristics in the waveform.

**Metrics:**

- `rise_time`: 10%-90% rise time (seconds)
- `fall_time`: 90%-10% fall time (seconds)
- `pulse_width`: Width at 50% amplitude (seconds)
- `overshoot_percent`: Overshoot above steady-state (%)

## Creating Your Own Plugins

See the Plugin Development Guide in the main documentation for details.

### Quick Example

```python
from ms_viewer.roi_statistics import CustomStatisticPlugin
import numpy as np

class MyPlugin(CustomStatisticPlugin):
    def calculate(self, times, amplitudes, roi_start, roi_end):
        # Your calculations here
        my_value = np.max(amplitudes) - np.min(amplitudes)
        
        return {
            'my_metric': my_value
        }
    
    def get_metrics(self):
        return ['my_metric']
    
    def get_info(self):
        return {
            'name': 'My Custom Plugin',
            'version': '1.0',
            'author': 'Your Name',
            'description': 'Does something interesting'
        }
```

Save as `my_plugin.py` in your plugins directory and enable it in the config.
