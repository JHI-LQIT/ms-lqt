import os
import subprocess
import sys
from pathlib import Path


def test_viewer_starts_and_closes_headlessly(tmp_path):
    script = """
import os
import sys

from PyQt5.QtCore import QTimer
from ms_viewer import main

class SmokeTestWindow(main.MainWindow):
    def show(self):
        super().show()
        QTimer.singleShot(0, self.close)

main.MainWindow = SmokeTestWindow
home = sys.argv.pop(1)
os.environ["HOME"] = home
os.environ["USERPROFILE"] = home
main.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / ".waveform_viewer_config.yaml").is_file()
    log = (tmp_path / ".ms_lqit.log").read_text(encoding="utf-8")
    assert "Application started successfully" in log
    assert "Application exiting with code 0" in log
