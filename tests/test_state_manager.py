"""
Unit tests for state_manager.py
"""

import os
import tempfile
from pathlib import Path

import pytest

from waveform_viewer.state_manager import get_default_state_path, load_state, save_state


class TestStateManager:
    """Tests for state management functions."""

    def test_load_nonexistent_file(self):
        """Test loading from a non-existent file returns default state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "nonexistent.yaml")
            state = load_state(state_path)

            # Should return default state
            assert "mqtt" in state
            assert "gui" in state
            assert "preferences" in state
            assert state["mqtt"]["broker_host"] == "localhost"

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "test_state.yaml")

            # Create test state
            test_state = {
                "mqtt": {
                    "broker_host": "test.example.com",
                    "broker_port": 1234,
                    "input_topic": "test/input",
                    "output_topic": "test/output",
                },
                "gui": {
                    "regions": [{"start": 0.1, "end": 0.2, "name": "Test Region"}],
                    "window_geometry": {
                        "x": 100,
                        "y": 200,
                        "width": 800,
                        "height": 600,
                    },
                },
                "preferences": {"last_export_dir": "/tmp/exports"},
            }

            # Save state
            assert save_state(state_path, test_state)

            # Load state
            loaded_state = load_state(state_path)

            # Verify
            assert loaded_state["mqtt"]["broker_host"] == "test.example.com"
            assert loaded_state["mqtt"]["broker_port"] == 1234
            assert loaded_state["mqtt"]["input_topic"] == "test/input"
            assert loaded_state["gui"]["regions"][0]["name"] == "Test Region"
            assert loaded_state["gui"]["window_geometry"]["width"] == 800
            assert loaded_state["preferences"]["last_export_dir"] == "/tmp/exports"

    def test_load_corrupted_file(self):
        """Test loading a corrupted YAML file returns default state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "corrupted.yaml")

            # Write corrupted YAML
            with open(state_path, "w") as f:
                f.write("this is not: valid: yaml: content:::::")

            # Should return default state
            state = load_state(state_path)
            assert "mqtt" in state
            assert state["mqtt"]["broker_host"] == "localhost"

    def test_load_invalid_data(self):
        """Test loading a file with invalid data (not a dict) returns default state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "invalid.yaml")

            # Write valid YAML but not a dict
            with open(state_path, "w") as f:
                f.write("- item1\n- item2\n")

            # Should return default state
            state = load_state(state_path)
            assert "mqtt" in state
            assert state["mqtt"]["broker_host"] == "localhost"

    def test_save_creates_directories(self):
        """Test that save_state creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "subdir", "nested", "state.yaml")

            test_state = {
                "mqtt": {"broker_host": "localhost", "broker_port": 1883},
                "gui": {"regions": []},
                "preferences": {},
            }

            # Save should create directories
            assert save_state(state_path, test_state)
            assert os.path.exists(state_path)

    def test_partial_state_merge(self):
        """Test that partial states are merged with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "partial.yaml")

            # Save a partial state (missing some fields)
            partial_state = {
                "mqtt": {
                    "broker_host": "custom.example.com"
                    # Missing other mqtt fields
                }
                # Missing gui and preferences
            }

            save_state(state_path, partial_state)

            # Load should merge with defaults
            loaded_state = load_state(state_path)

            # Custom field preserved
            assert loaded_state["mqtt"]["broker_host"] == "custom.example.com"

            # Default fields present
            assert loaded_state["mqtt"]["broker_port"] == 1883
            assert "gui" in loaded_state
            assert "preferences" in loaded_state

    def test_get_default_state_path(self):
        """Test getting the default state path."""
        path = get_default_state_path()

        # Should be a string
        assert isinstance(path, str)

        # Should be in home directory
        assert str(Path.home()) in path

        # Should have .yaml extension
        assert path.endswith(".yaml")
