"""
Unit tests for trapping_parameters module.

Tests cover:
- BasicTrapControls and ExcitationStep dataclasses
- TrapControls initialization, conversion, and calculation methods
- Edge cases and error handling
"""

import unittest

from ms_viewer.trapping_parameters import (
    BasicTrapControls,
    ExcitationStep,
    TrapControls,
)


class TestBasicTrapControls(unittest.TestCase):
    """Test BasicTrapControls dataclass."""

    def test_default_initialization(self):
        """Test default initialization of BasicTrapControls."""
        basic = BasicTrapControls()

        self.assertEqual(basic.r0_cm, 0.475)
        self.assertEqual(basic.f_RF_MHz, 2.112)
        self.assertEqual(basic.injection_duration_ms, 5.0)
        self.assertEqual(basic.extraction_duration_ms, 5.0)
        self.assertFalse(basic.enable_trapping)
        self.assertFalse(basic.turn_off_rf_after_extraction)

    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        basic = BasicTrapControls(
            r0_cm=0.5,
            f_RF_MHz=2.0,
            enable_trapping=True,
            injection_duration_ms=10.0,
        )

        self.assertEqual(basic.r0_cm, 0.5)
        self.assertEqual(basic.f_RF_MHz, 2.0)
        self.assertTrue(basic.enable_trapping)
        self.assertEqual(basic.injection_duration_ms, 10.0)


class TestExcitationStep(unittest.TestCase):
    """Test ExcitationStep dataclass."""

    def test_default_initialization(self):
        """Test default initialization of ExcitationStep."""
        step = ExcitationStep()

        self.assertEqual(step.step_name, "")
        self.assertTrue(step.step_active)
        self.assertEqual(step.step_duration_ms, 100.0)
        self.assertFalse(step.enable_frequency_sweep)
        self.assertEqual(step.frequency_begin_kHz, 50.0)
        self.assertEqual(step.frequency_end_kHz, 100.0)
        self.assertFalse(step.enable_ac_amplitude_sweep)
        self.assertFalse(step.enable_rf_amplitude_sweep)

    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        step = ExcitationStep(
            step_name="Test Step",
            step_active=False,
            step_duration_ms=200.0,
            enable_frequency_sweep=True,
            frequency_begin_kHz=100.0,
            frequency_end_kHz=200.0,
        )

        self.assertEqual(step.step_name, "Test Step")
        self.assertFalse(step.step_active)
        self.assertEqual(step.step_duration_ms, 200.0)
        self.assertTrue(step.enable_frequency_sweep)
        self.assertEqual(step.frequency_begin_kHz, 100.0)
        self.assertEqual(step.frequency_end_kHz, 200.0)


class TestTrapControls(unittest.TestCase):
    """Test TrapControls class."""

    def test_default_initialization(self):
        """Test default initialization of TrapControls."""
        trap = TrapControls()

        self.assertIsInstance(trap.basic_controls, BasicTrapControls)
        self.assertEqual(len(trap.excitation_steps), 1)
        self.assertIsInstance(trap.excitation_steps[0], ExcitationStep)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        trap = TrapControls()
        trap.basic_controls.r0_cm = 0.5
        trap.excitation_steps[0].step_name = "Test"

        result = trap.to_dict()

        self.assertIn("basic_controls", result)
        self.assertIn("excitation_steps", result)
        self.assertEqual(result["basic_controls"]["r0_cm"], 0.5)
        self.assertEqual(result["excitation_steps"][0]["step_name"], "Test")

    def test_from_dict(self):
        """Test creation from structured dictionary."""
        data = {
            "basic_controls": {
                "r0_cm": 0.5,
                "f_RF_MHz": 2.0,
                "enable_trapping": True,
            },
            "excitation_steps": [
                {
                    "step_name": "Step1",
                    "step_duration_ms": 150.0,
                    "enable_frequency_sweep": True,
                },
                {
                    "step_name": "Step2",
                    "step_duration_ms": 200.0,
                },
            ],
        }

        trap = TrapControls.from_dict(data)

        self.assertEqual(trap.basic_controls.r0_cm, 0.5)
        self.assertEqual(trap.basic_controls.f_RF_MHz, 2.0)
        self.assertTrue(trap.basic_controls.enable_trapping)
        self.assertEqual(len(trap.excitation_steps), 2)
        self.assertEqual(trap.excitation_steps[0].step_name, "Step1")
        self.assertEqual(trap.excitation_steps[0].step_duration_ms, 150.0)
        self.assertTrue(trap.excitation_steps[0].enable_frequency_sweep)
        self.assertEqual(trap.excitation_steps[1].step_name, "Step2")
        self.assertEqual(trap.excitation_steps[1].step_duration_ms, 200.0)

    def test_from_dict_empty_sections(self):
        """Test from_dict with empty or missing sections."""
        data = {}
        trap = TrapControls.from_dict(data)

        # Should still create with defaults
        self.assertIsInstance(trap.basic_controls, BasicTrapControls)
        self.assertEqual(len(trap.excitation_steps), 1)

    def test_to_flat_dict(self):
        """Test conversion to flat dictionary."""
        trap = TrapControls()
        trap.basic_controls.r0_cm = 0.5
        trap.basic_controls.enable_trapping = True
        trap.excitation_steps[0].step_name = "Test"
        trap.excitation_steps[0].frequency_begin_kHz = 75.0

        result = trap.to_flat_dict()

        # Should contain both basic controls and first excitation step
        self.assertEqual(result["r0_cm"], 0.5)
        self.assertTrue(result["enable_trapping"])
        self.assertEqual(result["step_name"], "Test")
        self.assertEqual(result["frequency_begin_kHz"], 75.0)

    def test_from_flat_dict(self):
        """Test creation from flat dictionary."""
        data = {
            "r0_cm": 0.5,
            "f_RF_MHz": 2.0,
            "enable_trapping": True,
            "step_name": "FlatTest",
            "step_duration_ms": 150.0,
            "enable_frequency_sweep": True,
            "frequency_begin_kHz": 75.0,
        }

        trap = TrapControls.from_flat_dict(data)

        self.assertEqual(trap.basic_controls.r0_cm, 0.5)
        self.assertEqual(trap.basic_controls.f_RF_MHz, 2.0)
        self.assertTrue(trap.basic_controls.enable_trapping)
        self.assertEqual(len(trap.excitation_steps), 1)
        self.assertEqual(trap.excitation_steps[0].step_name, "FlatTest")
        self.assertEqual(trap.excitation_steps[0].step_duration_ms, 150.0)
        self.assertTrue(trap.excitation_steps[0].enable_frequency_sweep)
        self.assertEqual(trap.excitation_steps[0].frequency_begin_kHz, 75.0)

    def test_from_flat_dict_with_string_conversions(self):
        """Test from_flat_dict with string values that need conversion."""
        data = {
            "r0_cm": "0.5",  # String float
            "enable_trapping": "true",  # String bool
            "step_duration_ms": "150.0",  # String float
            "enable_frequency_sweep": "1",  # String bool (numeric)
        }

        trap = TrapControls.from_flat_dict(data)

        self.assertEqual(trap.basic_controls.r0_cm, 0.5)
        self.assertTrue(trap.basic_controls.enable_trapping)
        self.assertEqual(trap.excitation_steps[0].step_duration_ms, 150.0)
        self.assertTrue(trap.excitation_steps[0].enable_frequency_sweep)

    def test_from_flat_dict_bool_conversion_variations(self):
        """Test various string representations of booleans."""
        # Test "true" variations
        for true_val in ["true", "True", "TRUE", "1", "yes", "Yes"]:
            data = {"enable_trapping": true_val}
            trap = TrapControls.from_flat_dict(data)
            self.assertTrue(trap.basic_controls.enable_trapping)

        # Test "false" variations
        for false_val in ["false", "False", "FALSE", "0", "no", "No"]:
            data = {"enable_trapping": false_val}
            trap = TrapControls.from_flat_dict(data)
            self.assertFalse(trap.basic_controls.enable_trapping)

    def test_calculate_step_start_time(self):
        """Test calculation of step start time."""
        trap = TrapControls()
        trap.basic_controls.injection_duration_ms = 10.0
        trap.excitation_steps = [
            ExcitationStep(step_duration_ms=100.0),
            ExcitationStep(step_duration_ms=150.0),
            ExcitationStep(step_duration_ms=200.0),
        ]

        # Step 0: starts right after injection
        self.assertEqual(trap.calculate_step_start_time(0), 10.0)

        # Step 1: injection + step0.duration
        self.assertEqual(trap.calculate_step_start_time(1), 110.0)

        # Step 2: injection + step0.duration + step1.duration
        # = 10 + 100 + 150 = 260
        self.assertEqual(trap.calculate_step_start_time(2), 260.0)

    def test_calculate_total_duration(self):
        """Test calculation of total duration."""
        trap = TrapControls()
        trap.basic_controls.injection_duration_ms = 10.0
        trap.excitation_steps = [ExcitationStep(step_duration_ms=100.0)]
        trap.basic_controls.extraction_duration_ms = 5.0

        # Without RF off: injection(10) + steps(100) + extraction(5) = 115
        total = trap.calculate_total_duration()
        self.assertEqual(total, 115.0)

        # With RF off
        trap.basic_controls.turn_off_rf_after_extraction = True
        trap.basic_controls.turn_off_rf_duration_ms = 20.0
        total = trap.calculate_total_duration()
        self.assertEqual(total, 135.0)

    def test_calculate_step_mz_range_basic(self):
        """Test m/z range calculation for basic case."""
        trap = TrapControls()
        trap.basic_controls.r0_cm = 0.5
        trap.basic_controls.f_RF_MHz = 2.0
        trap.excitation_steps[0].rf_amplitude_begin_V0p = 100.0
        trap.excitation_steps[0].frequency_begin_kHz = 50.0

        mz_range = trap.calculate_step_mz_range(0)

        self.assertIsNotNone(mz_range)
        # m/z = 34.45 * V_RF / (r0^2 * f_RF * f_secular)
        # m/z = 34.45 * 100 / (0.5^2 * 2.0 * 50.0) = 3445 / 50 = 68.9
        expected_mz = 34.45 * 100.0 / (0.5**2 * 2.0 * 50.0)
        self.assertAlmostEqual(mz_range[0], expected_mz, places=5)  # type: ignore
        self.assertAlmostEqual(mz_range[1], expected_mz, places=5)  # type: ignore

    def test_calculate_step_mz_range_with_sweeps(self):
        """Test m/z range calculation with frequency and RF sweeps."""
        trap = TrapControls()
        trap.basic_controls.r0_cm = 0.5
        trap.basic_controls.f_RF_MHz = 2.0

        step = trap.excitation_steps[0]
        step.enable_frequency_sweep = True
        step.frequency_begin_kHz = 50.0
        step.frequency_end_kHz = 100.0
        step.enable_rf_amplitude_sweep = True
        step.rf_amplitude_begin_V0p = 100.0
        step.rf_amplitude_end_V0p = 200.0

        mz_range = trap.calculate_step_mz_range(0)

        self.assertIsNotNone(mz_range)
        # m/z_begin = 34.45 * 100 / (0.5^2 * 2.0 * 50.0)
        expected_mz_begin = 34.45 * 100.0 / (0.5**2 * 2.0 * 50.0)
        # m/z_end = 34.45 * 200 / (0.5^2 * 2.0 * 100.0)
        expected_mz_end = 34.45 * 200.0 / (0.5**2 * 2.0 * 100.0)

        self.assertAlmostEqual(mz_range[0], expected_mz_begin, places=5)  # type: ignore
        self.assertAlmostEqual(mz_range[1], expected_mz_end, places=5)  # type: ignore

    def test_calculate_step_mz_range_invalid_index(self):
        """Test m/z range calculation with invalid step index."""
        trap = TrapControls()

        # Negative index
        self.assertIsNone(trap.calculate_step_mz_range(-1))

        # Index out of range
        self.assertIsNone(trap.calculate_step_mz_range(10))

    def test_calculate_step_mz_range_zero_values(self):
        """Test m/z range calculation with zero values (should return None)."""
        trap = TrapControls()

        # Zero r0
        trap.basic_controls.r0_cm = 0.0
        self.assertIsNone(trap.calculate_step_mz_range(0))

        # Zero f_RF
        trap.basic_controls.r0_cm = 0.5
        trap.basic_controls.f_RF_MHz = 0.0
        self.assertIsNone(trap.calculate_step_mz_range(0))

        # Zero frequency
        trap.basic_controls.f_RF_MHz = 2.0
        trap.excitation_steps[0].frequency_begin_kHz = 0.0
        self.assertIsNone(trap.calculate_step_mz_range(0))

    def test_calculate_step_mz_range_negative_values(self):
        """Test m/z range calculation with negative values (should return None)."""
        trap = TrapControls()

        # Negative r0
        trap.basic_controls.r0_cm = -0.5
        self.assertIsNone(trap.calculate_step_mz_range(0))


class TestTrapControlsRoundTrip(unittest.TestCase):
    """Test round-trip conversions."""

    def test_dict_round_trip(self):
        """Test that to_dict -> from_dict preserves data."""
        original = TrapControls()
        original.basic_controls.r0_cm = 0.5
        original.basic_controls.enable_trapping = True
        original.excitation_steps = [
            ExcitationStep(
                step_name="Step1",
                step_duration_ms=150.0,
                enable_frequency_sweep=True,
            ),
            ExcitationStep(step_name="Step2", step_duration_ms=200.0),
        ]

        dict_form = original.to_dict()
        restored = TrapControls.from_dict(dict_form)

        self.assertEqual(restored.basic_controls.r0_cm, 0.5)
        self.assertTrue(restored.basic_controls.enable_trapping)
        self.assertEqual(len(restored.excitation_steps), 2)
        self.assertEqual(restored.excitation_steps[0].step_name, "Step1")
        self.assertEqual(restored.excitation_steps[0].step_duration_ms, 150.0)
        self.assertTrue(restored.excitation_steps[0].enable_frequency_sweep)

    def test_flat_dict_round_trip(self):
        """Test that to_flat_dict -> from_flat_dict preserves data."""
        original = TrapControls()
        original.basic_controls.r0_cm = 0.5
        original.basic_controls.enable_trapping = True
        original.excitation_steps[0].step_name = "FlatTest"
        original.excitation_steps[0].frequency_begin_kHz = 75.0

        flat_form = original.to_flat_dict()
        restored = TrapControls.from_flat_dict(flat_form)

        self.assertEqual(restored.basic_controls.r0_cm, 0.5)
        self.assertTrue(restored.basic_controls.enable_trapping)
        self.assertEqual(restored.excitation_steps[0].step_name, "FlatTest")
        self.assertEqual(restored.excitation_steps[0].frequency_begin_kHz, 75.0)


if __name__ == "__main__":
    unittest.main()
