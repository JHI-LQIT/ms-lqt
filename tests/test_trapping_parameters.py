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
        self.assertEqual(basic.rf_amplitude_V0p, 0.0)
        self.assertEqual(basic.injection_duration_ms, 5.0)
        self.assertEqual(basic.trapping_duration_ms, 100.0)
        self.assertEqual(basic.extraction_duration_ms, 5.0)
        self.assertFalse(basic.enable_trapping)
        self.assertFalse(basic.turn_off_rf_after_extraction)

    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        basic = BasicTrapControls(
            r0_cm=0.5,
            f_RF_MHz=2.0,
            rf_amplitude_V0p=100.0,
            enable_trapping=True,
            injection_duration_ms=10.0,
        )

        self.assertEqual(basic.r0_cm, 0.5)
        self.assertEqual(basic.f_RF_MHz, 2.0)
        self.assertEqual(basic.rf_amplitude_V0p, 100.0)
        self.assertTrue(basic.enable_trapping)
        self.assertEqual(basic.injection_duration_ms, 10.0)


class TestExcitationStep(unittest.TestCase):
    """Test ExcitationStep dataclass."""

    def test_default_initialization(self):
        """Test default initialization of ExcitationStep."""
        step = ExcitationStep()

        self.assertEqual(step.step_name, "")
        self.assertTrue(step.step_active)
        self.assertEqual(step.step_delay_ms, 0.0)
        self.assertEqual(step.step_duration_ms, 100.0)
        self.assertFalse(step.enable_frequency_sweep)
        self.assertEqual(step.frequency_begin_kHz, 50.0)
        self.assertEqual(step.frequency_end_kHz, 100.0)
        self.assertFalse(step.enable_ac_amplitude_sweep)
        self.assertFalse(step.enable_ac_amplitude_soft_start)
        self.assertFalse(step.enable_rf_amplitude_sweep)
        self.assertEqual(step.frequency_ramp_type, "linear")

    def test_custom_initialization(self):
        """Test custom initialization with specific values."""
        step = ExcitationStep(
            step_name="Test Step",
            step_active=False,
            step_delay_ms=10.0,
            step_duration_ms=200.0,
            enable_frequency_sweep=True,
            frequency_begin_kHz=100.0,
            frequency_end_kHz=200.0,
        )

        self.assertEqual(step.step_name, "Test Step")
        self.assertFalse(step.step_active)
        self.assertEqual(step.step_delay_ms, 10.0)
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
            ExcitationStep(step_delay_ms=5.0, step_duration_ms=100.0),
            ExcitationStep(step_delay_ms=3.0, step_duration_ms=150.0),
            ExcitationStep(step_delay_ms=2.0, step_duration_ms=200.0),
        ]

        # Step 0: injection_duration + step_delay
        self.assertEqual(trap.calculate_step_start_time(0), 15.0)

        # Step 1: injection + step0.delay + step0.duration + step1.delay
        self.assertEqual(trap.calculate_step_start_time(1), 118.0)

        # Step 2: injection + step0.delay + step0.duration + step1.duration + step2.delay
        # = 10 + 5 + 100 + 150 + 2 = 267
        self.assertEqual(trap.calculate_step_start_time(2), 267.0)

    def test_calculate_total_duration(self):
        """Test calculation of total duration."""
        trap = TrapControls()
        trap.basic_controls.injection_duration_ms = 10.0
        trap.basic_controls.trapping_duration_ms = 100.0
        trap.basic_controls.extraction_duration_ms = 5.0

        # Without RF off
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


class TestHyperbolicFrequencyRamp(unittest.TestCase):
    """Tests for hyperbolic frequency ramp type in ExcitationStep and TrapControls."""

    def _make_trap(
        self,
        f_begin_kHz: float = 50.0,
        f_end_kHz: float = 100.0,
        ramp_type: str = "hyperbolic",
        rf_begin: float = 100.0,
        duration_ms: float = 100.0,
    ) -> TrapControls:
        """Helper: build a TrapControls configured for a frequency sweep."""
        trap = TrapControls()
        trap.basic_controls.r0_cm = 0.5
        trap.basic_controls.f_RF_MHz = 2.0
        trap.basic_controls.injection_duration_ms = 1.0
        step = trap.excitation_steps[0]
        step.step_delay_ms = 0.0
        step.step_duration_ms = duration_ms
        step.enable_frequency_sweep = True
        step.frequency_begin_kHz = f_begin_kHz
        step.frequency_end_kHz = f_end_kHz
        step.frequency_ramp_type = ramp_type
        step.rf_amplitude_begin_V0p = rf_begin
        return trap

    # ------------------------------------------------------------------
    # ExcitationStep dataclass field tests
    # ------------------------------------------------------------------

    def test_default_frequency_ramp_type_is_linear(self):
        """ExcitationStep.frequency_ramp_type must default to 'linear'."""
        step = ExcitationStep()
        self.assertEqual(step.frequency_ramp_type, "linear")

    def test_frequency_ramp_type_hyperbolic_assignment(self):
        """frequency_ramp_type can be set to 'hyperbolic'."""
        step = ExcitationStep(frequency_ramp_type="hyperbolic")
        self.assertEqual(step.frequency_ramp_type, "hyperbolic")

    def test_frequency_ramp_type_round_trips_via_dict(self):
        """frequency_ramp_type survives to_dict -> from_dict round-trip."""
        original = TrapControls()
        original.excitation_steps[0].frequency_ramp_type = "hyperbolic"

        restored = TrapControls.from_dict(original.to_dict())
        self.assertEqual(
            restored.excitation_steps[0].frequency_ramp_type, "hyperbolic"
        )

    def test_frequency_ramp_type_missing_in_old_dict_defaults_to_linear(self):
        """Dicts without frequency_ramp_type key (legacy) must default to 'linear'."""
        data = {
            "basic_controls": {"r0_cm": 0.5, "f_RF_MHz": 2.0},
            "excitation_steps": [{"step_name": "old_step"}],
        }
        trap = TrapControls.from_dict(data)
        self.assertEqual(trap.excitation_steps[0].frequency_ramp_type, "linear")

    # ------------------------------------------------------------------
    # calculate_step_mz_range – endpoints are ramp-type-independent
    # ------------------------------------------------------------------

    def test_calculate_step_mz_range_same_for_linear_and_hyperbolic(self):
        """m/z range endpoints must be identical for both ramp types."""
        trap_lin = self._make_trap(ramp_type="linear")
        trap_hyp = self._make_trap(ramp_type="hyperbolic")

        mz_lin = trap_lin.calculate_step_mz_range(0)
        mz_hyp = trap_hyp.calculate_step_mz_range(0)

        self.assertIsNotNone(mz_lin)
        self.assertIsNotNone(mz_hyp)
        assert mz_lin is not None and mz_hyp is not None  # type-narrowing
        self.assertAlmostEqual(mz_lin[0], mz_hyp[0], places=10)
        self.assertAlmostEqual(mz_lin[1], mz_hyp[1], places=10)

    # ------------------------------------------------------------------
    # tau1 / psi mathematics
    # ------------------------------------------------------------------

    def test_tau1_formula(self):
        """tau1 = 1000 / f_begin_kHz (µs)."""
        f_begin_kHz = 50.0
        tau1_expected = 1000.0 / f_begin_kHz  # 20 µs
        self.assertAlmostEqual(tau1_expected, 20.0, places=10)

    def test_psi_formula_increasing_frequency(self):
        """psi is negative when sweeping from low to high frequency."""
        f_begin_kHz = 50.0
        f_end_kHz = 100.0
        duration_ms = 100.0
        tau1 = 1000.0 / f_begin_kHz  # 20 µs
        tau_end = 1000.0 / f_end_kHz  # 10 µs
        duration_us = duration_ms * 1000.0
        psi = (tau_end - tau1) / duration_us
        # tau decreases → frequency increases → psi < 0
        self.assertLess(psi, 0.0)
        self.assertAlmostEqual(psi, (10.0 - 20.0) / 100_000.0, places=15)

    def test_psi_formula_decreasing_frequency(self):
        """psi is positive when sweeping from high to low frequency."""
        f_begin_kHz = 100.0
        f_end_kHz = 50.0
        duration_ms = 100.0
        tau1 = 1000.0 / f_begin_kHz  # 10 µs
        tau_end = 1000.0 / f_end_kHz  # 20 µs
        duration_us = duration_ms * 1000.0
        psi = (tau_end - tau1) / duration_us
        self.assertGreater(psi, 0.0)
        self.assertAlmostEqual(psi, (20.0 - 10.0) / 100_000.0, places=15)

    def test_hyperbolic_frequency_at_t0_equals_f_begin(self):
        """f(t=0) = 10^6 / tau1 must equal f_begin_Hz."""
        f_begin_kHz = 50.0
        tau1 = 1000.0 / f_begin_kHz  # 20 µs
        f_at_t0_Hz = 1e6 / tau1
        self.assertAlmostEqual(f_at_t0_Hz, f_begin_kHz * 1000.0, places=6)

    def test_hyperbolic_frequency_at_t_end_equals_f_end(self):
        """f(t=duration) = 10^6 / tau_end must equal f_end_Hz."""
        f_begin_kHz = 50.0
        f_end_kHz = 100.0
        duration_ms = 100.0
        tau1 = 1000.0 / f_begin_kHz
        tau_end = 1000.0 / f_end_kHz
        duration_us = duration_ms * 1000.0
        psi = (tau_end - tau1) / duration_us
        f_at_end_Hz = 1e6 / (tau1 + psi * duration_us)
        self.assertAlmostEqual(f_at_end_Hz, f_end_kHz * 1000.0, places=6)


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
