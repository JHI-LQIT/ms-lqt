from dataclasses import asdict, dataclass


@dataclass
class ExcitationStep:
    """Data structure for a single excitation step in a frequency sweep."""

    step_name: str = ""
    step_active: bool = True
    step_delay_ms: float = 0.0
    step_duration_ms: float = 100.0

    enable_frequency_sweep: bool = False
    frequency_begin_kHz: float = 50.0
    frequency_end_kHz: float = 100.0

    enable_ac_amplitude_sweep: bool = False
    ac_amplitude_begin_mV: float = 0.0
    ac_amplitude_end_mV: float = 0.0

    enable_ac_amplitude_soft_start: bool = False
    ac_soft_start_duration_ms: float = 50.0

    enable_rf_amplitude_sweep: bool = False
    rf_amplitude_begin_V0p: float = 0.0
    rf_amplitude_end_V0p: float = 0.0


@dataclass
class BasicTrapControls:
    """Data structure for trap control parameters."""

    # Trap dimensions
    r0_cm: float = (
        0.475  # Characteristic trap dimension in cm, used for calculating secular frequencies
    )
    f_RF_MHz: float = 2.112  # RF drive frequency in MHz

    # RF parameters
    rf_amplitude_V0p: float = 0.0

    # Injection parameters
    injection_duration_ms: float = 5.0
    injection_open_voltage_V: float = 0.0
    injection_close_voltage_V: float = 5.0
    injection_rf_amplitude_V: float = 0.0

    # Trapping parameters
    trapping_duration_ms: float = 100.0
    trapping_rf_amplitude_V0p: float = 0.0

    # Extraction parameters
    extraction_duration_ms: float = 5.0
    extraction_open_voltage_V: float = 0.0
    extraction_close_voltage_V: float = 5.0
    turn_off_rf_after_extraction: bool = False
    turn_off_rf_duration_ms: float = 10.0

    # Control toggles
    enable_trapping: bool = False

    # DC Compensation parameters
    Uax_V: float = 0.0
    Udiff_mV: float = 0.0
    UABdiff_mV: float = 0.0
    UCDdiff_mV: float = 0.0


class TrapControls:
    """Data structure for all trap control parameters, including excitation steps."""

    def __init__(self):
        self.basic_controls = BasicTrapControls()
        self.excitation_steps = [ExcitationStep()]

    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "basic_controls": asdict(self.basic_controls),
            "excitation_steps": [asdict(step) for step in self.excitation_steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrapControls":
        """Create TrapControls instance from structured dictionary.

        Args:
            data: Dictionary with 'basic_controls' and 'excitation_steps' keys

        Returns:
            TrapControls instance
        """
        trap_controls = cls()

        # Populate basic controls
        if "basic_controls" in data and data["basic_controls"]:
            trap_controls.basic_controls = BasicTrapControls(**data["basic_controls"])

        # Populate excitation steps
        if "excitation_steps" in data and data["excitation_steps"]:
            trap_controls.excitation_steps = [
                ExcitationStep(**step_data) for step_data in data["excitation_steps"]
            ]

        return trap_controls

    def to_flat_dict(self):
        """
        Convert to flat dictionary with all parameters at top level.

        This is used for compatibility with existing code that expects a flat dict.
        Currently uses only the first excitation step (index 0).
        """
        result = {}

        # Add basic controls
        for key, value in asdict(self.basic_controls).items():
            result[key] = value

        # Add first excitation step parameters
        if self.excitation_steps:
            for key, value in asdict(self.excitation_steps[0]).items():
                result[key] = value

        return result

    @classmethod
    def from_flat_dict(cls, data: dict) -> "TrapControls":
        """
        Create TrapControls instance from flat dictionary.

        This is used for compatibility with existing code that uses flat dicts.
        Creates a TrapControls with one excitation step.

        Args:
            data: Flat dictionary with all parameters

        Returns:
            TrapControls instance
        """
        trap_controls = cls()

        # Get field names for each dataclass
        basic_fields = {f.name for f in BasicTrapControls.__dataclass_fields__.values()}
        excitation_fields = {
            f.name for f in ExcitationStep.__dataclass_fields__.values()
        }

        # Populate basic controls
        basic_data = {}
        for key, value in data.items():
            if key in basic_fields:
                # Convert string values to appropriate types
                if isinstance(value, str):
                    field_type = BasicTrapControls.__dataclass_fields__[key].type
                    if field_type == float:
                        value = float(value)
                    elif field_type == bool:
                        value = value.lower() in ("true", "1", "yes")
                basic_data[key] = value

        if basic_data:
            trap_controls.basic_controls = BasicTrapControls(**basic_data)

        # Populate excitation step
        excitation_data = {}
        for key, value in data.items():
            if key in excitation_fields:
                # Convert string values to appropriate types
                if isinstance(value, str):
                    field_type = ExcitationStep.__dataclass_fields__[key].type
                    if field_type == float:
                        value = float(value)
                    elif field_type == bool:
                        value = value.lower() in ("true", "1", "yes")
                excitation_data[key] = value

        if excitation_data:
            trap_controls.excitation_steps = [ExcitationStep(**excitation_data)]

        return trap_controls

    def calculate_step_start_time(self, step_index: int) -> float:
        """Calculate the start time of a step based on ALL previous steps (including inactive).

        Args:
            step_index: Index of the step to calculate start time for

        Returns:
            Start time of the step in milliseconds
        """
        basic = self.basic_controls
        start_time = basic.injection_duration_ms

        # Account for all steps before this one, regardless of active status
        for i in range(step_index):
            step = self.excitation_steps[i]
            if i == 0:
                start_time += step.step_delay_ms
            start_time += step.step_duration_ms

        if step_index > 0:
            # For steps after the first, add their delay
            start_time += self.excitation_steps[step_index].step_delay_ms
        else:
            # For the first step, add its delay
            start_time += self.excitation_steps[0].step_delay_ms

        return start_time

    def calculate_total_duration(self) -> float:
        """Calculate total duration of the trapping cycle.

        Returns:
            Total duration in milliseconds
        """
        basic = self.basic_controls

        total = (
            basic.injection_duration_ms
            + basic.trapping_duration_ms
            + basic.extraction_duration_ms
        )

        # Add RF off duration if enabled
        if basic.turn_off_rf_after_extraction:
            total += basic.turn_off_rf_duration_ms

        return total

    def calculate_step_mz_range(self, step_index: int) -> tuple[float, float] | None:
        """Calculate m/z range for a given excitation step.

        The m/z calculation uses the formula:
        m/z = 34.45 * V_RF / (r0^2 * f_RF * f_secular)

        Args:
            step_index: Index of the excitation step

        Returns:
            Tuple of (mz_begin, mz_end) or None if calculation not possible
            (e.g., due to zero values or invalid parameters)
        """
        if step_index < 0 or step_index >= len(self.excitation_steps):
            return None

        excitation = self.excitation_steps[step_index]
        r0_cm = self.basic_controls.r0_cm
        f_RF_MHz = self.basic_controls.f_RF_MHz

        # Determine RF amplitude values
        if excitation.enable_rf_amplitude_sweep:
            rf_begin = excitation.rf_amplitude_begin_V0p
            rf_end = excitation.rf_amplitude_end_V0p
        else:
            rf_begin = rf_end = excitation.rf_amplitude_begin_V0p

        # Determine frequency values
        if excitation.enable_frequency_sweep:
            freq_begin = excitation.frequency_begin_kHz
            freq_end = excitation.frequency_end_kHz
        else:
            freq_begin = freq_end = excitation.frequency_begin_kHz

        # Check for division by zero
        if r0_cm <= 0 or f_RF_MHz <= 0 or freq_begin <= 0 or freq_end <= 0:
            return None

        # Calculate m/z values using the formula: m/z = 34.45 * V_RF / (r0^2 * f_RF * f_secular)
        mz_begin = 34.45 * rf_begin / (r0_cm**2 * f_RF_MHz * freq_begin)
        mz_end = 34.45 * rf_end / (r0_cm**2 * f_RF_MHz * freq_end)

        return (mz_begin, mz_end)


# Legacy class for backward compatibility
@dataclass
class TrappingParameters:
    """Data class for trapping parameters (legacy, for backward compatibility)."""

    secular_excitation_enabled: bool
    secular_excitation_begin_ms: float
    secular_excitation_end_ms: float
    frequency_begin_khz: float
    frequency_end_khz: float
    frequency_sweep: bool
    rf_amplitude_begin_v0p: float
    rf_amplitude_end_v0p: float
    rf_amplitude_sweep: bool
    r0_cm: float
    f_RF_MHz: float
