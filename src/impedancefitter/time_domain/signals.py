import numpy as np


def rectangle(
    timestep: float,
    amplitude: float,
    start_gap: float,
    pulse_width: float,
    interpulse_gap: float,
    frequency: float,
    biphasic: bool = False,
):
    """Rectangular pulse."""
    if timestep > 1.0 / (10.0 * frequency):
        raise ValueError(
            "The provided time step fits less than ten times in the "
            "characteristic time of the pulse, choose a smaller timestep."
        )
    duration = 1.0 / frequency
    n_steps = int(np.round(duration / timestep))
    signal = np.zeros(n_steps)
    for idx in range(n_steps):
        time = idx * timestep
        if start_gap < time < start_gap + pulse_width:
            signal[idx] = amplitude
        elif (
            start_gap + pulse_width + interpulse_gap
            < time
            < start_gap + 2.0 * pulse_width + interpulse_gap
        ):
            if biphasic:
                signal[idx] = -amplitude
    return signal


def biphasic_with_time_shift(
    timestep,
    amplitude,
    step_time,
    pulse_width,
    interpulse_gap,
    frequency,
    total_time,
    time_shift=0.0,
    biphasic=True,
    start_negative=True,
):
    """
    Generate a biphasic (or monophasic) pulse train with an optional time shift.

    Parameters
    ----------
    timestep : float
        Time step size of the signal in seconds.
    amplitude : float
        Pulse amplitude.
    step_time : float
        Time at which pulses start (in seconds).
    pulse_width : float
        Duration of each pulse phase (in seconds).
    interpulse_gap : float
        Gap between phases in a biphasic pulse (in seconds).
    frequency : float
        Repetition frequency of pulses (in Hz).
    total_time : float
        Total duration of the signal (in seconds).
    time_shift : float, optional
        Time shift applied to the signal (default is 0.0).
    biphasic : bool, optional
        If True, generate biphasic pulses; if False, monophasic (default is True).
    start_negative : bool, optional
        If True, first phase starts with negative amplitude (default is True).

    Returns
    -------
    numpy.ndarray
        Array of signal values representing the pulse train.
    """
    num_points = int(np.round(total_time / timestep))
    t = np.arange(num_points) * timestep

    # Apply time shift
    t_shifted = t + time_shift

    signal = np.zeros(num_points)
    initial_amp = -amplitude if start_negative else amplitude

    # Constant amplitude before step_time (now in shifted time)
    signal[t_shifted < step_time] = initial_amp

    # Pulse parameters
    period = 1.0 / frequency
    pulse_start_time = step_time
    phase_signs = [1, -1] if not start_negative else [-1, 1]

    # Generate pulses starting at step_time in shifted time frame
    while pulse_start_time < total_time + time_shift:
        # First phase
        first_start = pulse_start_time
        first_end = first_start + pulse_width
        signal[(t_shifted >= first_start) & (t_shifted < first_end)] = (
            amplitude * phase_signs[0]
        )

        if biphasic:
            # Second phase
            second_start = first_end + interpulse_gap
            second_end = second_start + pulse_width
            signal[(t_shifted >= second_start) & (t_shifted < second_end)] = (
                amplitude * phase_signs[1]
            )

        pulse_start_time += period

    return signal
