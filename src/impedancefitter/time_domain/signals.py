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
