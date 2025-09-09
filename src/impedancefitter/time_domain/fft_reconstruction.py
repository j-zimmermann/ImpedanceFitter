import lmfit
import numpy as np

from impedancefitter import get_equivalent_circuit_model


def calculate_impedance_spectrum_using_fft(
    dt: float, voltage: np.ndarray, current: np.ndarray, current_threshold=0.1
):
    """Extract impedance spectrum from time domain data using the FFT.

    Parameters
    ----------
    dt : float
        Time step between samples in seconds.
    voltage : np.ndarray
        Time-domain voltage signal.
    current : np.ndarray
        Time-domain current signal.
    current_threshold : float, optional
        Threshold with respect to maximum current in frequency domain
        to avoid division by noise (default is 0.1).

    Returns
    -------
    frequencies : np.ndarray
        Array of frequency components in Hz.
    impedance : np.ndarray
        Complex impedance values corresponding to each frequency.

    Notes
    -----
    The implemented approach relies on [Zimmermann2021]_ and [Erbsloeh2025]_.

    References
    ----------

    .. [Zimmermann2021] Zimmermann, J., Budde, K., Arbeiter, N., Molina, F.,
                        Storch, A., Uhrmacher, A. M., & van Rienen, U. (2021).
                        Using a digital twin of an electrical stimulation device
                        to monitor and control the electrical stimulation of cells
                        in vitro.
                        Frontiers in bioengineering and biotechnology, 9, 765516.
                        https://doi.org/10.3389/fbioe.2021.765516

    .. [Erbsloeh2025] Erbsloeh, A., Zimmermann, J., Ingebrandt, S., Mokwa, W.,
                      Seidl, K., van Rienen, U., ... & Kokozinski, R. (2025).
                      Prediction of impedance characteristic during electrical
                      stimulation with microelectrode arrays.
                      Journal of Neural Engineering, 22(2), 026056.
                      http://doi.org/10.1088/1741-2552/adc2d5

    """
    fft_freqs = np.fft.rfftfreq(len(voltage), d=dt)
    fft_current = np.fft.rfft(current)
    fft_voltage = np.fft.rfft(voltage)
    # this factor is the amplitude that the current has to have
    # compared to the maximal current
    ignore_indices = np.abs(fft_current) < (
        current_threshold * np.max(np.abs(fft_current))
    )
    Z = fft_voltage / fft_current
    Z[ignore_indices] = np.nan + 1j * np.nan

    return fft_freqs, Z


def predict_time_domain_signal(
    timestep: float,
    excitation_signal: np.ndarray,
    ecm: lmfit.Model,
    ecm_parameters: dict,
    mode: str = "VC",
    offset: float = 0.0,
):
    """TODO."""
    if mode not in ["VC", "CC"]:
        raise ValueError(f"Mode has to be either VC or CC, not {mode}")
    Is = np.fft.rfft(excitation_signal)
    omega = 2.0 * np.pi * np.fft.rfftfreq(n=len(excitation_signal), d=timestep)
    Z = ecm.eval(omega=omega, **ecm_parameters)

    if mode == "VC":
        Is /= Z
        # take away the DC component if NaN or zero
        if np.isnan(Is[0]) or np.isclose(Is[0], 0.0):
            Is[0] = 0j
        measured_signal = np.fft.irfft(Is, n=len(excitation_signal))
    elif mode == "CC":
        # take away the DC component if NaN
        if np.isnan(Z[0]):
            Z[0] = 0j
        measured_signal = np.fft.irfft(Z * Is, n=len(excitation_signal))
    return measured_signal - offset


def fit_impedance_from_time_domain(
    ecm_model: str,
    parameters: dict,
    dt: float,
    excitation_signal: np.ndarray,
    measured_signal: np.ndarray,
    mode: str = "VC",
):
    """TODO."""
    if mode not in ["VC", "CC"]:
        raise ValueError(f"Mode has to be either VC or CC, not {mode}")
    freqs = np.fft.rfftfreq(excitation_signal.size, d=dt)
    fft_excitation_signal = np.fft.rfft(excitation_signal)
    ecm = get_equivalent_circuit_model(ecm_model)
    fit_params = lmfit.create_params(**parameters)
    # parameter values need to be positive
    for param in fit_params:
        if fit_params[param].min < 0:
            fit_params[param].min = 0
    fit_params.add(name="offset", value=0.0)

    # get ecm and Is from function
    def _time_domain_residual(pars, x):
        # unpack parameters: extract .value attribute for each parameter
        parvals = pars.valuesdict()
        # offset to shift curve
        offset = parvals["offset"]
        Z = ecm.eval(omega=x, **parvals)
        # predict voltage
        if mode == "CC":
            # take away the DC component if NaN
            if np.isnan(Z[0]):
                Z[0] = 0j

            predicted_signal = np.fft.irfft(
                Z * fft_excitation_signal, n=len(excitation_signal)
            )
        elif mode == "VC":
            Is = fft_excitation_signal / Z
            # take away the DC component if NaN or zero
            if np.isnan(Is[0]) or np.isclose(Is[0], 0.0):
                Is[0] = 0j

            predicted_signal = np.fft.irfft(Is, n=len(excitation_signal))

        return predicted_signal - offset - measured_signal

    fit_result = lmfit.minimize(
        _time_domain_residual, fit_params, args=(2.0 * np.pi * freqs,)
    )
    return fit_result
