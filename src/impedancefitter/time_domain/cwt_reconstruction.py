import logging

import numpy as np
import numpy.matlib
import scipy.fftpack

logger = logging.getLogger(__name__)


def cwt_morlet(dt, signal, f_min, f_max, nb=35, f0=1.0):
    """
    Calculate CWT using Morlet wavelet for a given signal.

    Parameters
    ----------
    dt : float
        Sampling period in seconds
    signal : numpy.ndarray
        Input signal (voltage or current)
    f_min : float
        Minimum frequency in Hz
    f_max : float
        Maximum frequency in Hz
    nb : int
        Number of frequency points (voices)
    f0 : float
        Morlet wavelet central frequency

    Returns
    -------
    dict : Contains wavelet coefficients and frequencies

    Notes
    -----
    Note that this an experimental, unoptimized implementation
    based on the code presented in [Nusev2021]_ and the theory
    described in [Boskoski2017]_.
    Please cite them accordingly.

    References
    ----------

    .. [Nusev2021] Nusev, Gjorgji, et al. (2021)
                    Condition monitoring of solid oxide fuel cells by fast
                    electrochemical impedance spectroscopy:
                    A case example of detecting deficiencies in fuel supply.
                    Journal of Power Sources 489: 229491.
                    https://doi.org/10.1016/j.jpowsour.2021.229491

    .. [Boskoski2017] Boškoski, P., Debenjak, A., & Mileva Boshkoska, B. (2017).
                      Fast electrochemical impedance spectroscopy.
                      In Fast Electrochemical Impedance Spectroscopy:
                      As a Statistical Condition Monitoring Tool (pp. 9-22).
                      Cham: Springer International Publishing.
    """
    # Prepare signal
    # signal = signal - np.mean(signal)
    if len(signal) % 2 != 0:
        signal = signal[:-1]
    # Calculate scales
    f0_rad = 2 * np.pi * f0
    morlet_fourier_factor = 4 * np.pi / (f0_rad + np.sqrt(2 + f0_rad**2))
    s0 = 1.0 / (f_max * morlet_fourier_factor)
    mx = 1.0 / (f_min * morlet_fourier_factor)
    ds = np.log(mx / s0) / np.log(2) / (nb - 1)
    scales = s0 * (2 ** (np.arange(0, nb) * ds))

    # FFT of signal
    fft_signal = scipy.fftpack.fft(signal)
    omega = 2.0 * np.pi * scipy.fftpack.fftfreq(len(signal), d=dt)

    # Create Morlet wavelet in frequency domain
    n_scales = len(scales)
    wft = np.zeros((n_scales, len(omega)))
    mul = (np.pi ** (-0.25)) * np.sqrt(omega[1]) * np.sqrt(len(omega))
    for idx_scale in range(n_scales):
        expnt = -0.5 * (scales[idx_scale] * omega - f0_rad) ** 2
        correct = -0.5 * (f0_rad**2 + (scales[idx_scale] * omega) ** 2)
        wft[idx_scale, :] = (
            mul * np.sqrt(scales[idx_scale]) * (np.exp(expnt) - np.exp(correct))
        )

    # Convolution via FFT
    cwtcfs = scipy.fftpack.ifft(
        np.multiply(np.matlib.repmat(fft_signal.T, n_scales, 1), wft)
    )
    cwtcfs = cwtcfs[:, : len(signal)]
    # apply cone of influence
    coi = scales * np.sqrt(2) / dt
    for i, coi_i in enumerate(coi):
        border_idx = int(np.ceil(coi_i))
        cwtcfs[i, :border_idx] = np.nan
        cwtcfs[i, -border_idx:] = np.nan
    # Calculate frequencies
    frequencies = f0 / scales
    return frequencies, cwtcfs


def calculate_impedance_spectrum_using_cwt(
    dt, voltage, current, f_min=0.1, f_max=1000, nb=35, f0=1.0
):
    """
    Calculate impedance spectrum from voltage and current signals using CWT.

    Parameters
    ----------
    voltage : numpy.array
        Voltage signal
    current : numpy.array
        Current signal
    dt : float
        Sampling period in seconds
    f_min : float
        Minimum frequency in Hz
    f_max : float
        Maximum frequency in Hz
    nb : int
        Number of frequency points
    f0: float
        Morlet wavelet central frequency

    Returns
    -------
    frequencies : numpy.array
        Frequency points in Hz
    impedance : numpy.array (complex)
        Complex impedance values

    Notes
    -----
    Note that this an experimental, unoptimized implementation
    based on the code presented in [Nusev2021]_ and the theory
    described in [Boskoski2017]_.
    Please cite them accordingly.

    """
    # Calculate CWT for both signals
    frequencies, u_cof = cwt_morlet(dt, voltage, f_min, f_max, nb, f0)
    _, i_cof = cwt_morlet(dt, current, f_min, f_max, nb, f0)

    # Calculate impedance Z = U/I
    s11 = np.nanmean(np.multiply(u_cof, np.conj(u_cof)), axis=1)
    s12 = np.nanmean(np.multiply(i_cof, np.conj(u_cof)), axis=1)
    s22 = np.nanmean(np.multiply(i_cof, np.conj(i_cof)), axis=1)
    rho = np.divide(np.divide(s12, np.sqrt(s11)), np.sqrt(s22))
    s1 = np.sqrt(s11)
    s2 = np.sqrt(s22)
    impedance = np.divide(np.multiply(np.conj(rho), s1), s2)

    return frequencies, impedance
