import lmfit
import numpy as np

import impedancefitter as ifit
from impedancefitter.time_domain import (
    calculate_impedance_spectrum_using_cwt,
    calculate_impedance_spectrum_using_fft,
    fit_impedance_from_time_domain,
    predict_time_domain_signal,
    rectangle,
)

fs = 10000  # 10 kHz sampling
dt = 1.0 / fs
frequency = 0.1  # Hz
t = np.arange(0, 1 / frequency, dt)

voltage = rectangle(dt, 1.0, 0.1, 0.1, 0, frequency)
ecm_model = "R + C"
ecm = ifit.get_equivalent_circuit_model(ecm_model)
ecm_parameters = {"R": 1.0, "C": 0.05}
current = predict_time_domain_signal(dt, voltage, ecm, ecm_parameters)

# Calculate impedance spectrum
frequencies, impedance = calculate_impedance_spectrum_using_cwt(
    dt, voltage, current, f_min=1.0, f_max=500, nb=35, f0=0.5
)

ifit.plot_time_domain_signals_with_impedance(
    t,
    frequencies,
    voltage,
    current,
    impedance,
    impedance_expected=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    save_file="impedance_cwt_low_F.pdf",
    current_scale=5,
)

ifit.plot_impedance(
    omega=2.0 * np.pi * frequencies,
    Z_fit=impedance,
    Z=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    residual="absolute",
    show=True,
)

fit_result = fit_impedance_from_time_domain(
    ecm_model, ecm_parameters, dt, voltage, current
)
print(lmfit.fit_report(fit_result))

ifit.plot_time_domain_signals_with_impedance(
    t,
    frequencies,
    voltage,
    current,
    ecm.eval(omega=2.0 * np.pi * frequencies, **fit_result.params.valuesdict()),
    impedance_expected=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    save_file="impedance_td_fit_low_F.pdf",
    current_scale=5,
)


# use longer signal for FFT
t = np.arange(0, 7 / frequency, dt)  # 7 seconds of data
voltage = np.tile(voltage, 7)
current = np.tile(current, 7)

frequencies, impedance = calculate_impedance_spectrum_using_fft(
    dt, voltage, current, current_threshold=0.05
)

ifit.plot_time_domain_signals_with_impedance(
    t,
    frequencies,
    voltage,
    current,
    impedance,
    impedance_expected=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    save_file="impedance_fft_low_F.pdf",
    current_scale=5,
)
