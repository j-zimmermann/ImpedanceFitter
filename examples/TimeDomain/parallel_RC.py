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
frequency = 10  # Hz
t = np.arange(0, 1 / frequency, dt)

voltage = rectangle(dt, 1.0, 0.01, 0.01, 0, frequency)
ecm_model = "parallel(R, C)"
ecm = ifit.get_equivalent_circuit_model(ecm_model)
ecm_parameters = {"R": 500.2, "C": 202.38e-9}
current = predict_time_domain_signal(dt, voltage, ecm, ecm_parameters)

# Calculate impedance spectrum
# f_min and f_max are chosen according to the power spectrum of the excitation pulse
# f0 can be adjusted
frequencies, impedance = calculate_impedance_spectrum_using_cwt(
    dt, voltage, current, f_min=50, f_max=5000, nb=20, f0=1.0
)

ifit.plot_time_domain_signals_with_impedance(
    t,
    frequencies,
    voltage,
    current,
    impedance,
    impedance_expected=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    save_file="impedance_cwt_parallel_RC.pdf",
    current_scale=50,
    t_zoom_range=(0.005, 0.025),
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
    save_file="impedance_td_fit_parallel_RC.pdf",
    current_scale=50,
    t_zoom_range=(0.005, 0.025),
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
    save_file="impedance_fft_parallel_RC.pdf",
    current_scale=50,
    t_zoom_range=(0.005, 0.025),
)
