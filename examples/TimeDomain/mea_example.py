import lmfit
import numpy as np
import pandas as pd

import impedancefitter as ifit
from impedancefitter.time_domain import (
    calculate_impedance_spectrum_using_cwt,
    calculate_impedance_spectrum_using_fft,
    fit_impedance_from_time_domain,
)

data = pd.read_csv("MEA_signal.zip")
time = data["time"].to_numpy()
dt = time[1] - time[0]

# use only short signal
t = time[:55000]
voltage = data["V"][:55000].to_numpy()
current = data["I"][:55000].to_numpy()

ecm_model = "R_tis + W_war + parallel(R_ct, C_dl)"
ecm = ifit.get_equivalent_circuit_model(ecm_model)

ecm_parameters = {
    "dl_C": 1.1344484471382533e-07,
    "ct_R": 2156394.136657303,
    "war_Aw": 2531099.2267359486,
    "tis_R": 9887.767470472341,
}

# Calculate impedance spectrum
frequencies, impedance = calculate_impedance_spectrum_using_cwt(
    dt, voltage, current, f_min=1e5, f_max=1e6, nb=20, f0=1.0
)

frequencies_tmp, impedance_tmp = calculate_impedance_spectrum_using_cwt(
    dt, voltage, current, f_min=1e4, f_max=1e5, nb=20, f0=1.0
)

frequencies = np.append(np.flip(frequencies_tmp), np.flip(frequencies))
impedance = np.append(np.flip(impedance_tmp), np.flip(impedance))

# undersample at low frequencies
frequencies_tmp, impedance_tmp = calculate_impedance_spectrum_using_cwt(
    dt * 4, voltage[::4], current[::4], f_min=1e3, f_max=1e4, nb=20, f0=0.5
)

frequencies = np.append(np.flip(frequencies_tmp), frequencies)
impedance = np.append(np.flip(impedance_tmp), impedance)


ifit.plot_time_domain_signals_with_impedance(
    t,
    frequencies,
    voltage,
    current,
    impedance,
    impedance_expected=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    save_file="impedance_cwt_MEA.pdf",
    current_scale=5e3,
    t_zoom_range=(0, 0.00005),
)

ifit.plot_impedance(
    omega=2.0 * np.pi * frequencies,
    Z_fit=impedance,
    Z=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    residual="absolute",
    show=True,
)

ecm_fit_parameters = {
    "dl_C": {"value": 1.1344484471382533e-07, "vary": False},
    "ct_R": {"value": 12156394.136657303},
    "war_Aw": {"value": 12531099.2267359486},
    "tis_R": {"value": 19887.767470472341},
}


fit_result = fit_impedance_from_time_domain(
    ecm_model, ecm_fit_parameters, dt, voltage, current
)
print(lmfit.fit_report(fit_result))

ifit.plot_time_domain_signals_with_impedance(
    t,
    frequencies,
    voltage,
    current,
    ecm.eval(omega=2.0 * np.pi * frequencies, **fit_result.params.valuesdict()),
    impedance_expected=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    save_file="impedance_td_fit_MEA.pdf",
    current_scale=5e3,
    t_zoom_range=(0, 0.00005),
)


# use longer signal for FFT
t = time
voltage = data["V"].to_numpy()
current = data["I"].to_numpy()

frequencies, impedance = calculate_impedance_spectrum_using_fft(
    dt, voltage, current, current_threshold=0.1
)

ifit.plot_time_domain_signals_with_impedance(
    t,
    frequencies,
    voltage,
    current,
    impedance,
    impedance_expected=ecm.eval(omega=2.0 * np.pi * frequencies, **ecm_parameters),
    save_file="impedance_fft_MEA.pdf",
    current_scale=5e3,
    t_zoom_range=(0, 0.00005),
)
