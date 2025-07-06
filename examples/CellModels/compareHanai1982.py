# Figures 1 and 2
# from Analysis of dielectric relaxations of w/o emulsions
# in the light of theories of interfacial polarization
# by T. Hanai, T. Imakita, and N. Koizumi
import numpy as np
from scipy.constants import epsilon_0 as e0

import impedancefitter as ifit
from impedancefitter.suspensionmodels import bhcubic_eps_model, eps_sus_MW

freq = np.logspace(1, 7)
omega = 2.0 * np.pi * freq

# parameters for Hanai model
kmed = 0.023e-7
emed = 2.23
kp = 2.93e-4
ep = 79.6
c0 = 1e-12
p = 0.736

epsi_med = emed - 1j * kmed / (e0 * omega)
epsi_p = ep - 1j * kp / (e0 * omega)

epsc = bhcubic_eps_model(epsi_med, epsi_p, p)
eps_r = epsc.real
conductivity = -epsc.imag * e0 * omega
Z = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)

# parameters for Maxwell-Wagner model
kmed = 0.023e-7
emed = 2.23
kp = 1.28e-4
ep = 41.7
c0 = 1e-12
p = 0.947

epsi_med = emed - 1j * kmed / (e0 * omega)
epsi_p = ep - 1j * kp / (e0 * omega)

epscMW = eps_sus_MW(epsi_med, epsi_p, p)
epsMW_r = epscMW.real
conductivityMW = -epscMW.imag * e0 * omega
ZMW = ifit.utils.convert_diel_properties_to_impedance(
    omega, epsMW_r, conductivityMW, c0
)
ZMWfit = ifit.utils.convert_diel_properties_to_impedance(
    omega, epsMW_r, conductivityMW, c0
)
ifit.plot_dielectric_properties(
    omega, Z, c0, Z_comp=ZMWfit, logscale=None, labels=["Hanai", "MW"]
)

Z = ifit.utils.convert_diel_properties_to_impedance(
    omega, eps_r, conductivity - conductivity[0], c0
)
ZMW = ifit.utils.convert_diel_properties_to_impedance(
    omega, epsMW_r, conductivityMW - conductivityMW[0], c0
)
ZMWfit = ifit.utils.convert_diel_properties_to_impedance(
    omega, epsMW_r, conductivityMW - conductivityMW[0], c0
)
ifit.plot_cole_cole(omega, Z, c0, Z_comp=ZMW, labels=["Hanai", "MW"])
