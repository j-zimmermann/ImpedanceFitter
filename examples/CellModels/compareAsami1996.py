# Figure 2

import numpy as np
import impedancefitter as ifit
from impedancefitter.suspensionmodels import bhcubic_eps_model, eps_sus_MW
from impedancefitter.single_shell_wall import eps_cell_single_shell_wall
from scipy.constants import epsilon_0 as e0

em = 5.
dw = 250e-9
Rc = 2.75e-6 - dw
dm = 7e-9
km = 0.0
kcp = 0.5
ecp = 50
kmed = 0.3
emed = 78.
c0 = 1e-12
p = 0.5
ew = 60
kw = 1.0 * kmed

freq = np.logspace(5, 9, num=100)
omega = 2. * np.pi * freq

# cell permittivities
eps_c = eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw)
epsi_med = emed - 1j * kmed / (e0 * omega)
esus = eps_sus_MW(epsi_med, eps_c, p)
Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
Z_fit = 1 / Ys

epsc = bhcubic_eps_model(epsi_med, eps_c, p)
eps_r = epsc.real
conductivity = -epsc.imag * e0 * omega
Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)

ifit.plot_dielectric_properties(omega, Zcubic, c0, Z_comp=Z_fit, labels=["Cubic", "MW"], limits=[(50, 3000), (0.05, 0.5)])

kwlist = [1.0 * kmed, 0.25 * kmed, 0.1 * kmed]
for kw in kwlist:
    eps_c = eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw)
    epsi_med = emed - 1j * kmed / (e0 * omega)
    epsc = bhcubic_eps_model(epsi_med, eps_c, p)
    # epsc = eps_sus_MW(epsi_med, eps_c, p)
    eps_r = epsc.real
    conductivity = -epsc.imag * e0 * omega

    Z = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)
    label = "{:.2f}".format(kw / kmed)
    if np.isclose(kw, kwlist[-1]):
        append = False
        show = True
    else:
        append = True
        show = False
    ifit.plot_dielectric_properties(omega, Z, c0, labels=[label, ""], limits=[(50, 3000), (0.05, .5)], append=append, show=show)
