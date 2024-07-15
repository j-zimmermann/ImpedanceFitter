# Figure 2

import numpy as np
from scipy.constants import epsilon_0 as e0

import impedancefitter as ifit
from impedancefitter.double_shell_wall import eps_cell_double_shell_wall
from impedancefitter.suspensionmodels import bhcubic_eps_model, eps_sus_MW

em = 5.0
R0 = 2.75e-6
dw = 250e-9
Rc = 2.75e-6 - dw
dm = 7e-9
km = 0.0
kcp = 0.5
ecp = 50
ene = 5
kne = 0
knp = 1.0
enp = 50
kmed = 0.3
emed = 78.0
c0 = 1e-12
p = 0.5
ew = 60
kw = 1.0 * kmed
dn = 7e-9
Rn = 0.3 * Rc

freq = np.logspace(5, 9, num=100)
omega = 2.0 * np.pi * freq

# cell permittivities
eps_c = eps_cell_double_shell_wall(
    omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw
)
epsi_med = emed - 1j * kmed / (e0 * omega)
esus = eps_sus_MW(epsi_med, eps_c, p)
Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
Z_fit = 1 / Ys

epsc = bhcubic_eps_model(epsi_med, eps_c, p)
eps_r = epsc.real
conductivity = -epsc.imag * e0 * omega
Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)

ifit.plot_dielectric_properties(
    omega,
    Zcubic,
    c0,
    Z_comp=Z_fit,
    labels=["Cubic", "MW"],
    limits=[(50, 3000), (0.05, 0.5)],
)

Rnlist = [1e-9, 0.3 * Rc, 0.5 * Rc]
for Rn in Rnlist:
    eps_c = eps_cell_double_shell_wall(
        omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw
    )
    epsi_med = emed - 1j * kmed / (e0 * omega)
    epsc = bhcubic_eps_model(epsi_med, eps_c, p)
    # epsc = eps_sus_MW(epsi_med, eps_c, p)
    eps_r = epsc.real
    conductivity = -epsc.imag * e0 * omega

    Z = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)
    label = f"{Rn / Rc:.2f}"
    if np.isclose(Rn, Rnlist[-1]):
        append = False
        show = True
    else:
        append = True
        show = False
    ifit.plot_dielectric_properties(
        omega,
        Z,
        c0,
        labels=[label, ""],
        limits=[(50, 3000), (0.05, 0.5)],
        append=append,
        show=show,
    )
