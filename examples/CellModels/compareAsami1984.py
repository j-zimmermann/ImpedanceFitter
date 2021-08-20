import numpy as np
import impedancefitter as ifit
from impedancefitter.suspensionmodels import bhcubic_eps_model, bh_eps_model, eps_sus_MW
from impedancefitter.single_shell import eps_cell_single_shell
from impedancefitter.double_shell import eps_cell_double_shell
from scipy.constants import epsilon_0 as e0

em = 10.
Rc = 0.5e-6
dm = 7e-9
dn = 7e-9
km = 1e-6
kcp = 1.0
ecp = 80
kmed = 1.
emed = 80.
c0 = 1e-12
p = 0.3
kne = 1e-6
knp = 1.0
enp = 80.
ene = 10.
doc = 0
Rn = Rc - doc

freq = np.logspace(3, 9, num=100)
omega = 2. * np.pi * freq

epscDS_MW = eps_cell_double_shell(omega, km, em, kcp, ecp, ene, kne, knp, enp, dm, Rc, dn, Rn)
eps_cMW = eps_cell_single_shell(omega, em, km, kcp, ecp, dm, Rc)
epsi_med = emed - 1j * kmed / (e0 * omega)
esus = eps_sus_MW(epsi_med, eps_cMW, p)
esusDS = eps_sus_MW(epsi_med, epscDS_MW, p)
Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
Z_fit = 1 / Ys
YsDS = 1j * esusDS * omega * c0  # cell suspension admittance spectrum
ZDS = 1 / YsDS

epsc = bhcubic_eps_model(omega, epsi_med, eps_cMW, p)
epsc2 = bh_eps_model(omega, epsi_med, eps_cMW, p)
eps_r = epsc.real
conductivity = -epsc.imag * e0 * omega
eps2_r = epsc2.real
conductivity2 = -epsc2.imag * e0 * omega

epscDS = bhcubic_eps_model(omega, epsi_med, epscDS_MW, p)

Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)
Zint = ifit.utils.convert_diel_properties_to_impedance(omega, eps2_r, conductivity2, c0)
ifit.plot_dielectric_properties(omega, Zcubic, c0, Z_comp=Zint, labels=["Cubic", "Integral"], logscale=None, limits=[(0, 600), (0.5, 1.1)])


ifit.plot_dielectric_properties(omega, Z_fit, c0, Z_comp=Zcubic, labels=["Maxwell-Wagner", "Hanai"], logscale=None, limits=[(0, 600), (0.5, 1.1)])

Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity - conductivity[0], c0)
epsMW_r = esus.real
conductivityMW = -esus.imag * e0 * omega
Z_fit = ifit.utils.convert_diel_properties_to_impedance(omega, epsMW_r, conductivityMW - conductivityMW[0], c0)
ifit.plot_cole_cole(omega, Z_fit, c0, Z_comp=Zcubic, labels=["Maxwell-Wagner", "Hanai"], limits=[(0, 600), (0, 600)])
print("done")
