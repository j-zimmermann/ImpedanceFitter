import numpy as np
import impedancefitter as ifit
from impedancefitter.suspensionmodels import bhcubic_eps_model, bh_eps_model, eps_sus_MW
from impedancefitter.single_shell import eps_cell_single_shell
from scipy.constants import epsilon_0 as e0

em = 6.5
Rc = 3.8e-6 / 2.
dm = 5e-9
km = 0.0
kcp = 0.25
ecp = 50
kmed = 0.25
emed = 80.
c0 = 1e-12
p = 0.9

freq = np.logspace(3, 9, num=100)
omega = 2. * np.pi * freq


epsi_med = emed - 1j * kmed / (e0 * omega)

eps_cMW = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
epsc = bhcubic_eps_model(epsi_med, eps_cMW, p)

epsc2 = bh_eps_model(epsi_med, eps_cMW, p)
eps_r = epsc.real
conductivity = -epsc.imag * e0 * omega
eps2_r = epsc2.real
conductivity2 = -epsc2.imag * e0 * omega

Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)
Zint = ifit.utils.convert_diel_properties_to_impedance(omega, eps2_r, conductivity2, c0)
ifit.plot_dielectric_properties(omega, Zcubic, c0, Z_comp=Zint, labels=["Cubic", "Integral"])

esus = eps_sus_MW(epsi_med, eps_cMW, p)
Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
Z_fit = 1 / Ys

ifit.plot_dielectric_properties(omega, Z_fit, c0, Z_comp=Zcubic, labels=["Maxwell-Wagner", "Hanai"])
ifit.plot_dielectric_dispersion(omega, Z_fit, c0, Z_comp=Zcubic, labels=["Maxwell-Wagner", "Hanai"])

Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity - conductivity[0], c0)
epsMW_r = esus.real
conductivityMW = -esus.imag * e0 * omega
Z_fit = ifit.utils.convert_diel_properties_to_impedance(omega, epsMW_r, conductivityMW - conductivityMW[0], c0)
ifit.plot_cole_cole(omega, Z_fit, c0, Z_comp=Zcubic, labels=["Maxwell-Wagner", "Hanai"], limits=[(0, 4000), (0, 2000)])
print("done")
p = 0.1
while p < 0.95:
    print(p)
    eps_cMW = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    epsc = bhcubic_eps_model(epsi_med, eps_cMW, p)
    eps_r = epsc.real
    conductivity = -epsc.imag * e0 * omega
    Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity - conductivity[0], c0)
    if np.isclose(p, 0.9):
        ifit.plot_cole_cole(omega, Zcubic, c0, labels=["{:.1f}".format(p), "{:.1f} MW".format(p)], limits=[(0, 4000), (0, 2000)])
    else:
        ifit.plot_cole_cole(omega, Zcubic, c0, labels=["{:.1f}".format(p), "{:.1f} MW".format(p)], limits=[(0, 4000), (0, 2000)], append=True, show=False)
    p += 0.1
p = 0.05
while p < 0.95:
    print(p)
    eps_cMW = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    epsc = bhcubic_eps_model(epsi_med, eps_cMW, p)
    eps_r = epsc.real
    conductivity = -epsc.imag * e0 * omega
    Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity - conductivity[0], c0)
    Zcubicfull = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)
    esus = eps_sus_MW(epsi_med, eps_cMW, p)
    epsMW_r = esus.real
    conductivityMW = -esus.imag * e0 * omega
    Z_fit = ifit.utils.convert_diel_properties_to_impedance(omega, epsMW_r, conductivityMW - conductivityMW[0], c0)
    Z_fitfull = ifit.utils.convert_diel_properties_to_impedance(omega, epsMW_r, conductivityMW, c0)
    if np.isclose(p, 0.9):
        ifit.plot_cole_cole(omega, Zcubic, c0, Z_comp=Z_fit, labels=["{:.2f}".format(p), "{:.2f} MW".format(p)], limits=[(0, 4000), (0, 2000)])
        ifit.plot_dielectric_properties(omega, Z_fitfull, c0, Z_comp=Zcubicfull, labels=["{:.2f}".format(p), "{:.2f} MW".format(p)], markers=[None, "o"], markevery=0.1)
    else:
        ifit.plot_cole_cole(omega, Zcubic, c0, Z_comp=Z_fit, labels=["{:.2f}".format(p), "{:.2f} MW".format(p)], limits=[(0, 4000), (0, 2000)], append=True, show=False)
        ifit.plot_dielectric_properties(omega, Z_fitfull, c0, Z_comp=Zcubicfull, labels=["{:.2f}".format(p), "{:.2f} MW".format(p)], append=True, show=False, markers=[None, "o"], markevery=0.1)
    if np.isclose(p, 0.05):
        p = 0.3
    else:
        p += 0.3
