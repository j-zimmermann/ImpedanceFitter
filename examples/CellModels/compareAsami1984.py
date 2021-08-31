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
ene = 10.
knp = 1.0
enp = 80.
doc = 0.0 # Rc 
Rn = Rc - dm - doc

freq = np.logspace(4, 9, num=100)
omega = 2. * np.pi * freq

# cell permittivities
epsDS_c = eps_cell_double_shell(omega, km, em, kcp, ecp, ene, kne, knp, enp, dm, Rc, dn, Rn)
eps_c = eps_cell_single_shell(omega, em, km, kcp, ecp, dm, Rc)
epsi_med = emed - 1j * kmed / (e0 * omega)
esus = eps_sus_MW(epsi_med, eps_c, p)
esusDS = eps_sus_MW(epsi_med, epsDS_c, p)
Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
Z_fit = 1 / Ys
YsDS = 1j * esusDS * omega * c0  # cell suspension admittance spectrum
ZDS = 1 / YsDS

epsc = bhcubic_eps_model(epsi_med, eps_c, p)
epsc2 = bh_eps_model(epsi_med, eps_c, p)
eps_r = epsc.real
conductivity = -epsc.imag * e0 * omega
eps2_r = epsc2.real
conductivity2 = -epsc2.imag * e0 * omega

epscDS = bhcubic_eps_model(epsi_med, epsDS_c, p)
epsDS_r = epscDS.real
conductivityDS = -epscDS.imag * e0 * omega


Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)
ZSS = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity, c0)
Zint = ifit.utils.convert_diel_properties_to_impedance(omega, eps2_r, conductivity2, c0)
ZDSH = ifit.utils.convert_diel_properties_to_impedance(omega, epsDS_r, conductivityDS, c0)
ifit.plot_dielectric_properties(omega, Zcubic, c0, Z_comp=Zint, labels=["Cubic", "Integral"], logscale=None, limits=[(0, 600), (0.5, 1.1)])
ifit.plot_dielectric_properties(omega, Z_fit, c0, Z_comp=Zcubic, labels=["Maxwell-Wagner", "Hanai"], logscale=None, limits=[(0, 600), (0.5, 1.1)])
ifit.plot_dielectric_properties(omega, ZDSH, c0, Z_comp=Zcubic, labels=["Double shell", "Single shell"], logscale=None, limits=[(0, 600), (0.5, 1.1)])

Zcubic = ifit.utils.convert_diel_properties_to_impedance(omega, eps_r, conductivity - conductivity[0], c0)
epsMW_r = esus.real
conductivityMW = -esus.imag * e0 * omega
Z_fit = ifit.utils.convert_diel_properties_to_impedance(omega, epsMW_r, conductivityMW - conductivityMW[0], c0)
ifit.plot_cole_cole(omega, Z_fit, c0, Z_comp=Zcubic, labels=["Maxwell-Wagner", "Hanai"], limits=[(0, 600), (0, 600)])
print("done")

doclist = [0, 5e-9, 20e-9, 80e-9]

for doc in doclist:
    Rn = Rc - dm - doc
    epsDS_c = eps_cell_double_shell(omega, km, em, kcp, ecp, ene, kne, knp, enp, dm, Rc, dn, Rn)
    epscDS = bhcubic_eps_model(epsi_med, epsDS_c, p)
    epsDS_r = epscDS.real
    conductivityDS = -epscDS.imag * e0 * omega
    ZDSH = ifit.utils.convert_diel_properties_to_impedance(omega, epsDS_r, conductivityDS, c0)
    label = "{:.0f}".format(doc / 1e-9)
    ifit.plot_dielectric_properties(omega, ZDSH, c0, labels=[label, ""], logscale=None, limits=[(0, 600), (0.5, 1.1)], append=True, show=False)
    if np.isclose(doc, doclist[-1]):
        ifit.plot_dielectric_properties(omega, ZSS, c0, labels=["Single shell", ""], logscale=None, limits=[(0, 600), (0.5, 1.1)])
