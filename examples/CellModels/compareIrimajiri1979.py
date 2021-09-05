import numpy as np
import impedancefitter as ifit
from impedancefitter.suspensionmodels import bhcubic_eps_model, bh_eps_model, eps_sus_MW
from impedancefitter.double_shell import eps_cell_double_shell
from scipy.constants import epsilon_0 as e0

em = 9
Rc = 6.5e-6
dm = 8e-9
dn = 40e-9
km = 8e-8
kcp = 1
ecp = 77 
kmed = 1
emed = 77.
c0 = 1e-12
p = 0.1
kne = 4e-4
ene = 20.
knp = 1
enp = 300.
Rn = 2.5e-6 

freq = np.logspace(4, 8, num=100)
omega = 2. * np.pi * freq

# cell permittivities
epsDS_c = eps_cell_double_shell(omega, km, em, kcp, ecp, ene, kne, knp, enp, dm, Rc, dn, Rn)
epsi_med = emed - 1j * kmed / (e0 * omega)
esusDS = eps_sus_MW(epsi_med, epsDS_c, p)
YsDS = 1j * esusDS * omega * c0  # cell suspension admittance spectrum
ZDS = 1 / YsDS

epscDS = bhcubic_eps_model(epsi_med, epsDS_c, p)
epsDS_r = epscDS.real
conductivityDS = -epscDS.imag * e0 * omega


ZDSH = ifit.utils.convert_diel_properties_to_impedance(omega, epsDS_r, conductivityDS, c0)
ifit.plot_dielectric_properties(omega, ZDSH, c0, labels=["Double shell", "Single shell"], logscale=None, limits=[(0, 1800), (0.8, 1.1)])

Rnlist = [1e-12, 2.5e-6, 4.67e-6, 6.25e-6]

for Rn in Rnlist:
    epsDS_c = eps_cell_double_shell(omega, km, em, kcp, ecp, ene, kne, knp, enp, dm, Rc, dn, Rn)
    epscDS = bhcubic_eps_model(epsi_med, epsDS_c, p)
    epsDS_r = epscDS.real
    conductivityDS = -epscDS.imag * e0 * omega
    ZDSH = ifit.utils.convert_diel_properties_to_impedance(omega, epsDS_r, conductivityDS, c0)
    label = "{:.2f}".format(Rn / 1e-6)
    if np.isclose(Rn, Rnlist[-1]):
        ifit.plot_dielectric_properties(omega, ZDSH, c0, labels=[label, ""], logscale=None, limits=[(0, 1800), (0.8, 1.1)])
    else:
        ifit.plot_dielectric_properties(omega, ZDSH, c0, labels=[label, ""], logscale=None, limits=[(0, 1800), (0.8, 1.1)], append=True, show=False)
