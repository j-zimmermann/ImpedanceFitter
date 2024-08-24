import numpy as np
from impedancefitter.single_shell import eps_cell_single_shell
from impedancefitter.suspensionmodels import bh_eps_model, bhcubic_eps_model
from scipy.constants import epsilon_0 as e0

em = 10.0
Rc = 0.5e-6
dm = 7e-9
dn = 7e-9
km = 1e-6
kcp = 1.0
ecp = 80
kmed = 1.0
emed = 80.0
c0 = 1e-12
p = 0.3
kne = 1e-6
ene = 10.0
knp = 1.0
enp = 80.0
doc = 0.0  # Rc
Rn = Rc - dm - doc

freq = np.logspace(4, 9, num=100)
omega = 2.0 * np.pi * freq

# cell permittivities
eps_c = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
epsi_med = emed - 1j * kmed / (e0 * omega)


def test_solver_equality():
    """Test that cubic solvers yield the same."""
    # integral and cubic equation solver
    epsc = bh_eps_model(epsi_med, eps_c, p)
    epsc2 = bhcubic_eps_model(epsi_med, eps_c, p)

    eps_r = epsc.real
    conductivity = -epsc.imag * e0 * omega
    eps2_r = epsc2.real
    conductivity2 = -epsc2.imag * e0 * omega
    assert np.all(np.isclose(eps_r, eps2_r, rtol=1e-2))
    assert np.all(np.isclose(conductivity, conductivity2, rtol=1e-2))
