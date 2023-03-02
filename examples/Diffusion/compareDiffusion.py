import numpy as np
import impedancefitter as ifit

Rw = 1.00
gamma = 0.8
wd = 1

freq = np.logspace(-0.75, 3, num=100)
omega = 2. * np.pi * freq

model = "ADIbR"

ecm = ifit.get_equivalent_circuit_model(model)
Z = ecm.eval(omega=omega, Rw=Rw, gamma=gamma, wd=wd)
ifit.plot_impedance(omega, Z, Nyquist_conventional=True)
