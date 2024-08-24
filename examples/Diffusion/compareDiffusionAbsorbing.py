import impedancefitter as ifit
import numpy as np

Rw = 1.00
gamma = 0.8
wd = 1

freq = np.logspace(-3, 4, num=100)
omega = 2.0 * np.pi * freq

models = ["Ws", "ADIaA", "ADIbA", "ADIIA"]

for model in models:
    ecm = ifit.get_equivalent_circuit_model(model)
    Z = ecm.eval(omega=omega, Rw=Rw, gamma=gamma, wd=wd)
    ifit.plot_impedance(omega, Z, Nyquist_conventional=True, title=model)
