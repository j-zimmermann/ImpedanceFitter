import numpy as np
import pandas as pd
from collections import OrderedDict
from impedancefitter import get_equivalent_circuit_model, PostProcess, Fitter

f = np.logspace(1, 8)
omega = 2. * np.pi * f
R = 1000.
C = 1e-6

data = OrderedDict()
data['f'] = f

samples = 500

model = "parallel(R, C)"
m = get_equivalent_circuit_model(model)
for i in range(samples):
    Ri = 0.05 * R * np.random.randn() + R
    Ci = 0.05 * C * np.random.randn() + C

    Z = m.eval(omega=omega, R=Ri, C=Ci)
    Z += np.random.randn(Z.size)

    data['real' + str(i)] = Z.real
    data['imag' + str(i)] = Z.imag
pd.DataFrame(data=data).to_csv('test.csv', index=False)

fitter = Fitter('CSV', LogLevel='WARNING')
parameters = {'R': {'value': R},
              'C': {'value': C}}
fitter.run(model, parameters=parameters)

postp = PostProcess(fitter.fit_data)
postp.plot_histograms()

print(postp.best_model_bic('R', ['Normal', 'Beta', 'Gamma']))
print(postp.best_model_chisquared('R', ['Normal', 'Beta', 'Gamma']))
print(postp.best_model_kolmogorov('R', ['Normal', 'Beta', 'Gamma']))
print("Normal(mu = {}, sigma = {}".format(R, 0.05 * R))
