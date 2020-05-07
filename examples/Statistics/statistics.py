import numpy as np
import os
import pandas as pd
from collections import OrderedDict
from impedancefitter import get_equivalent_circuit_model, PostProcess, Fitter

# parameters
f = np.logspace(1, 8)
omega = 2. * np.pi * f
R = 1000.
C = 1e-6

data = OrderedDict()
data['f'] = f

samples = 1000

model = "parallel(R, C)"
m = get_equivalent_circuit_model(model)
# generate random samples
for i in range(samples):
    Ri = 0.05 * R * np.random.randn() + R
    Ci = 0.05 * C * np.random.randn() + C

    Z = m.eval(omega=omega, R=Ri, C=Ci)
    # add some noise
    Z += np.random.randn(Z.size)

    data['real' + str(i)] = Z.real
    data['imag' + str(i)] = Z.imag

# save data to file
pd.DataFrame(data=data).to_csv('test.csv', index=False)

# initialize fitter
# LogLevel should be WARNING; otherwise there
# will be a lot of output


fitter = Fitter('CSV', LogLevel='WARNING')
os.remove('test.csv')
parameters = {'R': {'value': R},
              'C': {'value': C}}
fitter.run(model, parameters=parameters)

postp = PostProcess(fitter.fit_data)
# show histograms
postp.plot_histograms()

# compare different algorithms to find matching model
print("BIC best model for R:",
      postp.best_model_bic('R', ['Normal', 'Beta', 'Gamma'])[0])
print("Chisquared best model for R:",
      postp.best_model_chisquared('R', ['Normal', 'Beta', 'Gamma'])[0])
print("Kolmogorov best model for R:",
      postp.best_model_kolmogorov('R', ['Normal', 'Beta', 'Gamma'])[0])
print("Expected result:\nNormal(mu = {}, sigma = {})".format(R, 0.05 * R))
