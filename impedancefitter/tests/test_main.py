import os
import pytest
import numpy as np
import pandas as pd
from collections import OrderedDict
from impedancefitter import Fitter, get_equivalent_circuit_model


R = 1000.
C = 1e-6
model = "parallel(R, C)"


@pytest.fixture
def fitter():
    f = np.logspace(1, 8)
    omega = 2. * np.pi * f

    data = OrderedDict()
    data['f'] = f

    samples = 500

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
    os.remove('test.csv')
    return fitter


def test_run(fitter):
    parameters = {'R': {'value': R},
                  'C': {'value': C}}

    fitter.run(model, parameters=parameters)
    assert hasattr(fitter, "fit_data")


def test_sequential_run(fitter2):
    parameters1 = {'R': {'value': R},
                   'C': {'value': C}}

    fitter2.sequential_run(model, parameters1=parameters1)
    assert hasattr(fitter, "fit_data")
