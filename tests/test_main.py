import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from impedancefitter import Fitter, get_equivalent_circuit_model

R = 1000.0
C = 1e-6
model = "parallel(R, C)"

Rc = 5.3e-6
dm = 7e-9
ecp = 60
kcp = 0.5
p = 0.2
km = 5e-11
em = 7.9
p = 0.1
kmed = 0.1
emed = 80
c0 = 1

k = 1e-7
alpha = 0.9
el = 1e3
tau = 1
a = 0.95
sigma = kmed
eh = 80
model1 = "ColeCole + CPE"
model2 = "SingleShell + CPE"


@pytest.fixture
def fitter():
    """Initialising fitter from CSV."""
    f = np.logspace(1, 8)
    omega = 2.0 * np.pi * f

    data = OrderedDict()
    data["f"] = f

    samples = 5

    m = get_equivalent_circuit_model(model)
    for i in range(samples):
        Ri = 0.05 * R * np.random.randn() + R
        Ci = 0.05 * C * np.random.randn() + C

        Z = m.eval(omega=omega, R=Ri, C=Ci)
        Z += np.random.randn(Z.size)

        data["real" + str(i)] = Z.real
        data["imag" + str(i)] = Z.imag
    pd.DataFrame(data=data).to_csv("test.csv", index=False)

    fitter = Fitter("CSV", LogLevel="WARNING")
    os.remove("test.csv")
    return fitter


@pytest.fixture
def fitter_df():
    """Initialising fitter from DataFrame."""
    f = np.logspace(1, 8)
    omega = 2.0 * np.pi * f

    data = OrderedDict()
    data["f"] = f

    samples = 5

    m = get_equivalent_circuit_model(model)
    for i in range(samples):
        Ri = 0.05 * R * np.random.randn() + R
        Ci = 0.05 * C * np.random.randn() + C

        Z = m.eval(omega=omega, R=Ri, C=Ci)
        Z += np.random.randn(Z.size)

        data["real" + str(i)] = Z.real
        data["imag" + str(i)] = Z.imag

    df = pd.DataFrame(data=data)
    fitter = Fitter(
        "DF", df=df, df_freq_column="f", df_real_column="real", df_imag_column="imag"
    )
    return fitter


def test_run(fitter):
    """Example run of main fitting routine."""
    parameters = {"R": {"value": R}, "C": {"value": C}}

    fitter.run(model, parameters=parameters)
    assert hasattr(fitter, "fit_data")


@pytest.fixture
def fitter2():
    """Initialising fitter from CSV data."""
    f = np.logspace(1, 8)
    omega = 2.0 * np.pi * f

    data = OrderedDict()
    data["f"] = f

    samples = 5

    m = get_equivalent_circuit_model(model2)
    for i in range(samples):
        km1 = 0.05 * km * np.random.randn() + km
        em1 = 0.05 * em * np.random.randn() + em

        Z = m.eval(
            omega=omega,
            Rc=Rc,
            dm=dm,
            km=km1,
            em=em1,
            ecp=ecp,
            kcp=kcp,
            kmed=kmed,
            emed=emed,
            p=p,
            c0=c0,
            k=k,
            alpha=alpha,
        )
        Z += np.random.randn(Z.size)

        data["real" + str(i)] = Z.real
        data["imag" + str(i)] = Z.imag
    pd.DataFrame(data=data).to_csv("test.csv", index=False)

    fitter = Fitter("CSV", LogLevel="WARNING")
    os.remove("test.csv")
    return fitter


@pytest.fixture
def fitter2_df():
    """Initialising fitter from DataFrame."""
    f = np.logspace(1, 8)
    omega = 2.0 * np.pi * f

    data = OrderedDict()
    data["f"] = f

    samples = 5

    m = get_equivalent_circuit_model(model2)
    for i in range(samples):
        km1 = 0.05 * km * np.random.randn() + km
        em1 = 0.05 * em * np.random.randn() + em

        Z = m.eval(
            omega=omega,
            Rc=Rc,
            dm=dm,
            km=km1,
            em=em1,
            ecp=ecp,
            kcp=kcp,
            kmed=kmed,
            emed=emed,
            p=p,
            c0=c0,
            k=k,
            alpha=alpha,
        )
        Z += np.random.randn(Z.size)

        data["real" + str(i)] = Z.real
        data["imag" + str(i)] = Z.imag

    df = pd.DataFrame(data=data)
    fitter = Fitter(
        "DF", df=df, df_freq_column="f", df_real_column="real", df_imag_column="imag"
    )
    return fitter
