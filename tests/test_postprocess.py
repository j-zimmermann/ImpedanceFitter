import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from openturns import Distribution, TestResult

from impedancefitter import Fitter, PostProcess, get_equivalent_circuit_model


@pytest.fixture
def postprocessRC():
    """Initialise postprocessing on RC circuit data."""
    f = np.logspace(1, 8)
    omega = 2.0 * np.pi * f
    R = 1000.0
    C = 1e-6

    data = OrderedDict()
    data["f"] = f

    samples = 50

    model = "parallel(R, C)"
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
    parameters = {"R": {"value": R}, "C": {"value": C}}
    fitter.run(model, parameters=parameters)
    os.remove("test.csv")
    return PostProcess(fitter.fit_data)


def test_bic(postprocessRC):
    """Fit best distribution."""
    model, best_res = postprocessRC.best_model_bic("R", ["Normal", "Beta", "Gamma"])
    assert isinstance(model, Distribution) and isinstance(best_res, float)


def test_chisquared(postprocessRC):
    """Fit best distribution."""
    model, best_res = postprocessRC.best_model_chisquared(
        "R", ["Normal", "Beta", "Gamma"]
    )
    assert isinstance(model, Distribution) and isinstance(best_res, TestResult)


def test_kolmogorov(postprocessRC):
    """Fit best distribution."""
    model, best_res = postprocessRC.best_model_kolmogorov(
        "R", ["Normal", "Beta", "Gamma"]
    )
    assert isinstance(model, Distribution) and isinstance(best_res, TestResult)


def test_lilliefors(postprocessRC):
    """Fit best distribution."""
    model, best_res = postprocessRC.best_model_lilliefors(
        "R", ["Normal", "Beta", "Gamma"]
    )
    assert isinstance(model, Distribution) and isinstance(best_res, TestResult)


def test_fit_normal(postprocessRC):
    """Fit to normal distribution."""
    model = postprocessRC.fit_to_normal_distribution("R")
    assert isinstance(model, Distribution)


def test_fit_histogram(postprocessRC):
    """Fit to histogram distribution."""
    model = postprocessRC.fit_to_histogram_distribution("R")
    assert isinstance(model, Distribution)
