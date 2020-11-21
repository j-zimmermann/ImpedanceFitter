from impedancefitter.utils import get_equivalent_circuit_model
import pytest
from lmfit import CompositeModel, Model
import numpy as np


def test_single():
    model1 = 'R'
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, Model)


def test_single2():
    model1 = 'Randles'
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, Model)


def test_parallel():
    model1 = "parallel(R, C)"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_series():
    model1 = "R + C"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_series_parallel():
    model1 = "R + L + parallel(ColeCole, C)"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_parallel_series():
    model1 = "parallel(L, C) + R + ColeCole"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_parallel_parallel():
    model1 = "parallel(parallel(R_f1, C_f1), C_f2)"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_logscale():
    model1 = "parallel(parallel(R_f1, C_f1), C_f2)"
    model = get_equivalent_circuit_model(model1)
    logmodel = get_equivalent_circuit_model(model1, logscale=True)
    omega = np.logspace(0, 8)
    parameters = {'f1_R': 100.,
                  'f1_C': 1e-6,
                  'f2_C': 1e-6}
    Z1 = model.eval(omega=omega, **parameters)
    Z2 = logmodel.eval(omega=omega, **parameters)
    assert np.all(np.isclose(np.log10(Z1), Z2))


def test_wrong_circuit():
    model1 = "R, L , C"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit2():
    model1 = "parallel(R + L + C)"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit3():
    model1 = "parallel(R, C"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)
