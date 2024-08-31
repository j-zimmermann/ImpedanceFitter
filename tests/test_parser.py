import numpy as np
import pytest
from impedancefitter.utils import get_equivalent_circuit_model
from lmfit import CompositeModel, Model


def test_single():
    """Test single resistor."""
    model1 = "R"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, Model)


def test_single2():
    """Test Randles model."""
    model1 = "Randles"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, Model)


def test_parallel():
    """Test parallel RC."""
    model1 = "parallel(R, C)"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_series():
    """Test RC in series."""
    model1 = "R + C"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_series_parallel():
    """Test mixed circuit."""
    model1 = "R + L + parallel(ColeCole, C)"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_parallel_series():
    """Test mixed circuit."""
    model1 = "parallel(L, C) + R + ColeCole"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_parallel_parallel():
    """Test nested parallel circuit."""
    model1 = "parallel(parallel(R_f1, C_f1), C_f2)"
    model = get_equivalent_circuit_model(model1)
    assert isinstance(model, CompositeModel)


def test_logscale():
    """Test log-scale conversion."""
    model1 = "parallel(parallel(R_f1, C_f1), C_f2)"
    model = get_equivalent_circuit_model(model1)
    logmodel = get_equivalent_circuit_model(model1, logscale=True)
    omega = np.logspace(0, 8)
    parameters = {"f1_R": 100.0, "f1_C": 1e-6, "f2_C": 1e-6}
    Z1 = model.eval(omega=omega, **parameters)
    Z2 = logmodel.eval(omega=omega, **parameters)
    assert np.all(np.isclose(np.log10(Z1), Z2))


def test_wrong_circuit():
    """Test wrong circuit."""
    model1 = "R, L , C"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit2():
    """Test wrong circuit."""
    model1 = "parallel(R + L + C)"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit3():
    """Test wrong circuit."""
    model1 = "parallel(R, C"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit4():
    """Test wrong circuit."""
    model1 = "parallel(R + L)"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit5():
    """Test wrong circuit."""
    model1 = "parallel(R_f1 + parallel(R_f2, L))"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit6():
    """Test wrong circuit."""
    model1 = "R_f1 + parallel(R_f2 + L)"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit7():
    """Test wrong circuit."""
    model1 = "R + parallel(R_f2, L)"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)


def test_wrong_circuit8():
    """Test wrong circuit."""
    model1 = "R_f1 + parallel(R, L)"
    with pytest.raises(Exception):
        get_equivalent_circuit_model(model1)
