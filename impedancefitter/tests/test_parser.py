from impedancefitter.utils import get_comp_model
from lmfit import CompositeModel


def test_parallel():
    model1 = "parallel(R, C)"
    model = get_comp_model(model1)
    assert isinstance(model, CompositeModel)


def test_series():
    model1 = "R + C"
    model = get_comp_model(model1)
    assert isinstance(model, CompositeModel)


def test_series_parallel():
    model1 = "R + L + parallel(ColeCole, C)"
    model = get_comp_model(model1)
    assert isinstance(model, CompositeModel)


def test_parallel_series():
    model1 = "parallel(L, C) + R + ColeCole"
    model = get_comp_model(model1)
    assert isinstance(model, CompositeModel)


def test_parallel_parallel():
    model1 = "parallel(parallel(R_f1, C_f1), C)"
    model = get_comp_model(model1)
    assert isinstance(model, CompositeModel)
