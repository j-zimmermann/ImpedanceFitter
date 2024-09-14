import numpy as np


def weighting_residual(params, omega, Zdata=None, model=None, model_kwargs={}):
    """Weight data using a model for the error.

    Parameters
    ----------
    params: dict
        Dictionary with parameters.
        Contains the entries for `stdA` and `stdPhi`, which are
        the standard deviations of the impedance magnitude and phase.
        They should be given in percent.
    omega: :class:`numpy.ndarray`, double
        frequency array
    Zdata: :class:`numpy.ndarray`, complex
        impedance data
    model: :py:class:`lmfit.model.CompositeModel` or :class:`lmfit.model.Model`
        impedance model
    model_kwargs: dict
        keyword arguments passed to model

    Notes
    -----
    The weighting model for the real and imaginary part is
    described in [Ciucci2013]_.

    [Ciucci2013] Ciucci, F. (2013).
                 Revisiting parameter identification in electrochemical
                 impedance spectroscopy: Weighted least squares and
                 optimal experimental design.
                 Electrochimica Acta, 87, 532–545.
                 https://doi.org/10.1016/j.electacta.2012.09.073
    """
    # convert from percent
    stdA = 1e-2 * params["stdA"]
    stdPhi = 1e-2 * params["stdPhi"]
    if not omega.size == Zdata.size:
        raise ValueError("Frequency and impedance array must have same length")
    # standard deviation of real and imaginary part
    stdR = np.abs(Zdata) * stdA
    stdI = np.abs(Zdata) * (stdA + stdPhi) * np.angle(Zdata)
    weights = 1.0 / stdR**2 + 1j / stdI**2
    Z1 = model.eval(omega=omega, params=params, **model_kwargs)
    diff = Z1 - Zdata
    diff = diff.ravel().view(float)
    weights = weights.ravel().view(float)

    diff = np.sum(np.log(1.0 / weights) + weights * diff * diff)
    return diff


def variance_estimate(Zdata, Zmodel, n=2):
    """Variance estimates.

    Parameters
    ----------
    Zdata: :class:`numpy.ndarray`, complex
        impedance data

    Zmodel: :class:`numpy.ndarray`, complex
        impedance model

    n: int
        number between 0 and 2.

    """
    P = len(Zdata)
    varA = np.sum(((np.abs(Zdata) - np.abs(Zmodel)) / np.abs(Zmodel)) ** 2) / (P - n)
    varPhi = np.sum(((np.angle(Zdata) - np.angle(Zmodel)) / np.angle(Zmodel)) ** 2) / (
        P - n
    )
    return varA, varPhi
