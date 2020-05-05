from .elements import Z_CPE, Z_w, parallel


def cpe_model(omega, k, alpha):
    r"""Constant Phase Element.

    .. math::

        Z_\mathrm{CPE} = k^{-1} (j \omega)^{-\alpha}

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    k: double
        CPE factor
    alpha: double
        CPE phase
    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array
    """
    Z_fit = Z_CPE(omega, k, alpha)
    return Z_fit


def cpe_ct_model(omega, k, alpha, Rct):
    """Constant Phase Element in parallel with charge transfer resistance.


    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    k: double
        CPE factor
    alpha: double
        CPE phase
    Rct: double
        charge transfer resistance
    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    See Also
    --------
    :meth:`impedancefitter.cpe.cpe_model`

    """

    Z_fit = parallel(Z_CPE(omega, k, alpha), Rct)
    return Z_fit


def cpe_ct_w_model(omega, k, alpha, Rct, Aw):
    """Constant Phase Element in parallel with
       charge transfer resistance, which is in series with Warburg element.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    k: double
        CPE factor
    alpha: double
        CPE phase
    Rct: double
        charge transfer resistance
    Aw: double
        Warburg coefficient


    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    See Also
    --------
    :meth:`impedancefitter.cpe.cpe_model`

    """
    Z_par = Rct + Z_w(omega, Aw)
    Z_fit = parallel(Z_CPE(omega, k, alpha), Z_par)
    return Z_fit
