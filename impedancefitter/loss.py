def Z_loss(omega, L, C, R):
    """Impedance for high loss materials, where LCR are in parallel.

    Described for instance in [Kordzadeh2016]_.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    L: double
        inductance
    C: double
        capacitance
    R: double
        resistance

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    References
    ----------
    .. [Kordzadeh2016] Kordzadeh, A., & De Zanche, N. (2016).
           Permittivity measurement of liquids, powders, and suspensions
           using a parallel-plate cell.
           Concepts in Magnetic Resonance Part B: Magnetic Resonance Engineering,
           46(1), 19–24. https://doi.org/10.1002/cmr.b.21318

    Notes
    -----

    As mentioned in [Kordzadeh2016]_, the unit of the
    capacitance is pF and the unit of the inductance
    is nH.
    """
    L *= 1e-9
    C *= 1e-12
    Y = 1. / R + 1. / (1j * omega * L) + 1j * omega * C
    Z = 1. / Y
    return Z


def Z_in(omega, L, R):
    """Lead inductance of wires connecting DUT.

    Described for instance in [Kordzadeh2016]_.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    L: double
        inductance
    C: double
        capacitance
    R: double
        resistance

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    References
    ----------
    .. [Kordzadeh2016] Kordzadeh, A., & De Zanche, N. (2016).
           Permittivity measurement of liquids, powders, and suspensions
           using a parallel-plate cell.
           Concepts in Magnetic Resonance Part B: Magnetic Resonance Engineering,
           46(1), 19–24. https://doi.org/10.1002/cmr.b.21318

    Notes
    -----

    As mentioned in [Kordzadeh2016]_, the unit of the
    inductance is nH.

    """
    L *= 1e-9
    return R + 1j * omega * L
