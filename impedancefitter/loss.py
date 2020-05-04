def Z_loss(omega, L, C, R):
    """Impedance for high loss materials, where LCR are in parallel.

    Described for instance in [5]_.

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
    .. [5] Kordzadeh, A., & De Zanche, N. (2016).
           Permittivity measurement of liquids, powders, and suspensions
           using a parallel-plate cell.
           Concepts in Magnetic Resonance Part B: Magnetic Resonance Engineering,
           46(1), 19–24. https://doi.org/10.1002/cmr.b.21318
    """
    Y = 1. / R + 1. / (1j * omega * L) + 1j * omega * C
    Z = 1. / Y
    return Z


def Z_in(omega, L, R):
    """Lead inductance of wires connecting DUT.

    Described for instance in [6]_.

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
    .. [6] Kordzadeh, A., & De Zanche, N. (2016).
           Permittivity measurement of liquids, powders, and suspensions
           using a parallel-plate cell.
           Concepts in Magnetic Resonance Part B: Magnetic Resonance Engineering,
           46(1), 19–24. https://doi.org/10.1002/cmr.b.21318

    """
    return R + 1j * omega * L
